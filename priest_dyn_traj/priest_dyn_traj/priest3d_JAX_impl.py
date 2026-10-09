"""PRIEST 3D with immutable configuration and explicit dynamic solve inputs.

Create one batch_crowd_nav_3d instance on the selected JAX device, then call
solve(state, reference, centers, radii, active, key). Shapes stay fixed:
state (9,), reference (way_point_shape, 3), centers/radii (num_obs, 3),
active (num_obs,), key a JAX PRNG key. Active radii must be positive.

All per-frame obstacle data flows through function arguments. self contains
only setup-time configuration and constant matrices; do not mutate it after
warm-up. Internal helpers are ordinary JAX functions traced by solve's JIT.
This interface replaces the old constructor that accepted obstacle radii.
"""

import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from jax import jit, lax

from . import bernstein_coeff_order10_arbitinterval

class batch_crowd_nav_3d:
    """
    3D extension of the PRIEST-style batch projection + CEM planner.

    State:
        [x, y, z, vx, vy, vz, ax, ay, az]

    Obstacles:
        axis-aligned ellipsoids
        ((x-xo)/a_i)^2 + ((y-yo)/b_i)^2 + ((z-zo)/c_i)^2 >= 1

    Notes:
    - This is a 3D generalization of the original 2D implementation.
    - Instead of 2D polar angles, projection uses a 3D unit-direction
      representation. This avoids singular spherical-angle bookkeeping.
    - Each obstacle may have its own a_i, b_i, c_i.
    """

    def __init__(
        self,
        v_max,
        v_min,
        a_max,
        num_obs,
        t_fin,
        num,
        num_batch,
        maxiter,
        maxiter_cem,
        weight_smoothness,
        weight_track,
        way_point_shape,
        v_des,
        goal_clearance=0.08,
        goal_lateral_weight=0.0,
    ):
        # Configuration is fixed for the lifetime of this compiled planner.
        self.goal_clearance = goal_clearance
        self.goal_lateral_weight = goal_lateral_weight
        self.maxiter = maxiter
        self.maxiter_cem = maxiter_cem

        self.weight_smoothness = weight_smoothness
        self.weight_track = weight_track

        self.v_des = v_des
        self.v_max = v_max
        self.v_min = v_min
        self.a_max = a_max

        self.t_fin = t_fin
        self.num = num
        self.t = t_fin / num

        self.num_batch = num_batch
        self.ellite_num_const = min(80, num_batch)
        self.ellite_num = min(20, self.ellite_num_const)

        self.num_obs = num_obs
        self.num_obs_proj = min(30, num_obs)

        self.way_point_shape = way_point_shape

        # ---------------------------------------------------------
        # Bernstein basis
        # ---------------------------------------------------------
        tot_time = np.linspace(0.0, t_fin, num)
        self.tot_time = jnp.asarray(tot_time)

        t_col = tot_time.reshape(num, 1)

        P, Pdot, Pddot = (
            bernstein_coeff_order10_arbitinterval
            .bernstein_coeff_order10_new(
                10,
                t_col[0],
                t_col[-1],
                t_col,
            )
        )

        self.P = jnp.asarray(P)
        self.Pdot = jnp.asarray(Pdot)
        self.Pddot = jnp.asarray(Pddot)

        self.P_jax = self.P
        self.Pdot_jax = self.Pdot
        self.Pddot_jax = self.Pddot

        self.nvar = self.P.shape[1]

        # Boundary constraints:
        # position(0), velocity(0), acceleration(0), position(T)
        self.A_eq = jnp.vstack(
            (
                self.P[0],
                self.Pdot[0],
                self.Pddot[0],
                self.P[-1],
            )
        )

        self.A_projection = jnp.eye(self.nvar)
        self.A_vel = self.Pdot
        self.A_acc = self.Pddot
        self.A_obs = jnp.tile(
            self.P,
            (self.num_obs_proj, 1),
        )

        # ---------------------------------------------------------
        # Optimization parameters
        # ---------------------------------------------------------
        self.rho_obs = 1.0
        self.rho_ineq = 1.0
        self.rho_track = 1.0
        self.rho_proj = 1.0

        self.maxitet_proj = 13

        self.num_sample = 6
        self.num_sample_warm = self.num_batch // 2

        self.beta = 4.0
        self.lamda = 0.9
        self.alpha = 0.7

        self.initial_up_sampling = 30

        # ---------------------------------------------------------
        # Safe-goal search configuration
        #
        # The safe-goal projection is performed inside this planner
        # with fixed-shape JAX arrays inside the public compiled solve.
        # ---------------------------------------------------------
        
        self.GOAL_BOUNDARY_MARGIN = 1e-3
        self.goal_search_num_directions = 1200
        self.goal_search_num_shells = 14
        self.goal_search_max_scale = 4.0

        i_dir = np.arange(
            self.goal_search_num_directions,
            dtype=float,
        )
        z_dir = (
            1.0
            - 2.0
            * (i_dir + 0.5)
            / self.goal_search_num_directions
        )
        r_dir = np.sqrt(
            np.maximum(0.0, 1.0 - z_dir**2)
        )
        golden_angle = np.pi * (
            3.0 - np.sqrt(5.0)
        )
        phi_dir = golden_angle * i_dir

        goal_dirs = np.column_stack(
            (
                r_dir * np.cos(phi_dir),
                r_dir * np.sin(phi_dir),
                z_dir,
            )
        )

        self.goal_search_directions = jnp.asarray(
            goal_dirs
        )

        # Fit arbitrary sampled position trajectories to Bernstein coeffs.
        self.cost_fit = (
            self.P.T @ self.P
            + 0.0001 * jnp.eye(self.nvar)
            + self.Pddot.T @ self.Pddot
        )
        self.cost_fit_inv = jnp.linalg.inv(self.cost_fit)

        # Each active projected obstacle contributes the same P.T @ P block.
        # Precompute a small inverse table for 0..num_obs_proj active obstacles.
        # The live mask selects a table entry on-device, avoiding a matrix
        # inversion or recompilation when the obstacle count changes.
        projection_base = (
            self.rho_proj * (self.A_projection.T @ self.A_projection)
            + self.rho_ineq * (self.A_acc.T @ self.A_acc)
            + self.rho_ineq * (self.A_vel.T @ self.A_vel)
        )
        zero_eq = jnp.zeros((self.A_eq.shape[0], self.A_eq.shape[0]))

        def inverse_for_count(count):
            hessian = (
                projection_base
                + self.rho_obs * count * (self.P.T @ self.P)
            )
            kkt = jnp.block([
                [hessian, self.A_eq.T],
                [self.A_eq, zero_eq],
            ])
            return jnp.linalg.inv(kkt)

        self.projection_inverse_by_count = jax.vmap(inverse_for_count)(
            jnp.arange(self.num_obs_proj + 1, dtype=self.P.dtype)
        )

        # ---------------------------------------------------------
        # Random 3D normal-plane offsets for initial trajectory set
        # ---------------------------------------------------------
        rng = np.random.default_rng(2)

        sample_count = self.num_batch * self.initial_up_sampling

        self.warm_u = jnp.asarray(
            rng.normal(0.0, 0.8, self.num_sample_warm)
        )
        self.warm_v = jnp.asarray(
            rng.normal(0.0, 0.8, self.num_sample_warm)
        )

        self.sample_u_25 = jnp.asarray(
            rng.normal(0.0, 0.8, sample_count)
        )
        self.sample_v_25 = jnp.asarray(
            rng.normal(0.0, 0.8, sample_count)
        )

        self.sample_u_50 = jnp.asarray(
            rng.normal(0.0, 0.8, sample_count)
        )
        self.sample_v_50 = jnp.asarray(
            rng.normal(0.0, 0.8, sample_count)
        )

        self.sample_u_75 = jnp.asarray(
            rng.normal(0.0, 0.8, sample_count)
        )
        self.sample_v_75 = jnp.asarray(
            rng.normal(0.0, 0.8, sample_count)
        )

        self.vec_product = jax.vmap(self.comp_prod, 0, out_axes=0)
        self.vectorized_projection = jax.vmap(
            self.compute_contouring_error,
            in_axes=(None, None, None, 0, 0, 0, None),
        )

    # =============================================================
    # Public compiled entry point. All changing arrays are explicit inputs.
    # =============================================================

    @partial(jit, static_argnums=(0,))
    def solve(
        self,
        initial_state,
        reference_waypoints,
        obstacle_centers,
        obstacle_radii,
        obstacle_active,
        key,
    ):
        """Return (positions[num, 3], safe_goal_success) on the input device.

        Call with fixed shapes/dtypes after one warm-up. This method does not
        mutate self, perform host transfers, or recreate the planner. It expects
        finite inputs and positive radii for active obstacles; the ROS adapter
        validates incoming messages. No-obstacle calls still use padded arrays.
        """
        active = jnp.asarray(obstacle_active, dtype=jnp.bool_)

        # Sort centers, radii and mask together; projection uses the nearest
        # num_obs_proj entries, while scoring considers all active obstacles.
        # Neutralize padded data before arithmetic, so NaNs in unused slots
        # cannot leak into masked costs or projections.
        centers = jnp.where(active[:, None], obstacle_centers, 0.0)
        radii = jnp.where(active[:, None], obstacle_radii, 1.0)
        distance = jnp.linalg.norm(centers - initial_state[None, :3], axis=1)
        order = jnp.argsort(jnp.where(active, distance, jnp.inf))
        centers, radii, active = centers[order], radii[order], active[order]

        xw, yw, zw = reference_waypoints.T
        _, arc, dx, dy, dz = self.path_spline(xw, yw, zw)
        gx, gy, gz, _ = self.compute_nominal_local_goal(
            *initial_state[:3], self.v_des, xw, yw, zw, arc
        )
        xf, yf, zf, goal_corrected, success = self.project_safe_goal(
            jnp.stack((gx, gy, gz)),
            centers, radii, active,
            initial_state[:3],
            normalized_clearance=self.goal_clearance,
            lateral_weight=self.goal_lateral_weight,
        )

        # Build stationary obstacle trajectories and reference-based seeds.
        xt, yt, zt = [jnp.repeat(c[:, None], self.num, axis=1) for c in centers.T]
        warm = self.compute_warm_traj(
            initial_state, self.v_des,
            xw, yw, zw, arc, dx, dy, dz, xf, yf, zf,
        )
        guess = self.compute_traj_guess(
            initial_state, xt, yt, zt, self.v_des,
            xw, yw, zw, arc, *warm, dx, dy, dz, xf, yf, zf,
            obstacle_radii=radii,
            obstacle_active=active,
        )
        cx, cy, cz, x, y, z, vx, vy, vz, ax, ay, az, mean, cov, *_ = guess
        zero = jnp.zeros((self.num_batch, self.nvar), dtype=initial_state.dtype)

        result = self.compute_cem(
            key, initial_state,
            xf, yf, zf,
            zero, zero, zero,
            xt, yt, zt,
            xt[:self.num_obs_proj], yt[:self.num_obs_proj], zt[:self.num_obs_proj],
            cx, cy, cz,
            x, y, z, vx, vy, vz, ax, ay, az,
            xw, yw, zw, arc, mean, cov,
            obstacle_radii=radii,
            obstacle_active=active,
        )
        return (
            jnp.stack(result[9:12], axis=1),
            success,
            goal_corrected,
        )

    # =============================================================
    # Reference path utilities
    # =============================================================

    def path_spline(
        self,
        x_waypoint,
        y_waypoint,
        z_waypoint,
    ):
        x_diff = jnp.diff(x_waypoint)
        y_diff = jnp.diff(y_waypoint)
        z_diff = jnp.diff(z_waypoint)

        segment = jnp.sqrt(
            x_diff**2 + y_diff**2 + z_diff**2
        )

        arc_vec = jnp.concatenate(
            (
                jnp.array([0.0]),
                jnp.cumsum(segment),
            )
        )

        arc_length = arc_vec[-1]

        return (
            arc_length,
            arc_vec,
            x_diff,
            y_diff,
            z_diff,
        )


    def compute_nominal_local_goal(
        self,
        x_init,
        y_init,
        z_init,
        v_des,
        x_waypoint,
        y_waypoint,
        z_waypoint,
        arc_vec,
    ):
        """
        Compute the normal look-ahead endpoint on the 3D reference path.

        This belongs in the planner rather than the visualization so the
        same endpoint-selection rule is used by every caller.
        """
        dist = jnp.sqrt(
            (x_waypoint - x_init)**2
            + (y_waypoint - y_init)**2
            + (z_waypoint - z_init)**2
        )

        index = jnp.argmin(dist)
        arc_point = arc_vec[index]

        target_arc = jnp.clip(
            arc_point + v_des * self.t_fin,
            arc_vec[0],
            arc_vec[-1],
        )

        index_final = jnp.argmin(
            jnp.abs(target_arc - arc_vec)
        )

        return (
            x_waypoint[index_final],
            y_waypoint[index_final],
            z_waypoint[index_final],
            index_final,
        )
    # =============================================================
    # Safe local goal: radial candidates, then sampled-shell fallback.
    # =============================================================

    def _goal_safety_mask(self, candidates, centers, radii, active, required_sq):
        """Check every candidate against every active inflated ellipsoid."""
        normalized = (candidates[:, None, :] - centers[None, :, :]) / radii[None, :, :]
        distance_sq = jnp.sum(normalized**2, axis=-1)
        return jnp.all((~active)[None, :] | (distance_sq >= required_sq), axis=1)

    def _goal_candidate_cost(self, candidates, goal, robot_position, lateral_weight):
        """Prefer the nominal goal, optionally penalizing lateral displacement."""
        cost = jnp.sum((candidates - goal[None, :])**2, axis=1)
        direction = goal - robot_position
        direction = direction / (jnp.linalg.norm(direction) + 1e-9)
        relative = candidates - robot_position[None, :]
        perpendicular = relative - (relative @ direction)[:, None] * direction[None, :]
        return cost + lateral_weight * jnp.sum(perpendicular**2, axis=1)

    def _shell_search_goal(
        self, goal, centers, radii, active, required_scale,
        robot_position, lateral_weight,
    ):
        """Search sampled shells only when no radial candidate is safe."""
        scales = jnp.linspace(
            required_scale * (1.0 + self.GOAL_BOUNDARY_MARGIN),
            self.goal_search_max_scale,
            self.goal_search_num_shells,
        )
        required_sq = required_scale**2

        def shell_body(shell_index, best):
            def obstacle_body(obstacle_index, current_best):
                def evaluate(carry):
                    best_cost, best_goal = carry
                    candidates = (
                        centers[obstacle_index][None, :]
                        + scales[shell_index]
                        * self.goal_search_directions
                        * radii[obstacle_index][None, :]
                    )
                    safe = self._goal_safety_mask(
                        candidates, centers, radii, active, required_sq
                    )
                    cost = jnp.where(
                        safe,
                        self._goal_candidate_cost(
                            candidates, goal, robot_position, lateral_weight
                        ),
                        jnp.inf,
                    )
                    index = jnp.argmin(cost)
                    better = cost[index] < best_cost
                    return (
                        jnp.where(better, cost[index], best_cost),
                        jnp.where(better, candidates[index], best_goal),
                    )

                # Padded slots generate no candidates and consume no search work.
                return lax.cond(active[obstacle_index], evaluate, lambda x: x, current_best)

            return lax.fori_loop(0, self.num_obs, obstacle_body, best)

        return lax.fori_loop(
            0,
            self.goal_search_num_shells,
            shell_body,
            (jnp.array(jnp.inf, dtype=goal.dtype), goal),
        )

    def project_safe_goal(
        self,
        goal,
        obstacle_centers,
        obstacle_radii,
        obstacle_active,
        robot_position,
        normalized_clearance=0.08,
        lateral_weight=0.0,
    ):
        """Return x/y/z, corrected, success using the revised radial projection.

        The nominal goal must be outside all active inflated ellipsoids.
        Unsafe goals first try radial boundary candidates; overlapping obstacles
        may require the sampled-shell fallback. Neither search executes when
        the nominal goal is already safe. Both branches compile during warm-up.
        """
        centers, radii, active = obstacle_centers, obstacle_radii, obstacle_active
        required_scale = 1.0 + normalized_clearance
        required_sq = required_scale**2
        nominal_safe = self._goal_safety_mask(
            goal[None, :], centers, radii, active, required_sq
        )[0]

        def project(_):
            direction = (goal[None, :] - centers) / radii
            norm = jnp.linalg.norm(direction, axis=1, keepdims=True)
            unit = jnp.where(
                norm > 1e-6,
                direction / jnp.maximum(norm, 1e-12),
                jnp.array([[1.0, 0.0, 0.0]], dtype=goal.dtype),
            )
            scale = required_scale * (1.0 + self.GOAL_BOUNDARY_MARGIN)
            candidates = centers + radii * unit * scale
            safe = active & self._goal_safety_mask(
                candidates, centers, radii, active, required_sq
            )
            cost = jnp.where(
                safe,
                self._goal_candidate_cost(candidates, goal, robot_position, lateral_weight),
                jnp.inf,
            )
            index = jnp.argmin(cost)
            radial_found = jnp.isfinite(cost[index])

            def fallback(_):
                best_cost, best_goal = self._shell_search_goal(
                    goal, centers, radii, active, required_scale,
                    robot_position, lateral_weight,
                )
                return best_goal, jnp.isfinite(best_cost)

            return lax.cond(
                radial_found,
                lambda _: (candidates[index], jnp.array(True)),
                fallback,
                operand=None,
            )

        chosen, success = lax.cond(
            nominal_safe,
            lambda _: (goal, jnp.array(True)),
            project,
            operand=None,
        )
        return chosen[0], chosen[1], chosen[2], ~nominal_safe, success

    def compute_contouring_error(
        self,
        x_waypoint,
        y_waypoint,
        z_waypoint,
        x_target,
        y_target,
        z_target,
        arc_vec,
    ):
        dist = jnp.sqrt(
            (x_waypoint - x_target)**2
            + (y_waypoint - y_target)**2
            + (z_waypoint - z_target)**2
        )

        index = jnp.argmin(dist)

        return (
            arc_vec[index],
            x_waypoint[index],
            y_waypoint[index],
            z_waypoint[index],
        )

    # =============================================================
    # 3D local normal plane
    # =============================================================

    def _normal_frame(self, tx, ty, tz):
        """
        Return two perpendicular unit vectors spanning the normal
        plane of the 3D path tangent.
        """
        tangent = jnp.array([tx, ty, tz])
        tangent = tangent / (
            jnp.linalg.norm(tangent) + 1e-8
        )

        # Avoid cross-product degeneracy if tangent is near Z.
        ref = jnp.where(
            jnp.abs(tangent[2]) < 0.9,
            jnp.array([0.0, 0.0, 1.0]),
            jnp.array([0.0, 1.0, 0.0]),
        )

        n1 = jnp.cross(tangent, ref)
        n1 = n1 / (jnp.linalg.norm(n1) + 1e-8)

        n2 = jnp.cross(tangent, n1)
        n2 = n2 / (jnp.linalg.norm(n2) + 1e-8)

        return n1, n2

    def _path_point_and_frame(
        self,
        fraction,
        arc_point,
        v_des,
        arc_vec,
        x_waypoint,
        y_waypoint,
        z_waypoint,
        x_diff,
        y_diff,
        z_diff,
    ):
        target_arc = arc_point + fraction * v_des * self.t_fin
        target_arc = jnp.clip(
            target_arc,
            arc_vec[0],
            arc_vec[-1],
        )

        idx = jnp.argmin(jnp.abs(target_arc - arc_vec))

        # diff arrays have one fewer element than waypoint arrays.
        idx_tangent = jnp.minimum(
            idx,
            x_diff.shape[0] - 1,
        )

        p = jnp.array(
            [
                x_waypoint[idx],
                y_waypoint[idx],
                z_waypoint[idx],
            ]
        )

        n1, n2 = self._normal_frame(
            x_diff[idx_tangent],
            y_diff[idx_tangent],
            z_diff[idx_tangent],
        )

        return p, n1, n2

    # =============================================================
    # Boundary vectors
    # =============================================================

    def compute_boundary_vec(
        self,
        initial_state,
        x_fin,
        y_fin,
        z_fin,
    ):
        (
            x0, y0, z0,
            vx0, vy0, vz0,
            ax0, ay0, az0,
        ) = initial_state

        n = self.num_batch

        def col(value):
            return value * jnp.ones((n, 1))

        b_eq_x = jnp.hstack(
            (col(x0), col(vx0), col(ax0), col(x_fin))
        )
        b_eq_y = jnp.hstack(
            (col(y0), col(vy0), col(ay0), col(y_fin))
        )
        b_eq_z = jnp.hstack(
            (col(z0), col(vz0), col(az0), col(z_fin))
        )

        return b_eq_x, b_eq_y, b_eq_z

    # =============================================================
    # Warm trajectories
    # =============================================================

    def compute_warm_traj(
        self,
        initial_state,
        v_des,
        x_waypoint,
        y_waypoint,
        z_waypoint,
        arc_vec,
        x_diff,
        y_diff,
        z_diff,
        x_fin,
        y_fin,
        z_fin,
    ):
        (
            x0, y0, z0,
            vx0, vy0, vz0,
            ax0, ay0, az0,
        ) = initial_state

        dist = jnp.sqrt(
            (x_waypoint - x0)**2
            + (y_waypoint - y0)**2
            + (z_waypoint - z0)**2
        )

        idx = jnp.argmin(dist)
        arc_point = arc_vec[idx]

        p50, n1, n2 = self._path_point_and_frame(
            0.50,
            arc_point,
            v_des,
            arc_vec,
            x_waypoint,
            y_waypoint,
            z_waypoint,
            x_diff,
            y_diff,
            z_diff,
        )

        mid = (
            p50[None, :]
            + self.warm_u[:, None] * n1[None, :]
            + self.warm_v[:, None] * n2[None, :]
        )

        nw = self.num_sample_warm

        def col(value):
            return value * jnp.ones((nw, 1))

        b_x = jnp.hstack(
            (
                col(x0),
                col(vx0),
                col(ax0),
                mid[:, 0:1],
                col(x_fin),
            )
        )

        b_y = jnp.hstack(
            (
                col(y0),
                col(vy0),
                col(ay0),
                mid[:, 1:2],
                col(y_fin),
            )
        )

        b_z = jnp.hstack(
            (
                col(z0),
                col(vz0),
                col(az0),
                mid[:, 2:3],
                col(z_fin),
            )
        )

        mid_index = self.num // 2

        Aeq = jnp.vstack(
            (
                self.P[0],
                self.Pdot[0],
                self.Pddot[0],
                self.P[mid_index],
                self.P[-1],
            )
        )

        Q = self.Pddot.T @ self.Pddot
        Z = jnp.zeros((Aeq.shape[0], Aeq.shape[0]))

        KKT = jnp.vstack(
            (
                jnp.hstack((Q, Aeq.T)),
                jnp.hstack((Aeq, Z)),
            )
        )

        KKT_inv = jnp.linalg.inv(KKT)

        zero_linear = jnp.zeros((nw, self.nvar))

        def solve_axis(b):
            rhs = jnp.hstack((zero_linear, b))
            sol = (KKT_inv @ rhs.T).T
            coeff = sol[:, :self.nvar]
            traj = (self.P @ coeff.T).T
            return traj

        x_warm = solve_axis(b_x)
        y_warm = solve_axis(b_y)
        z_warm = solve_axis(b_z)

        return x_warm, y_warm, z_warm

    # =============================================================
    # Initial large sampled population
    # =============================================================

    def compute_traj_guess(
        self,
        initial_state,
        x_obs_trajectory,
        y_obs_trajectory,
        z_obs_trajectory,
        v_des,
        x_waypoint,
        y_waypoint,
        z_waypoint,
        arc_vec,
        x_guess_warm,
        y_guess_warm,
        z_guess_warm,
        x_diff,
        y_diff,
        z_diff,
        x_fin,
        y_fin,
        z_fin,
        *,
        obstacle_radii,
        obstacle_active,
    ):
        a_obs_flat, b_obs_flat, c_obs_flat = jnp.repeat(
            obstacle_radii, self.num, axis=0
        ).T

        (
            x0, y0, z0,
            vx0, vy0, vz0,
            ax0, ay0, az0,
        ) = initial_state

        dist = jnp.sqrt(
            (x_waypoint - x0)**2
            + (y_waypoint - y0)**2
            + (z_waypoint - z0)**2
        )

        idx = jnp.argmin(dist)
        arc_point = arc_vec[idx]

        p25, n1_25, n2_25 = self._path_point_and_frame(
            0.25, arc_point, v_des, arc_vec,
            x_waypoint, y_waypoint, z_waypoint,
            x_diff, y_diff, z_diff,
        )

        p50, n1_50, n2_50 = self._path_point_and_frame(
            0.50, arc_point, v_des, arc_vec,
            x_waypoint, y_waypoint, z_waypoint,
            x_diff, y_diff, z_diff,
        )

        p75, n1_75, n2_75 = self._path_point_and_frame(
            0.75, arc_point, v_des, arc_vec,
            x_waypoint, y_waypoint, z_waypoint,
            x_diff, y_diff, z_diff,
        )

        p25_samples = (
            p25[None, :]
            + self.sample_u_25[:, None] * n1_25[None, :]
            + self.sample_v_25[:, None] * n2_25[None, :]
        )

        p50_samples = (
            p50[None, :]
            + self.sample_u_50[:, None] * n1_50[None, :]
            + self.sample_v_50[:, None] * n2_50[None, :]
        )

        p75_samples = (
            p75[None, :]
            + self.sample_u_75[:, None] * n1_75[None, :]
            + self.sample_v_75[:, None] * n2_75[None, :]
        )

        ns = self.num_batch * self.initial_up_sampling

        def col(value):
            return value * jnp.ones((ns, 1))

        # Same constraint ordering as original code:
        # init pos, init vel, init acc, 25%, 75%, 50%, final
        bx = jnp.hstack(
            (
                col(x0),
                col(vx0),
                col(ax0),
                p25_samples[:, 0:1],
                p75_samples[:, 0:1],
                p50_samples[:, 0:1],
                col(x_fin),
            )
        )

        by = jnp.hstack(
            (
                col(y0),
                col(vy0),
                col(ay0),
                p25_samples[:, 1:2],
                p75_samples[:, 1:2],
                p50_samples[:, 1:2],
                col(y_fin),
            )
        )

        bz = jnp.hstack(
            (
                col(z0),
                col(vz0),
                col(az0),
                p25_samples[:, 2:3],
                p75_samples[:, 2:3],
                p50_samples[:, 2:3],
                col(z_fin),
            )
        )

        i25 = int(round(0.25 * (self.num - 1)))
        i50 = int(round(0.50 * (self.num - 1)))
        i75 = int(round(0.75 * (self.num - 1)))

        Aeq = jnp.vstack(
            (
                self.P[0],
                self.Pdot[0],
                self.Pddot[0],
                self.P[i25],
                self.P[i75],
                self.P[i50],
                self.P[-1],
            )
        )

        Q = self.Pddot.T @ self.Pddot
        Z = jnp.zeros((Aeq.shape[0], Aeq.shape[0]))

        KKT = jnp.vstack(
            (
                jnp.hstack((Q, Aeq.T)),
                jnp.hstack((Aeq, Z)),
            )
        )

        KKT_inv = jnp.linalg.inv(KKT)
        zero_linear = jnp.zeros((ns, self.nvar))

        def solve_samples(b):
            rhs = jnp.hstack((zero_linear, b))
            sol = (KKT_inv @ rhs.T).T
            c = sol[:, :self.nvar]
            return (self.P @ c.T).T

        xs = solve_samples(bx)
        ys = solve_samples(by)
        zs = solve_samples(bz)

        # ---------------------------------------------------------
        # Obstacle pre-filter in true 3D ellipsoid space
        # ---------------------------------------------------------
        dx = (
            xs - x_obs_trajectory[:, None]
        ).transpose(1, 0, 2).reshape(
            ns, self.num_obs * self.num
        )

        dy = (
            ys - y_obs_trajectory[:, None]
        ).transpose(1, 0, 2).reshape(
            ns, self.num_obs * self.num
        )

        dz = (
            zs - z_obs_trajectory[:, None]
        ).transpose(1, 0, 2).reshape(
            ns, self.num_obs * self.num
        )

        violation = (
            1.0
            - (dx / a_obs_flat)**2
            - (dy / b_obs_flat)**2
            - (dz / c_obs_flat)**2
        )

        violation = jnp.where(
            jnp.repeat(obstacle_active, self.num)[None, :], violation, 0.0
        )
        obs_penalty = jnp.linalg.norm(
            jnp.maximum(0.0, violation),
            axis=1,
        )

        order = jnp.argsort(obs_penalty)

        keep = self.num_batch - self.num_sample_warm

        x_sample = xs[order[:keep]]
        y_sample = ys[order[:keep]]
        z_sample = zs[order[:keep]]

        x_guess = jnp.vstack((x_sample, x_guess_warm))
        y_guess = jnp.vstack((y_sample, y_guess_warm))
        z_guess = jnp.vstack((z_sample, z_guess_warm))

        # ---------------------------------------------------------
        # Fit all selected trajectories to Bernstein coefficients
        # ---------------------------------------------------------
        def fit_axis(traj):
            lincost = -(self.P.T @ traj.T).T
            coeff = (
                self.cost_fit_inv @ (-lincost).T
            ).T
            return coeff[:, :self.nvar]

        cx = fit_axis(x_guess)
        cy = fit_axis(y_guess)
        cz = fit_axis(z_guess)

        x_guess = (self.P @ cx.T).T
        xdot_guess = (self.Pdot @ cx.T).T
        xddot_guess = (self.Pddot @ cx.T).T

        y_guess = (self.P @ cy.T).T
        ydot_guess = (self.Pdot @ cy.T).T
        yddot_guess = (self.Pddot @ cy.T).T

        z_guess = (self.P @ cz.T).T
        zdot_guess = (self.Pdot @ cz.T).T
        zddot_guess = (self.Pddot @ cz.T).T

        coeff_all = jnp.hstack((cx, cy, cz))

        c_mean = jnp.mean(coeff_all, axis=0)
        c_cov = jnp.cov(coeff_all.T) + 1e-6 * jnp.eye(
            3 * self.nvar
        )

        return (
            cx, cy, cz,
            x_guess, y_guess, z_guess,
            xdot_guess, ydot_guess, zdot_guess,
            xddot_guess, yddot_guess, zddot_guess,
            c_mean, c_cov,
            x_fin, y_fin, z_fin,
        )

    # =============================================================
    # Auxiliary feasible variables
    # =============================================================

    def _auxiliary_from_trajectory(
        self,
        x, y, z,
        xdot, ydot, zdot,
        xddot, yddot, zddot,
        x_obs,
        y_obs,
        z_obs,
        lamda_x,
        lamda_y,
        lamda_z,
        *,
        obstacle_radii,
        obstacle_active,
    ):
        a_obs_flat, b_obs_flat, c_obs_flat = jnp.repeat(
            obstacle_radii[:self.num_obs_proj], self.num, axis=0
        ).T

        batch = x.shape[0]
        obs_flat_count = self.num_obs_proj * self.num

        dx = (
            x - x_obs[:, None]
        ).transpose(1, 0, 2).reshape(
            batch, obs_flat_count
        )

        dy = (
            y - y_obs[:, None]
        ).transpose(1, 0, 2).reshape(
            batch, obs_flat_count
        )

        dz = (
            z - z_obs[:, None]
        ).transpose(1, 0, 2).reshape(
            batch, obs_flat_count
        )

        # ---------------------------------------------------------
        # Obstacle ellipsoid direction
        #
        # q is position expressed in normalized ellipsoid coordinates.
        # Outside obstacle: d = ||q|| >= 1 and residual is zero.
        # Inside obstacle: d is clipped to 1, pushing projected point
        # to the ellipsoid boundary.
        # ---------------------------------------------------------
        qx = dx / a_obs_flat
        qy = dy / b_obs_flat
        qz = dz / c_obs_flat

        qnorm = jnp.sqrt(
            qx**2 + qy**2 + qz**2 + 1e-12
        )

        uox = qx / qnorm
        uoy = qy / qnorm
        uoz = qz / qnorm

        d_obs = jnp.maximum(1.0, qnorm)

        obs_proj_x = (
            a_obs_flat * d_obs * uox
        )
        obs_proj_y = (
            b_obs_flat * d_obs * uoy
        )
        obs_proj_z = (
            c_obs_flat * d_obs * uoz
        )

        active_flat = jnp.repeat(obstacle_active[:self.num_obs_proj], self.num)
        res_obs_x = jnp.where(active_flat, dx - obs_proj_x, 0.0)
        res_obs_y = jnp.where(active_flat, dy - obs_proj_y, 0.0)
        res_obs_z = jnp.where(active_flat, dz - obs_proj_z, 0.0)

        # ---------------------------------------------------------
        # Velocity sphere
        # ---------------------------------------------------------
        vnorm = jnp.sqrt(
            xdot**2 + ydot**2 + zdot**2 + 1e-12
        )

        uvx = xdot / vnorm
        uvy = ydot / vnorm
        uvz = zdot / vnorm

        d_v = jnp.minimum(self.v_max, vnorm)

        res_vx = xdot - d_v * uvx
        res_vy = ydot - d_v * uvy
        res_vz = zdot - d_v * uvz

        # ---------------------------------------------------------
        # Acceleration sphere
        # ---------------------------------------------------------
        anorm = jnp.sqrt(
            xddot**2 + yddot**2 + zddot**2 + 1e-12
        )

        uax = xddot / anorm
        uay = yddot / anorm
        uaz = zddot / anorm

        d_a = jnp.minimum(self.a_max, anorm)

        res_ax = xddot - d_a * uax
        res_ay = yddot - d_a * uay
        res_az = zddot - d_a * uaz

        # ---------------------------------------------------------
        # Dual updates in coefficient space
        # ---------------------------------------------------------
        lamda_x = (
            lamda_x
            - self.rho_obs * (
                self.A_obs.T @ res_obs_x.T
            ).T
            - self.rho_ineq * (
                self.A_acc.T @ res_ax.T
            ).T
            - self.rho_ineq * (
                self.A_vel.T @ res_vx.T
            ).T
        )

        lamda_y = (
            lamda_y
            - self.rho_obs * (
                self.A_obs.T @ res_obs_y.T
            ).T
            - self.rho_ineq * (
                self.A_acc.T @ res_ay.T
            ).T
            - self.rho_ineq * (
                self.A_vel.T @ res_vy.T
            ).T
        )

        lamda_z = (
            lamda_z
            - self.rho_obs * (
                self.A_obs.T @ res_obs_z.T
            ).T
            - self.rho_ineq * (
                self.A_acc.T @ res_az.T
            ).T
            - self.rho_ineq * (
                self.A_vel.T @ res_vz.T
            ).T
        )

        res_obs = jnp.hstack(
            (res_obs_x, res_obs_y, res_obs_z)
        )
        res_vel = jnp.hstack(
            (res_vx, res_vy, res_vz)
        )
        res_acc = jnp.hstack(
            (res_ax, res_ay, res_az)
        )

        res_norm = (
            jnp.linalg.norm(res_obs, axis=1)
            + jnp.linalg.norm(res_vel, axis=1)
            + jnp.linalg.norm(res_acc, axis=1)
        )

        return (
            uox, uoy, uoz, d_obs,
            uax, uay, uaz, d_a,
            uvx, uvy, uvz, d_v,
            lamda_x, lamda_y, lamda_z,
            res_norm,
        )

    def initial_alpha_d(
        self,
        x, y, z,
        xdot, ydot, zdot,
        xddot, yddot, zddot,
        x_obs, y_obs, z_obs,
        lamda_x, lamda_y, lamda_z,
        *,
        obstacle_radii,
        obstacle_active,
    ):
        return self._auxiliary_from_trajectory(
            x, y, z,
            xdot, ydot, zdot,
            xddot, yddot, zddot,
            x_obs, y_obs, z_obs,
            lamda_x, lamda_y, lamda_z,
            obstacle_radii=obstacle_radii,
            obstacle_active=obstacle_active,
        )

    def compute_alph_d_proj(
        self,
        x, y, z,
        xdot, ydot, zdot,
        xddot, yddot, zddot,
        x_obs, y_obs, z_obs,
        lamda_x, lamda_y, lamda_z,
        *,
        obstacle_radii,
        obstacle_active,
    ):
        return self._auxiliary_from_trajectory(
            x, y, z,
            xdot, ydot, zdot,
            xddot, yddot, zddot,
            x_obs, y_obs, z_obs,
            lamda_x, lamda_y, lamda_z,
            obstacle_radii=obstacle_radii,
            obstacle_active=obstacle_active,
        )

    # =============================================================
    # Coefficient-space projection
    # =============================================================

    def compute_projection(
        self,
        x_obs,
        y_obs,
        z_obs,
        uox, uoy, uoz, d_obs,
        uax, uay, uaz, d_a,
        uvx, uvy, uvz, d_v,
        lamda_x,
        lamda_y,
        lamda_z,
        b_eq_x,
        b_eq_y,
        b_eq_z,
        cx_sample,
        cy_sample,
        cz_sample,
        *,
        obstacle_radii,
        obstacle_active,
    ):
        a_obs_flat, b_obs_flat, c_obs_flat = jnp.repeat(
            obstacle_radii[:self.num_obs_proj], self.num, axis=0
        ).T

        xobs_flat = x_obs.reshape(
            self.num_obs_proj * self.num
        )
        yobs_flat = y_obs.reshape(
            self.num_obs_proj * self.num
        )
        zobs_flat = z_obs.reshape(
            self.num_obs_proj * self.num
        )

        b_obs_x = (
            xobs_flat
            + a_obs_flat * d_obs * uox
        )
        b_obs_y = (
            yobs_flat
            + b_obs_flat * d_obs * uoy
        )
        b_obs_z = (
            zobs_flat
            + c_obs_flat * d_obs * uoz
        )

        # Mask both the linear term and its matching Hessian contribution.
        active_proj = obstacle_active[:self.num_obs_proj]
        active_flat = jnp.repeat(active_proj, self.num)
        b_obs_x = jnp.where(active_flat, b_obs_x, 0.0)
        b_obs_y = jnp.where(active_flat, b_obs_y, 0.0)
        b_obs_z = jnp.where(active_flat, b_obs_z, 0.0)
        inverse = self.projection_inverse_by_count[jnp.sum(active_proj, dtype=jnp.int32)]

        b_ax = d_a * uax
        b_ay = d_a * uay
        b_az = d_a * uaz

        b_vx = d_v * uvx
        b_vy = d_v * uvy
        b_vz = d_v * uvz

        def solve_axis(
            c_sample,
            lamda,
            b_obs,
            b_acc,
            b_vel,
            b_eq,
        ):
            lincost = (
                -self.rho_proj * (
                    self.A_projection.T @ c_sample.T
                ).T
                - lamda
                - self.rho_obs * (
                    self.A_obs.T @ b_obs.T
                ).T
                - self.rho_ineq * (
                    self.A_acc.T @ b_acc.T
                ).T
                - self.rho_ineq * (
                    self.A_vel.T @ b_vel.T
                ).T
            )

            rhs = jnp.hstack((-lincost, b_eq))

            sol = (
                inverse @ rhs.T
            ).T

            return sol[:, :self.nvar]

        cx = solve_axis(
            cx_sample, lamda_x,
            b_obs_x, b_ax, b_vx, b_eq_x,
        )
        cy = solve_axis(
            cy_sample, lamda_y,
            b_obs_y, b_ay, b_vy, b_eq_y,
        )
        cz = solve_axis(
            cz_sample, lamda_z,
            b_obs_z, b_az, b_vz, b_eq_z,
        )

        x = (self.P @ cx.T).T
        xdot = (self.Pdot @ cx.T).T
        xddot = (self.Pddot @ cx.T).T

        y = (self.P @ cy.T).T
        ydot = (self.Pdot @ cy.T).T
        yddot = (self.Pddot @ cy.T).T

        z = (self.P @ cz.T).T
        zdot = (self.Pdot @ cz.T).T
        zddot = (self.Pddot @ cz.T).T

        return (
            cx, cy, cz,
            x, y, z,
            xdot, ydot, zdot,
            xddot, yddot, zddot,
        )

    # =============================================================
    # Repeated batch projection
    # =============================================================

    def compute_projection_sampling(
        self,
        key,
        cx_sample,
        cy_sample,
        cz_sample,
        x_obs_proj,
        y_obs_proj,
        z_obs_proj,
        lamda_x,
        lamda_y,
        lamda_z,
        x_guess,
        y_guess,
        z_guess,
        xdot_guess,
        ydot_guess,
        zdot_guess,
        xddot_guess,
        yddot_guess,
        zddot_guess,
        initial_state,
        x_fin,
        y_fin,
        z_fin,
        *,
        obstacle_radii,
        obstacle_active,
    ):
        b_eq_x, b_eq_y, b_eq_z = self.compute_boundary_vec(
            initial_state,
            x_fin,
            y_fin,
            z_fin,
        )

        aux = self.initial_alpha_d(
            x_guess, y_guess, z_guess,
            xdot_guess, ydot_guess, zdot_guess,
            xddot_guess, yddot_guess, zddot_guess,
            x_obs_proj, y_obs_proj, z_obs_proj,
            lamda_x, lamda_y, lamda_z,
            obstacle_radii=obstacle_radii,
            obstacle_active=obstacle_active,
        )

        (
            uox, uoy, uoz, d_obs,
            uax, uay, uaz, d_a,
            uvx, uvy, uvz, d_v,
            lamda_x, lamda_y, lamda_z,
            res_norm,
        ) = aux

        zero_c = jnp.zeros(
            (self.num_batch, self.nvar)
        )
        zero_t = jnp.zeros(
            (self.num_batch, self.num)
        )

        carry_init = (
            zero_c, zero_c, zero_c,
            zero_t, zero_t, zero_t,
            zero_t, zero_t, zero_t,
            zero_t, zero_t, zero_t,
            res_norm,
            uox, uoy, uoz, d_obs,
            uax, uay, uaz, d_a,
            uvx, uvy, uvz, d_v,
            lamda_x, lamda_y, lamda_z,
        )

        def body(carry, _):
            (
                cx, cy, cz,
                x, y, z,
                xdot, ydot, zdot,
                xddot, yddot, zddot,
                res_norm,
                uox, uoy, uoz, d_obs,
                uax, uay, uaz, d_a,
                uvx, uvy, uvz, d_v,
                lamda_x, lamda_y, lamda_z,
            ) = carry

            (
                cx, cy, cz,
                x, y, z,
                xdot, ydot, zdot,
                xddot, yddot, zddot,
            ) = self.compute_projection(
                x_obs_proj,
                y_obs_proj,
                z_obs_proj,
                uox, uoy, uoz, d_obs,
                uax, uay, uaz, d_a,
                uvx, uvy, uvz, d_v,
                lamda_x,
                lamda_y,
                lamda_z,
                b_eq_x,
                b_eq_y,
                b_eq_z,
                cx_sample,
                cy_sample,
                cz_sample,
                obstacle_radii=obstacle_radii,
                obstacle_active=obstacle_active,
            )

            (
                uox, uoy, uoz, d_obs,
                uax, uay, uaz, d_a,
                uvx, uvy, uvz, d_v,
                lamda_x, lamda_y, lamda_z,
                res_norm,
            ) = self.compute_alph_d_proj(
                x, y, z,
                xdot, ydot, zdot,
                xddot, yddot, zddot,
                x_obs_proj,
                y_obs_proj,
                z_obs_proj,
                lamda_x,
                lamda_y,
                lamda_z,
                obstacle_radii=obstacle_radii,
                obstacle_active=obstacle_active,
            )

            new_carry = (
                cx, cy, cz,
                x, y, z,
                xdot, ydot, zdot,
                xddot, yddot, zddot,
                res_norm,
                uox, uoy, uoz, d_obs,
                uax, uay, uaz, d_a,
                uvx, uvy, uvz, d_v,
                lamda_x, lamda_y, lamda_z,
            )

            return new_carry, x

        carry_fin, _ = lax.scan(
            body,
            carry_init,
            jnp.arange(self.maxitet_proj),
        )

        (
            cx, cy, cz,
            x, y, z,
            xdot, ydot, zdot,
            xddot, yddot, zddot,
            res_norm,
            *_,
        ) = carry_fin

        return (
            cx, cy, cz,
            x, y, z,
            xdot, ydot, zdot,
            xddot, yddot, zddot,
            res_norm,
        )

    # =============================================================
    # CEM cost
    # =============================================================

    def compute_cost_batch(
        self,
        x, y, z,
        xdot, ydot, zdot,
        xddot, yddot, zddot,
        x_project,
        y_project,
        z_project,
        res_norm_batch,
        x_fin,
        y_fin,
        z_fin,
        x_obs,
        y_obs,
        z_obs,
        *,
        obstacle_radii,
        obstacle_active,
    ):
        a_obs_flat, b_obs_flat, c_obs_flat = jnp.repeat(
            obstacle_radii, self.num, axis=0
        ).T

        batch = x.shape[0]

        dx = (
            x - x_obs[:, None]
        ).transpose(1, 0, 2).reshape(
            batch, self.num_obs * self.num
        )

        dy = (
            y - y_obs[:, None]
        ).transpose(1, 0, 2).reshape(
            batch, self.num_obs * self.num
        )

        dz = (
            z - z_obs[:, None]
        ).transpose(1, 0, 2).reshape(
            batch, self.num_obs * self.num
        )

        violation = (
            1.0
            - (dx / a_obs_flat)**2
            - (dy / b_obs_flat)**2
            - (dz / c_obs_flat)**2
        )

        active_flat = jnp.repeat(obstacle_active, self.num)[None, :]
        cost_obs = jnp.linalg.norm(
            jnp.where(active_flat, jnp.maximum(0.0, violation), 0.0),
            axis=1,
        )

        # Preserve the original clearance-style score over real obstacles only.
        # Without this mask, padded slots would dominate trajectory ranking.
        clearance_cost = jnp.where(
            jnp.any(obstacle_active),
            -jnp.min(jnp.where(active_flat, violation, jnp.inf), axis=1),
            0.0,
        )

        acceleration_mag = jnp.sqrt(
            xddot**2 + yddot**2 + zddot**2
        )

        cost_smoothness = jnp.linalg.norm(
            acceleration_mag,
            axis=1,
        )

        tracking_error = jnp.sqrt(
            (x - x_project)**2
            + (y - y_project)**2
            + (z - z_project)**2
        )

        cost_track = jnp.linalg.norm(
            tracking_error,
            axis=1,
        )

        return (
            1.0 * res_norm_batch
            + self.weight_smoothness * cost_smoothness
            + self.weight_track * cost_track
            + 1.0 * cost_obs
            + 1.0 * clearance_cost
        )

    def comp_prod(self, diffs, d):
        return d * jnp.outer(diffs, diffs)

    # =============================================================
    # CEM resampling
    # =============================================================

    def compute_shifted_samples(
        self,
        key,
        cost_batch,
        cx_elite,
        cy_elite,
        cz_elite,
        x_obs,
        y_obs,
        z_obs,
        iteration,
        c_mean_prev,
        c_cov_prev,
        *,
        obstacle_radii,
        obstacle_active,
    ):
        a_obs_flat, b_obs_flat, c_obs_flat = jnp.repeat(
            obstacle_radii, self.num, axis=0
        ).T

        c_elite = jnp.hstack(
            (cx_elite, cy_elite, cz_elite)
        )

        beta_param = jnp.min(cost_batch)

        weights = jnp.exp(
            -(1.0 / self.lamda)
            * (cost_batch - beta_param)
        )

        sum_w = jnp.sum(weights) + 1e-12

        c_mean_new = (
            jnp.sum(
                c_elite * weights[:, None],
                axis=0,
            )
            / sum_w
        )

        c_mean = (
            (1.0 - self.alpha) * c_mean_prev
            + self.alpha * c_mean_new
        )

        diffs = c_elite - c_mean

        prod = self.vec_product(
            diffs,
            weights,
        )

        c_cov_new = (
            jnp.sum(prod, axis=0)
            / sum_w
        )

        c_cov = (
            (1.0 - self.alpha) * c_cov_prev
            + self.alpha * c_cov_new
            + 1e-6 * jnp.eye(3 * self.nvar)
        )

        sample_count = (
            self.initial_up_sampling
            * self.num_batch
        )

        samples = jax.random.multivariate_normal(
            key,
            c_mean,
            c_cov,
            (sample_count,),
        )

        cx_temp = samples[:, :self.nvar]
        cy_temp = samples[
            :, self.nvar:2*self.nvar
        ]
        cz_temp = samples[
            :, 2*self.nvar:3*self.nvar
        ]

        xs = (self.P @ cx_temp.T).T
        ys = (self.P @ cy_temp.T).T
        zs = (self.P @ cz_temp.T).T

        dx = (
            xs - x_obs[:, None]
        ).transpose(1, 0, 2).reshape(
            sample_count,
            self.num_obs * self.num,
        )

        dy = (
            ys - y_obs[:, None]
        ).transpose(1, 0, 2).reshape(
            sample_count,
            self.num_obs * self.num,
        )

        dz = (
            zs - z_obs[:, None]
        ).transpose(1, 0, 2).reshape(
            sample_count,
            self.num_obs * self.num,
        )

        violation = (
            1.0
            - (dx / a_obs_flat)**2
            - (dy / b_obs_flat)**2
            - (dz / c_obs_flat)**2
        )

        violation = jnp.where(
            jnp.repeat(obstacle_active, self.num)[None, :], violation, 0.0
        )
        obs_penalty = jnp.linalg.norm(
            jnp.maximum(0.0, violation),
            axis=1,
        )

        order = jnp.argsort(obs_penalty)

        new_count = (
            self.num_batch
            - self.ellite_num_const
        )

        cx_shift = cx_temp[order[:new_count]]
        cy_shift = cy_temp[order[:new_count]]
        cz_shift = cz_temp[order[:new_count]]

        cx = jnp.vstack((cx_elite, cx_shift))
        cy = jnp.vstack((cy_elite, cy_shift))
        cz = jnp.vstack((cz_elite, cz_shift))

        x = (self.P @ cx.T).T
        xdot = (self.Pdot @ cx.T).T
        xddot = (self.Pddot @ cx.T).T

        y = (self.P @ cy.T).T
        ydot = (self.Pdot @ cy.T).T
        yddot = (self.Pddot @ cy.T).T

        z = (self.P @ cz.T).T
        zdot = (self.Pdot @ cz.T).T
        zddot = (self.Pddot @ cz.T).T

        return (
            cx, cy, cz,
            x, y, z,
            xdot, ydot, zdot,
            xddot, yddot, zddot,
            c_mean, c_cov,
        )

    # =============================================================
    # CEM outer loop
    # =============================================================

    def compute_cem(
        self,
        key,
        initial_state,
        x_fin,
        y_fin,
        z_fin,
        lamda_x,
        lamda_y,
        lamda_z,
        x_obs,
        y_obs,
        z_obs,
        x_obs_proj,
        y_obs_proj,
        z_obs_proj,
        cx,
        cy,
        cz,
        x_guess,
        y_guess,
        z_guess,
        xdot_guess,
        ydot_guess,
        zdot_guess,
        xddot_guess,
        yddot_guess,
        zddot_guess,
        x_waypoint,
        y_waypoint,
        z_waypoint,
        arc_vec,
        c_mean,
        c_cov,
        *,
        obstacle_radii,
        obstacle_active,
    ):
        c_mean_prev = c_mean
        c_cov_prev = c_cov

        # Initialize for JIT tracing.
        idx_min = jnp.array(0)
        idx_sort = jnp.arange(self.ellite_num_const)

        for i in range(self.maxiter_cem):
            (
                cx_proj, cy_proj, cz_proj,
                x, y, z,
                xdot, ydot, zdot,
                xddot, yddot, zddot,
                res_norm,
            ) = self.compute_projection_sampling(
                key,
                cx, cy, cz,
                x_obs_proj,
                y_obs_proj,
                z_obs_proj,
                lamda_x,
                lamda_y,
                lamda_z,
                x_guess,
                y_guess,
                z_guess,
                xdot_guess,
                ydot_guess,
                zdot_guess,
                xddot_guess,
                yddot_guess,
                zddot_guess,
                initial_state,
                x_fin,
                y_fin,
                z_fin,
                obstacle_radii=obstacle_radii,
                obstacle_active=obstacle_active,
            )

            projection_order = jnp.argsort(res_norm)
            elite_idx = projection_order[
                :self.ellite_num_const
            ]

            x_elite = x[elite_idx]
            y_elite = y[elite_idx]
            z_elite = z[elite_idx]

            xdot_elite = xdot[elite_idx]
            ydot_elite = ydot[elite_idx]
            zdot_elite = zdot[elite_idx]

            xddot_elite = xddot[elite_idx]
            yddot_elite = yddot[elite_idx]
            zddot_elite = zddot[elite_idx]

            cx_elite = cx_proj[elite_idx]
            cy_elite = cy_proj[elite_idx]
            cz_elite = cz_proj[elite_idx]

            res_elite = res_norm[elite_idx]

            xf = x_elite.reshape(
                self.ellite_num_const * self.num
            )
            yf = y_elite.reshape(
                self.ellite_num_const * self.num
            )
            zf = z_elite.reshape(
                self.ellite_num_const * self.num
            )

            (
                normal_distance,
                x_project,
                y_project,
                z_project,
            ) = self.vectorized_projection(
                x_waypoint,
                y_waypoint,
                z_waypoint,
                xf, yf, zf,
                arc_vec,
            )

            x_project = x_project.reshape(
                self.ellite_num_const,
                self.num,
            )
            y_project = y_project.reshape(
                self.ellite_num_const,
                self.num,
            )
            z_project = z_project.reshape(
                self.ellite_num_const,
                self.num,
            )

            cost_batch = self.compute_cost_batch(
                x_elite, y_elite, z_elite,
                xdot_elite, ydot_elite, zdot_elite,
                xddot_elite, yddot_elite, zddot_elite,
                x_project, y_project, z_project,
                res_elite,
                x_fin, y_fin, z_fin,
                x_obs, y_obs, z_obs,
                obstacle_radii=obstacle_radii,
                obstacle_active=obstacle_active,
            )

            idx_min = jnp.argmin(cost_batch)
            idx_sort = jnp.argsort(cost_batch)

            key, subkey = jax.random.split(key)

            (
                cx, cy, cz,
                x_guess, y_guess, z_guess,
                xdot_guess, ydot_guess, zdot_guess,
                xddot_guess, yddot_guess, zddot_guess,
                c_mean_prev, c_cov_prev,
            ) = self.compute_shifted_samples(
                subkey,
                cost_batch,
                cx_elite,
                cy_elite,
                cz_elite,
                x_obs,
                y_obs,
                z_obs,
                i,
                c_mean_prev,
                c_cov_prev,
                obstacle_radii=obstacle_radii,
                obstacle_active=obstacle_active,
            )

        cx_best = cx_elite[idx_min]
        cy_best = cy_elite[idx_min]
        cz_best = cz_elite[idx_min]

        x_best = x_elite[idx_min]
        y_best = y_elite[idx_min]
        z_best = z_elite[idx_min]

        xdot_best = xdot_elite[idx_min]
        ydot_best = ydot_elite[idx_min]
        zdot_best = zdot_elite[idx_min]

        warm_idx = idx_sort[:self.num_sample_warm]

        x_warm_next = x_guess[warm_idx]
        y_warm_next = y_guess[warm_idx]
        z_warm_next = z_guess[warm_idx]

        return (
            x_elite,
            y_elite,
            z_elite,
            x,
            y,
            z,
            cx_best,
            cy_best,
            cz_best,
            x_best,
            y_best,
            z_best,
            xdot_best,
            ydot_best,
            zdot_best,
            x_warm_next,
            y_warm_next,
            z_warm_next,
        )

    # =============================================================
    # Optional 3D velocity/acceleration command extraction
    # =============================================================

    def compute_controls(
        self,
        cx_best,
        cy_best,
        cz_best,
    ):
        xdot = self.Pdot @ cx_best
        ydot = self.Pdot @ cy_best
        zdot = self.Pdot @ cz_best

        xddot = self.Pddot @ cx_best
        yddot = self.Pddot @ cy_best
        zddot = self.Pddot @ cz_best

        return (
            xdot[0],
            ydot[0],
            zdot[0],
            xddot[0],
            yddot[0],
            zddot[0],
        )
