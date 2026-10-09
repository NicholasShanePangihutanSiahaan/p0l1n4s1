#!/usr/bin/env python3
import math
import time
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from threading import RLock

import numpy as np
import jax
from jax import random
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor, ExternalShutdownException
from rclpy.duration import Duration
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    DurabilityPolicy,
    qos_profile_sensor_data,
)
from geometry_msgs.msg import Pose, PoseStamped
from std_msgs.msg import String
from nav_msgs.msg import Odometry, Path
from pcl_cstm_msg.msg import AxisAlignedElipsoidArray
from . import priest3d_JAX_impl


def synchronized(callback):
    """Protect shared node state across the subscription and timer groups.

    Planning itself runs in a worker on copied arrays, outside this lock.
    RLock permits the timer to reuse on_path() when making a linear reference.
    """
    @wraps(callback)
    def wrapped(self, *args, **kwargs):
        with self.state_lock:
            return callback(self, *args, **kwargs)

    return wrapped


def resample(points, count):
    """Space reference samples evenly along the path, using traveled distance."""
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    arc = np.r_[0.0, np.cumsum(lengths)]
    target = np.linspace(0.0, arc[-1], count)
    return np.column_stack([np.interp(target, arc, points[:, i]) for i in range(3)])


def body_to_world(velocity, quaternion):
    """Rotate odometry velocity into the frame used by path positions."""
    q = np.array([quaternion.x, quaternion.y, quaternion.z, quaternion.w], dtype=float)
    norm = np.linalg.norm(q)
    if not np.isfinite(norm) or norm < 1e-8:
        raise ValueError('Invalid odometry orientation')
    q /= norm
    return velocity + 2.0 * np.cross(q[:3], np.cross(q[:3], velocity) + q[3] * velocity)


def collision_free(points, centers, radii):
    """Exact line-segment/axis-aligned-ellipsoid test for the published polyline."""
    if len(centers) == 0:
        return True

    # Scaling each axis by its radius turns each ellipsoid into a unit sphere.
    # Check the closest point on every segment, including between path samples.
    start = (points[:-1, None, :] - centers[None, :, :]) / radii[None, :, :]
    delta = np.diff(points, axis=0)[:, None, :] / radii[None, :, :]
    length2 = np.sum(delta * delta, axis=2)
    t = np.clip(-np.sum(start * delta, axis=2) / np.maximum(length2, 1e-15), 0.0, 1.0)
    closest = start + t[:, :, None] * delta
    return bool(np.all(np.sum(closest * closest, axis=2) >= 1.0))


class Planner:
    """Prepare ROS input arrays for the optimizer and validate its output.

    The numerical pipeline lives in priest3d_JAX_impl.batch_crowd_nav_3d.solve().
    This wrapper selects a device, pads inputs, warms up that same entry point,
    and transfers the final trajectory back for ROS publication.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = jax.devices(cfg['device'])[0]

        # Construct once on the chosen device. The optimizer stores only fixed
        # configuration/matrices; obstacle radii are supplied to every solve.
        with jax.default_device(self.device):
            self.optimizer = priest3d_JAX_impl.batch_crowd_nav_3d(
                v_max=cfg['v_max'],
                v_min=cfg['v_min'],
                a_max=cfg['a_max'],
                num_obs=cfg['max_obstacles'],
                t_fin=cfg['horizon_seconds'],
                num=cfg['num_samples'],
                num_batch=cfg['num_batch'],
                maxiter=1,
                maxiter_cem=cfg['cem_iterations'],
                weight_smoothness=cfg['weight_smoothness'],
                weight_track=cfg['weight_track'],
                way_point_shape=cfg['reference_samples'],
                v_des=cfg['v_des'],
                goal_clearance=cfg['goal_clearance'],
                goal_lateral_weight=cfg['goal_lateral_weight'],
            )
            self.key = random.PRNGKey(cfg['seed'])

    def warm_up(self):
        """Compile and execute the complete live solve once, without publishing.

        The dummy reference is a straight line in free space. Obstacle buffers,
        reference length, state shape, dtype, and PRNG format match live calls.
        JAX traces both branches of the revised safe-goal lax.cond during this
        compilation, although only the selected branch executes in this run.
        """
        state = np.zeros(9, dtype=np.float32)
        state[3] = self.cfg['v_des']
        reference = np.zeros((self.cfg['reference_samples'], 3), dtype=np.float32)
        reference[:, 0] = np.linspace(
            0.0,
            self.cfg['v_des'] * self.cfg['horizon_seconds'],
            self.cfg['reference_samples'],
            dtype=np.float32,
        )
        empty = np.empty((0, 3), dtype=np.float32)

        # solve() explicitly waits for device completion. Restore the PRNG key
        # so startup compilation does not consume the first live random sample.
        key_before_warm_up = self.key
        try:
            self.solve(state, reference, empty, empty)
        finally:
            self.key = key_before_warm_up

    def solve(self, state, reference, centers, radii):
        """Prepare fixed-size JAX inputs and validate the resulting polyline."""
        # Keep the nearest obstacles if this method is called directly with
        # more data than the fixed JAX buffer can hold. The ROS node normally
        # performs the same selection first so it can report dropped entries.
        n = self.cfg['max_obstacles']
        if len(centers) > n:
            distance = np.linalg.norm(centers - state[None, :3], axis=1)
            keep = np.argsort(distance)[:n]
            centers = centers[keep]
            radii = radii[keep]

        # Fixed array shapes allow JAX to reuse its compiled solve when the
        # obstacle count changes. Masked slots contribute no collision penalties
        # or obstacle projection constraints.
        count = len(centers)
        padded_centers = np.zeros((n, 3), dtype=np.float32)
        padded_radii = np.ones((n, 3), dtype=np.float32)
        padded_centers[:count], padded_radii[:count] = centers, radii
        active = np.arange(n) < count

        self.key, key = random.split(self.key)
        points, success, goal_corrected = self.optimizer.solve(
            jax.device_put(np.asarray(state, dtype=np.float32), self.device),
            jax.device_put(np.asarray(reference, dtype=np.float32), self.device),
            jax.device_put(padded_centers, self.device),
            jax.device_put(padded_radii, self.device),
            jax.device_put(active, self.device),
            key,
        )

        # GPU dispatch is asynchronous. Warm-up is finished only after the
        # output has actually completed, rather than merely been submitted.
        points.block_until_ready()
        success.block_until_ready()
        goal_corrected.block_until_ready()
        points = np.asarray(points, dtype=float)
        if not bool(success) or not np.isfinite(points).all():
            raise ValueError(
                'Planner did not find a finite path with a safe endpoint; '
                f'lookahead_goal_corrected={bool(goal_corrected)}'
            )
        if not collision_free(points, centers, radii):
            raise ValueError('Planner output intersects an inflated ellipsoid')
        return points, bool(goal_corrected)


class Priest3DNode(Node):
    """Cache ROS inputs, schedule one background solve, and publish valid paths."""

    def __init__(self):
        super().__init__('priest_3d_planner')

        self.jit_warm_up = False
        self.state_lock = RLock()
        self.timer_group = MutuallyExclusiveCallbackGroup()
        self.subscription_group = MutuallyExclusiveCallbackGroup()
        # Planner resolution, motion limits, clearance, and input freshness.
        # Override these at startup with: --ros-args -p parameter_name:=value
        defaults = dict(
            plan_rate_hz=2.0,
            goal_tolerance=0.15,
            horizon_seconds=5.0,

            num_samples=100,
            num_batch=110,
            max_obstacles=60,
            reference_samples=300,
            cem_iterations=13,

            v_max=0.90,
            v_min=0.0,
            a_max=0.5,
            v_des=0.4,
            weight_smoothness=0.025,
            weight_track=0.5,
            robot_radius=0.25,
            safety_margin=0.10,
            goal_clearance=0.08,
            goal_lateral_weight=0.0,

            odom_timeout=1.0,
            obstacle_timeout=2.0,
            max_solve_age=30.0,
            max_start_drift=0.50,
            require_obstacles=False,
            is_log_CEM_progress=True,
            cem_running_log_period=2.0,
            reference_trajectory_states=['WAIT_ORBIT'],
            device='cpu',
            seed=0,
        )
        self.cfg = {k: self.declare_parameter(k, v).value for k, v in defaults.items()}
        if not isinstance(self.cfg['is_log_CEM_progress'], bool):
            raise ValueError('is_log_CEM_progress must be a boolean')

        configured_states = self.cfg['reference_trajectory_states']
        if (
            not isinstance(configured_states, (list, tuple))
            or not all(isinstance(state, str) for state in configured_states)
        ):
            raise ValueError('reference_trajectory_states must be an array of strings')
        self.reference_trajectory_states = {
            state.strip()
            for state in configured_states
            if state.strip()
        }

        # Reject invalid settings before constructing the optimizer.
        for k in (
            'plan_rate_hz', 'horizon_seconds', 'v_max', 'a_max', 'v_des',
            'odom_timeout', 'obstacle_timeout', 'max_solve_age', 'max_start_drift',
            'cem_running_log_period',
        ):
            if self.cfg[k] <= 0:
                raise ValueError(f'{k} must be positive')
        for k in (
            'goal_tolerance', 'robot_radius', 'safety_margin', 'v_min',
            'goal_clearance', 'goal_lateral_weight',
        ):
            if self.cfg[k] < 0:
                raise ValueError(f'{k} must be nonnegative')
        if not self.cfg['v_min'] <= self.cfg['v_des'] <= self.cfg['v_max']:
            raise ValueError('Require v_min <= v_des <= v_max')
        if (
            min(self.cfg['num_samples'], self.cfg['reference_samples']) < 12
            or self.cfg['num_batch'] < 20
        ):
            raise ValueError('Use >=12 trajectory/reference samples and >=20 batch trajectories')
        if self.cfg['max_obstacles'] < 1 or self.cfg['cem_iterations'] < 1:
            raise ValueError('max_obstacles and cem_iterations must be >=1')

        # Callbacks run on the ROS executor; only optimization runs in the
        # worker. A revision number identifies the reference used by each job.
        self.planner = Planner(self.cfg)

        # Warm up BEFORE creating application subscriptions and the timer.
        # If compilation fails, startup stops with jit_warm_up still False.
        # This orders startup within this node, not other ROS processes.
        self.get_logger().info(
            f"JIT warm-up started on {self.planner.device}; waiting for completion"
        )
        warm_up_started = time.perf_counter()
        self.planner.warm_up()
        self.jit_warm_up = True
        self.get_logger().info(
            f'JIT warm-up complete in {time.perf_counter() - warm_up_started:.2f} s'
        )

        self.pool = ThreadPoolExecutor(max_workers=1)
        self.future = None
        self.odom = self.path = self.obstacles = None
        self.reference_path = None
        self.linear_goal = None
        self.reference_source = 'linear'
        self.fsm_state = ''
        self.path_revision = 0
        self.progress = 0.0
        self.cleared = False
        self.warning_time = 0.0
        self.cem_sequence = 0
        self.last_cem_running_log = 0.0

        # Sensor subscriptions accept best-effort publishers. Reference and
        # output paths use reliable delivery with only the newest queued item.
        reliable = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.pub = self.create_publisher(
            Path,
            '/navigation/priest/trajectory/out',
            reliable,
        )
        self.create_subscription(
            Odometry,
            '/mavros/local_position/odom',
            self.on_odom,
            qos_profile_sensor_data,
            callback_group=self.subscription_group,
        )
        self.create_subscription(
            Path,
            '/navigation/priest/trajectory/in',
            self.on_path,
            reliable,
            callback_group=self.subscription_group,
        )
        self.create_subscription(
            AxisAlignedElipsoidArray,
            '/ellipsoids',
            self.on_obstacles,
            qos_profile_sensor_data,
            callback_group=self.subscription_group,
        )

        self.create_subscription(
            Pose,
            '/navigation/priest/linear_goal',
            self.on_linear_goal,
            reliable,
            callback_group=self.subscription_group,
        )
        self.create_subscription(
            String,
            '/mission/fsm_state',
            self.on_fsm_state,
            reliable,
            callback_group=self.subscription_group,
        )

        # This group prevents overlapping timer callbacks. The executor has
        # multiple threads; copied-array optimization has its own single worker.
        self.control_timer = self.create_timer(
            1.0 / self.cfg['plan_rate_hz'],
            self.tick,
            callback_group=self.timer_group,
        )
        self.get_logger().info(
            'PRIEST Node 3D Ready \n trajectory FSM states: '
            f'{sorted(self.reference_trajectory_states)}'
        )

    def warn(self, message):
        now = time.monotonic()
        if now - self.warning_time > 2.0:
            self.get_logger().warning(message)
            self.warning_time = now

    def cem_info(self, message):
        """Print CEM lifecycle messages when terminal logging is enabled."""
        if self.cfg['is_log_CEM_progress']:
            self.get_logger().info(message)

    def clear(self, frame='map'):
        """Publish one empty path to invalidate the previous output."""
        if not self.cleared:
            msg = Path()
            msg.header.frame_id = frame
            msg.header.stamp = self.get_clock().now().to_msg()
            self.pub.publish(msg)
            self.cleared = True

    # ------------------------------------------------------------------
    # Input callbacks: validate and cache messages for the planning timer.
    # ------------------------------------------------------------------

    @synchronized
    def on_odom(self, msg):
        """Cache position and world-frame velocity; acceleration starts at zero."""
        try:
            p, v = msg.pose.pose.position, msg.twist.twist.linear
            position = np.array([p.x, p.y, p.z], dtype=float)
            velocity = body_to_world(np.array([v.x, v.y, v.z]), msg.pose.pose.orientation)
            self.odom = (position, velocity, time.monotonic())
        except ValueError as exc:
            self.odom = None
            self.warn(str(exc))

    @synchronized
    def on_path(self, msg):
        """Cache the incoming Path and activate it only in trajectory FSM states."""
        points = np.array(
            [
                [p.pose.position.x, p.pose.position.y, p.pose.position.z]
                for p in msg.poses
            ],
            dtype=float,
        ).reshape(-1, 3)

        frame = msg.header.frame_id
        if len(points) == 0:
            self.reference_path = None
            if self.reference_source == 'path':
                self.set_active_path(None, frame)
            return

        if (
            not np.isfinite(points).all()
            or any(
                p.header.frame_id and p.header.frame_id != frame
                for p in msg.poses
            )
        ):
            self.reference_path = None
            if self.reference_source == 'path':
                self.set_active_path(None, frame)
            self.warn('Invalid reference Path: require finite positions and matching frame_ids')
            return

        # Consecutive duplicates would create zero-length segments during
        # progress projection. Keep the final return point of closed paths.
        if len(points) > 1:
            points = points[np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-6]]
        arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
        closed = (
            len(points) > 2
            and np.linalg.norm(points[-1] - points[0]) <= self.cfg['goal_tolerance']
        )
        self.reference_path = (points, arc, frame, closed)
        if self.reference_source == 'path':
            self.set_active_path(self.reference_path)

    def set_active_path(self, selected, force=False):
        """Change the planning reference and invalidate a previous solve."""
        if (
            not force
            and selected is not None
            and self.path is not None
            and self.reference_source == 'path'
            and np.array_equal(selected[0], self.path[0])
        ):
            return  # Periodic republishes preserve progress around a loop.

        self.path = selected
        self.progress = 0.0
        self.path_revision += 1
        self.clear()

    @synchronized
    def on_linear_goal(self, msg):
        """Cache a goal; only linear FSM mode uses it as a straight reference.

        Pose has no header; this script interprets it in the map frame. Goal
        orientation is not a terminal constraint of this position planner.
        """
        goal = np.array(
            [msg.position.x, msg.position.y, msg.position.z],
            dtype=np.float32,
        )
        if not np.isfinite(goal).all():
            self.warn('Ignoring a nonfinite linear goal')
            return

        if self.linear_goal is not None and np.array_equal(goal, self.linear_goal):
            return

        self.linear_goal = goal
        if self.reference_source == 'linear':
            self.set_active_path(None)

    @synchronized
    def on_fsm_state(self, msg):
        """Choose a cached Path or linear goal based on the configured states."""
        self.fsm_state = msg.data
        selected_source = (
            'path' if self.fsm_state in self.reference_trajectory_states
            else 'linear'
        )
        if selected_source == self.reference_source:
            return

        self.reference_source = selected_source
        if selected_source == 'path':
            # Even without a cached Path, bump the revision so a previous
            # linear-goal solve cannot publish after the FSM transition.
            self.set_active_path(
                self.reference_path,
                force=True,
            )
        else:
            # The timer rebuilds the straight reference from current odometry.
            self.set_active_path(None)

    def prepare_linear_reference(self):
        """Build once per new goal, after the first valid odometry is available.

        Called under state_lock by tick(). Keep the start fixed so subsequent
        replans can use the same forward-progress logic as a normal Path.
        """
        if (
            self.reference_source != 'linear'
            or self.linear_goal is None
            or self.odom is None
            or self.path is not None
        ):
            return

        # A linear goal already within 15 cm requires no trajectory. It also
        # avoids constructing a zero-length segment for path projection.
        if np.linalg.norm(self.odom[0] - self.linear_goal) <= self.cfg['goal_tolerance']:
            self.clear('map')
            return

        points = np.vstack((self.odom[0], self.linear_goal)).astype(float)
        arc = np.array([0.0, np.linalg.norm(points[1] - points[0])])
        self.set_active_path((points, arc, 'map', False))

    @synchronized
    def on_obstacles(self, msg):
        """Cache centers and radii, with the configured clearance added per axis."""
        centers = np.array(
            [[e.center.x, e.center.y, e.center.z] for e in msg.elipsoids],
            dtype=float,
        ).reshape(-1, 3)
        radii = np.array(
            [[e.radii.x, e.radii.y, e.radii.z] for e in msg.elipsoids],
            dtype=float,
        ).reshape(-1, 3)

        if (
            not np.isfinite(np.r_[centers.ravel(), radii.ravel()]).all()
            or np.any(radii <= 0)
        ):
            self.obstacles = None
            self.warn('Invalid ellipsoid message: require finite, positive radii')
            return

        radii += self.cfg['robot_radius'] + self.cfg['safety_margin']
        frame_id = 'map'
        self.obstacles = (centers, radii, frame_id, time.monotonic())

    # ------------------------------------------------------------------
    # Reference progress and the input snapshot used for the next solve.
    # ------------------------------------------------------------------

    def reference(self, position):
        """Track forward progress and resample the remaining reference path."""
        points, arc, _, closed = self.path
        if len(points) > 1:
            # Project the robot onto reference segments to estimate arc length.
            delta = np.diff(points, axis=0)
            length = np.diff(arc)
            t = np.clip(
                np.sum((position - points[:-1]) * delta, axis=1) / length**2,
                0,
                1,
            )
            projected = points[:-1] + t[:, None] * delta
            candidates = arc[:-1] + t * length
            distance = np.linalg.norm(projected - position, axis=1)

            # Restrict the search to nearby forward progress to reduce jumps
            # to later parts of a circle or a path that crosses itself.
            allowed = (candidates >= self.progress - 0.20) & (
                candidates
                <= self.progress + self.cfg['v_des'] * self.cfg['horizon_seconds']
            )
            if np.any(allowed):
                best = np.argmin(np.where(allowed, distance, np.inf))
                self.progress = max(self.progress, float(candidates[best]))

        # A closed path ends near its start, so proximity alone would stop it
        # immediately. Require progress to the end of the loop as well.
        arrived = np.linalg.norm(position - points[-1]) <= self.cfg['goal_tolerance']
        if closed:
            arrived = arrived and arc[-1] - self.progress <= self.cfg['goal_tolerance']

        # Keep a fixed reference size for JAX, even as the remaining path shrinks.
        start = np.array([np.interp(self.progress, arc, points[:, i]) for i in range(3)])
        remaining = np.vstack([start, points[arc > self.progress + 1e-6]])
        if len(remaining) == 1:
            remaining = np.vstack([position, points[-1]])
        return resample(remaining, self.cfg['reference_samples']), arrived

    def inputs(self):
        """Return a usable snapshot only when frames and receive ages agree."""
        now = time.monotonic()
        if self.odom is None or self.path is None:
            return None

        position, velocity, updated = self.odom
        if now - updated > self.cfg['odom_timeout']:
            self.warn('Stale odometry or odometry/reference frame mismatch')
            return None

        if self.obstacles is None:
            if self.cfg['require_obstacles']:
                self.warn('Waiting for a valid ellipsoid message (an empty array is valid)')
                return None

            centers, radii = np.empty((0, 3)), np.empty((0, 3))
        else:
            centers, radii, obs_frame, updated = self.obstacles
            if now - updated > self.cfg['obstacle_timeout']:
                self.warn('Stale obstacles or obstacle/odometry frame mismatch')
                return None

        dropped_obstacles = max(
            0,
            len(centers) - self.cfg['max_obstacles'],
        )
        if dropped_obstacles:
            # Centers and radii must use the same indices. Keep obstacles
            # closest to the current robot position because they are the most
            # relevant over the local planning horizon.
            distance = np.linalg.norm(centers - position[None, :], axis=1)
            keep = np.argsort(distance)[:self.cfg['max_obstacles']]
            centers = centers[keep]
            radii = radii[keep]

            self.warn(
                f'Received {len(distance)} obstacles; keeping the nearest '
                f'{len(centers)} and dropping {dropped_obstacles}'
            )

        return (
            position,
            velocity,
            self.path[2],
            centers,
            radii,
            dropped_obstacles,
        )

    # ------------------------------------------------------------------
    # Planning lifecycle: collect, validate, publish, then schedule again.
    # ------------------------------------------------------------------

    @synchronized
    def tick(self):
        """Poll the worker without blocking ROS subscription callbacks."""
        if not self.jit_warm_up:
            return

        self.prepare_linear_reference()
        data = self.inputs()
        completed = self.future if self.future is not None and self.future.done() else None
        if completed is not None:
            self.future = None
        elif self.future is not None:
            now = time.monotonic()
            if now - self.last_cem_running_log >= self.cfg['cem_running_log_period']:
                solve_id = self.job[3]
                elapsed_ms = (now - self.job[1]) * 1000.0
                self.cem_info(
                    f'CEM #{solve_id} is running ({elapsed_ms:.1f} ms elapsed)'
                )
                self.last_cem_running_log = now

        # Drain a finished job even if an input has become unavailable.
        # Its output must not be published against missing or stale inputs.
        if data is None:
            self.clear(self.path[2] if self.path is not None else '')
            if completed is not None:
                _, started, _, solve_id, kept_count, dropped_count = self.job
                try:
                    _, goal_corrected = completed.result()
                    self.cem_info(
                        f'CEM #{solve_id} finished in '
                        f'{(time.monotonic() - started) * 1000.0:.1f} ms, '
                        f'but input data became unavailable; result discarded '
                        f'(lookahead_goal_corrected={goal_corrected}, '
                        f'obstacles kept={kept_count}, dropped={dropped_count})'
                    )
                except Exception as exc:
                    self.cem_info(
                        f'CEM #{solve_id} failed after '
                        f'{(time.monotonic() - started) * 1000.0:.1f} ms'
                    )
                    self.warn(f'PRIEST solve failed: {exc}')
            return

        position, velocity, frame, centers, radii, dropped_obstacles = data
        reference, arrived = self.reference(position)

        # At the goal, clear the last output and discard any completed solve.
        # An already-running job may finish, but no new job is scheduled.
        if arrived:
            self.clear()
            if completed is not None:
                _, started, _, solve_id, kept_count, dropped_count = self.job
                try:
                    _, goal_corrected = completed.result()
                    self.cem_info(
                        f'CEM #{solve_id} finished in '
                        f'{(time.monotonic() - started) * 1000.0:.1f} ms, '
                        f'but the goal is within {self.cfg["goal_tolerance"]:.2f} m; '
                        f'result discarded '
                        f'(lookahead_goal_corrected={goal_corrected}, '
                        f'obstacles kept={kept_count}, '
                        f'dropped={dropped_count})'
                    )
                except Exception as exc:
                    self.cem_info(
                        f'CEM #{solve_id} failed after '
                        f'{(time.monotonic() - started) * 1000.0:.1f} ms'
                    )
                    self.warn(f'PRIEST solve failed: {exc}')
            return  # No new solve within goal_tolerance; discard an in-flight result.

        # Inputs may change while CEM runs. Check reference revision, robot
        # movement, and the latest obstacles before accepting the result.
        if completed is not None:
            try:
                points, goal_corrected = completed.result()
                (
                    revision,
                    started,
                    start_position,
                    solve_id,
                    kept_count,
                    dropped_count,
                ) = self.job
                elapsed_ms = (time.monotonic() - started) * 1000.0
                if revision != self.path_revision:
                    self.cem_info(
                        f'CEM #{solve_id} finished in {elapsed_ms:.1f} ms; '
                        'reference changed, result discarded; '
                        f'lookahead_goal_corrected={goal_corrected}'
                    )
                elif (
                    time.monotonic() - started > self.cfg['max_solve_age']
                    or np.linalg.norm(position - start_position)
                    > self.cfg['max_start_drift']
                ):
                    self.clear()
                    self.cem_info(
                        f'CEM #{solve_id} finished in {elapsed_ms:.1f} ms; '
                        'result discarded as stale; '
                        f'lookahead_goal_corrected={goal_corrected}'
                    )
                    self.warn('Discarded stale planner output')
                elif (
                    not collision_free(points, centers, radii)
                    or not collision_free(
                        np.vstack([position, points[0]]), centers, radii
                    )
                ):
                    self.clear()
                    self.cem_info(
                        f'CEM #{solve_id} finished in {elapsed_ms:.1f} ms; '
                        'result failed the latest collision check; '
                        f'lookahead_goal_corrected={goal_corrected}'
                    )
                    self.warn('Discarded output that intersects the latest obstacles')
                else:
                    self.publish(points)
                    self.cem_info(
                        f'CEM #{solve_id} finished in {elapsed_ms:.1f} ms; '
                        f'published {len(points)} poses '
                        f'(lookahead_goal_corrected={goal_corrected}, '
                        f'obstacles kept={kept_count}, dropped={dropped_count})'
                    )
            except Exception as exc:
                self.clear()
                solve_id = self.job[3]
                elapsed_ms = (time.monotonic() - self.job[1]) * 1000.0
                self.cem_info(
                    f'CEM #{solve_id} failed after {elapsed_ms:.1f} ms'
                )
                self.warn(f'PRIEST solve failed: {exc}')

        # Only one solve runs at a time. Copy inputs so callbacks can continue
        # updating their caches without changing the worker's snapshot.
        if self.future is None:
            state = np.r_[position, velocity, np.zeros(3)].astype(np.float32)
            self.cem_sequence += 1
            solve_id = self.cem_sequence
            started = time.monotonic()
            self.last_cem_running_log = started
            self.job = (
                self.path_revision,
                started,
                position.copy(),
                solve_id,
                len(centers),
                dropped_obstacles,
            )
            self.cem_info(
                f'CEM #{solve_id} started: state={self.fsm_state!r}, '
                f'reference={self.reference_source}, '
                f'obstacles kept={len(centers)}, dropped={dropped_obstacles}'
            )
            self.future = self.pool.submit(
                self.planner.solve,
                state,
                reference.astype(np.float32),
                centers.copy(),
                radii.copy(),
            )

    def publish(self, points, frame = 'map'):
        """Convert samples to a timed Path with yaw aligned to its XY tangent."""
        msg = Path()
        msg.header.frame_id = frame
        now = self.get_clock().now()
        msg.header.stamp = now.to_msg()

        # Position samples span the horizon. Orientation conveys heading only;
        # this node does not produce roll/pitch or vehicle control commands.
        dt = self.cfg['horizon_seconds'] / (len(points) - 1)
        for i, point in enumerate(points):
            pose = PoseStamped()
            pose.header.frame_id = frame
            pose.header.stamp = (now + Duration(seconds=i * dt)).to_msg()
            pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = map(float, point)
            direction = points[min(i + 1, len(points) - 1)] - points[max(0, i - 1)]
            yaw = math.atan2(direction[1], direction[0])
            pose.pose.orientation.z, pose.pose.orientation.w = (
                math.sin(yaw / 2),
                math.cos(yaw / 2),
            )
            msg.poses.append(pose)

        self.pub.publish(msg)
        self.cleared = False

    def destroy_node(self):
        # Finish the active worker before releasing the node's ROS resources.
        self.pool.shutdown(wait=True, cancel_futures=True)
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    executor = None
    try:
        node = Priest3DNode()
        executor = MultiThreadedExecutor(num_threads=2)
        executor.add_node(node)
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        # Stop callbacks before waiting for the optimizer and releasing ROS.
        if executor is not None:
            executor.shutdown()
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
