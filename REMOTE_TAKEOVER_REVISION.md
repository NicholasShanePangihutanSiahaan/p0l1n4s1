# Remote Takeover Revision

Changed runtime files:

- `beehive_drone/mission_state_machine.py`
- `beehive_drone/mission_state_machine_single_tree.py`
- `beehive_drone/dynamic_orbit_controller.py`
- `beehive_drone/vortex_avoidance_controller.py`
- `beehive_drone/velocity_controller.py`
- `beehive_drone/mission_params.py`
- `config/mission_real_pcl.yaml`

Behavior:

1. An RC mode change away from `GUIDED` is confirmed for 0.30 seconds.
2. FSM latches `PILOT_OVERRIDE`.
3. Orbit is stopped and the active tree is cleared.
4. Orbit, avoidance, and velocity nodes stop publishing flight targets.
5. The FSM does not issue mode, goal, return-home, or landing commands.
6. Switching back to `GUIDED` does not resume autonomy. Land/disarm and restart.
