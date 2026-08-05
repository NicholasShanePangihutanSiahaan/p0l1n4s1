#!/usr/bin/env python3
"""Compatibility alias for the active single-tree mission state machine."""

from beehive_drone.mission_state_machine import MissionStateMachine, main

__all__ = ["MissionStateMachine", "main"]


if __name__ == "__main__":
    main()
