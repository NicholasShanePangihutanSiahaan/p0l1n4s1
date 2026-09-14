from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from beehive_drone.mission_state_machine import MissionStateMachine

class BaseMissionStrategy(ABC):
    @property
    @abstractmethod
    def navigation_states(self) -> Sequence[str]:
        """State names where position setpoints are actively published."""
        pass

    @property
    @abstractmethod
    def timeout_exempt_states(self) -> Sequence[str]:
        """State names exempt from the state duration timeout watchdog."""
        pass
    
    @abstractmethod
    def execute(self, fsm: MissionStateMachine, active: bool, elapsed: float, cx: float, cy: float):
        """
        Execute mission state logic for the current iteration.
        
        :param fsm: Reference to the MissionStateMachine node instance.
        :param active: Boolean indicating whether the FSM is in an active state.
        :param elapsed: Elapsed time in seconds since entering the current state.
        :param cx: Current local X position.
        :param cy: Current local Y position.
        """
        pass