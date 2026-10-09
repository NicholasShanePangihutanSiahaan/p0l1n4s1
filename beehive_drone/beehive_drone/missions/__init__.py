from beehive_drone.missions.manual_spray_mission import ManualSprayMission
from beehive_drone.missions.basic_orbit import BasicOrbitMission
from beehive_drone.missions.approach_flower_mission import ApproachFlowerMission
from beehive_drone.missions.virtual_tree_test import VirtualTreeTestMission

# Dictionary mapping mission identifiers to strategy classes
MISSION_STRATEGIES = {
    # Add new missions here: 
    # <mission_name> : <mission_class>
    # 'survey_mission': SurveyMissionStrategy,
    'manual_spray_mission': ManualSprayMission,
    'basic_orbit': BasicOrbitMission, 
    'virtual_tree_test': VirtualTreeTestMission,
    'approach_flower': ApproachFlowerMission
}
