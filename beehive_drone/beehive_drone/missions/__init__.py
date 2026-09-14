from beehive_drone.missions.manual_spray_mission import ManualSprayMission
from beehive_drone.missions.basic_orbit import BasicOrbitMission

# Dictionary mapping mission identifiers to strategy classes
MISSION_STRATEGIES = {
    # Add new missions here: 
    # <mission_name> : <mission_class>
    # 'survey_mission': SurveyMissionStrategy,
    'manual_spray_mission': ManualSprayMission,
    'basic_orbit': BasicOrbitMission, 
}