"""Basic orbit mission with one detected tree and one synthetic target."""

import math
from copy import deepcopy

from uav_interfaces.msg import Tree

from beehive_drone.missions.basic_orbit import BasicOrbitMission


def homeward_point(tree_xy, home_xy, offset):
    dx = float(home_xy[0]) - float(tree_xy[0])
    dy = float(home_xy[1]) - float(tree_xy[1])
    distance = math.hypot(dx, dy)
    if distance < 1.0e-6:
        raise ValueError('tree and home positions are coincident')
    scale = float(offset) / distance
    return tree_xy[0] + scale * dx, tree_xy[1] + scale * dy


class VirtualTreeTestMission(BasicOrbitMission):
    """Keep BasicOrbit control, adding a virtual second tree only."""

    def __init__(self):
        super().__init__()
        self.virtual_tree = None

    def is_virtual(self, tree, fsm=None):
        if tree is None:
            return False
        virtual_id = 9001 if fsm is None else getattr(
            fsm, 'virtual_tree_id', 9001)
        return int(tree.id) == int(virtual_id)

    def decorate_tree_map(self, trees):
        result = list(trees)
        if self.virtual_tree is not None and not any(
                int(tree.id) == int(self.virtual_tree.id) for tree in result):
            result.append(deepcopy(self.virtual_tree))
        return result

    def tree_completed(self, completed_tree, fsm):
        if completed_tree is None or self.is_virtual(completed_tree, fsm) or \
                self.virtual_tree is not None:
            return None
        if fsm.home_pose is None:
            raise ValueError('home pose is unavailable')

        mode = fsm.virtual_tree_position_mode
        if mode == 'toward_home':
            virtual_x, virtual_y = homeward_point(
                (float(completed_tree.x), float(completed_tree.y)),
                (float(fsm.home_pose[0]), float(fsm.home_pose[1])),
                fsm.virtual_tree_offset_toward_home)
        elif mode == 'home_relative':
            virtual_x = float(fsm.home_pose[0]) + fsm.virtual_tree_position_x
            virtual_y = float(fsm.home_pose[1]) + fsm.virtual_tree_position_y
        elif mode == 'map':
            virtual_x = fsm.virtual_tree_position_x
            virtual_y = fsm.virtual_tree_position_y
        else:
            raise ValueError(
                'virtual_tree_position_mode harus toward_home, '
                'home_relative, atau map')

        target = Tree()
        target.id = int(fsm.virtual_tree_id)
        target.x = float(virtual_x)
        target.y = float(virtual_y)
        target.z = float(completed_tree.z)
        target.confidence = 1.0
        target.inspected = False
        target.validated = True
        target.orbit_count = 0
        self.virtual_tree = target
        return deepcopy(target)
