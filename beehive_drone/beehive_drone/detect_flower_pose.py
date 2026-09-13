import math

import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.node import Node

from zed_msgs.msg import ObjectsStamped
from geometry_msgs.msg import PoseStamped, Point, Pose
from uav_interfaces.msg import Tree, ActiveTree
from pcl_cstm_msg.msg import TrackedCylinderArray

class DetectFlowerNode(Node):
  def __init__(self):
    super().__init__("detect_flower_node")
    qos_reliable = QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST,
                depth=1
            )

    qos_sensor = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=10
            )
    self.current_tree = None
    self.flower_pose = None
    self.flower_collection = []
    
    # Subscribers
    self.tree_sub = self.create_subscription(ActiveTree, "/mission/current_tree", self.tree_cb, qos_reliable) # Deteksi tree yang lagi aktif di search
    self.global_det_sub = self.create_subscription(
      TrackedCylinderArray, "/global_cylinders", 
      self.obj_det, 
      QoSProfile(
          reliability=ReliabilityPolicy.BEST_EFFORT,
          history=HistoryPolicy.KEEP_LAST,
          depth=1
      )
    ) # Deteksi bunga yang didapatkan bb_proc
    
    # Publishers
    self.flower_pub = self.create_publisher(
        Pose,
        "/mission/current_flower",
        10
    )
  
  def obj_det(self,msg):
    flower_collection_temp = []
    for tracked in msg.cylinders:
      if tracked.type != 'bunga':
        continue
      else:
        point = Point()
        point.x = tracked.cylinder.pose.position.x
        point.y = tracked.cylinder.pose.position.y
        point.z = tracked.cylinder.pose.position.z
        flower_collection_temp.append(point)
    self.flower_collection - flower_collection_temp
  
  def tree_cb(self, msg):
    if msg.is_currenty_orbiting:
      self.current_tree = msg.tree
      
      if self.current_tree is not None:
        radius_tolerance = 4.0
        tree_x = self.current_tree.pose.position.x
        tree_y = self.current_tree.pose.position.y
        tree_z = self.current_tree.pose.position.z
        for flower in self.flower_collection:
          distance = math.sqrt(
              (flower.x - tree_x) ** 2 +
              (flower.y - tree_y) ** 2 +
              (flower.z - tree_z) ** 2
          )
          if distance <= radius_tolerance:
            self.flower_pose = Pose()
            self.flower_pose.position.x = flower.x
            self.flower_pose.position.y = flower.y
            self.flower_pose.position.z = flower.z
            self.flower_pub(self.flower_pose)
            pass
          else:
            continue
    else:
      self.current_tree = None
      self.flower_pose = None