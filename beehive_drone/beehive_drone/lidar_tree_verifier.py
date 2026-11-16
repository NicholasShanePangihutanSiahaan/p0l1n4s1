"""Verify a camera candidate using fresh, timestamped LiDAR geometry."""
from collections import deque
import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Float64MultiArray
from tf2_ros import Buffer, TransformListener, TransformException
from beehive_drone.lidar_trunk_geometry import fit_trunk, GEOMETRY_DEFAULTS, validate_geometry
from rcl_interfaces.msg import ParameterDescriptor


class Verifier(Node):
    def __init__(self):
        super().__init__('lidar_tree_verifier')
        defaults=dict(GEOMETRY_DEFAULTS, scan_window=5, min_consistent_scans=3,
            processing_period=0.15, scan_timeout=1.0, candidate_timeout=1.0,
            max_scan_gap=1.0, scan_center_tolerance=0.15, max_height_variation=0.5,
            lidar_topic='/livox/top/points', target_frame='odom')
        for name,value in defaults.items():
            self.declare_parameter(name,value,ParameterDescriptor(read_only=True))
        self.settings={name:self.get_parameter(name).value for name in defaults}
        self.geometry={name:self.settings[name] for name in GEOMETRY_DEFAULTS}
        validate_geometry(self.geometry)
        for name in ('processing_period','scan_timeout','candidate_timeout','max_scan_gap',
                     'scan_center_tolerance','max_height_variation'):
            if not np.isfinite(self.settings[name]) or self.settings[name]<=0:
                raise ValueError(f'{name} must be positive and finite')
        if not 2<=self.settings['min_consistent_scans']<=self.settings['scan_window']:
            raise ValueError('Require 2 <= min_consistent_scans <= scan_window')
        self.buffer=Buffer();self.listener=TransformListener(self.buffer,self)
        self.candidate=None;self.samples=deque(maxlen=self.settings["scan_window"]);self.last_stamp=None
        self.pending=None;self.pending_time=0.;self.last_fit_time=0.
        self.create_timer(self.settings["processing_period"],self.process_pending)
        self.create_subscription(Float64MultiArray,'/mission/lidar_candidate',self.request,10)
        self.create_subscription(PointCloud2,self.settings['lidar_topic'],self.scan,qos_profile_sensor_data)
        self.pub=self.create_publisher(Float64MultiArray,'/perception/lidar_verified_tree',10)

    def request(self,msg):
        if len(msg.data)!=4 or not all(np.isfinite(msg.data)):return
        key=int(msg.data[0])
        if self.candidate is None or key!=self.candidate[0]:self.samples.clear()
        self.candidate=(key,tuple(msg.data[1:]),self.get_clock().now())

    def scan(self,msg):
        self.pending=msg;self.pending_time=time.monotonic()

    def process_pending(self):
        if self.pending is None or time.monotonic()-self.pending_time>self.settings["scan_timeout"]:return
        self.process_scan(self.pending)

    def process_scan(self,msg):
        if self.candidate is None:return
        key,c,requested_at=self.candidate
        age=(self.get_clock().now()-requested_at).nanoseconds/1e9
        if age<0 or age>self.settings["candidate_timeout"]:self.samples.clear();return
        stamp=(msg.header.stamp.sec,msg.header.stamp.nanosec)
        if stamp==self.last_stamp:return
        try:t=self.buffer.lookup_transform(self.settings["target_frame"],msg.header.frame_id,Time.from_msg(msg.header.stamp)).transform
        except TransformException:
            return
        self.last_stamp=stamp
        if time.monotonic()-self.last_fit_time>self.settings["max_scan_gap"]:self.samples.clear()
        raw=point_cloud2.read_points(msg,field_names=('x','y','z'),skip_nans=True)
        p=np.column_stack([raw[n].reshape(-1) for n in ('x','y','z')]) if isinstance(raw,np.ndarray) and raw.dtype.names else np.asarray(list(raw),dtype=float)
        if p.size==0:return
        q=t.rotation;v=np.array([q.x,q.y,q.z]);w=q.w
        rotation=(w*w-v@v)*np.eye(3)+2*np.outer(v,v)+2*w*np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
        p=p@rotation.T+np.array([t.translation.x,t.translation.y,t.translation.z])
        fit=fit_trunk(p,c,**self.geometry)
        if fit is None:self.samples.clear();return
        if self.samples and np.linalg.norm(fit[:2]-self.samples[-1][:2])>self.settings["scan_center_tolerance"]:self.samples.clear()
        self.last_fit_time=time.monotonic()
        self.samples.append(fit)
        if len(self.samples)<self.settings["min_consistent_scans"]:return
        values=np.array(self.samples)
        if np.ptp(values[:,3])>self.settings["max_height_variation"]:return
        fit=np.median(values,axis=0)
        self.pub.publish(Float64MultiArray(data=[float(key),*map(float,fit)]))


def main():
    rclpy.init();node=Verifier()
    try:rclpy.spin(node)
    except KeyboardInterrupt:pass
    finally:node.destroy_node();rclpy.shutdown()
