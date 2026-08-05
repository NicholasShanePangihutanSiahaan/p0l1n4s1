import threading
import sys

import numpy as np
import open3d as o3d
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, qos_profile_services_default

from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
from sensor_msgs_py import point_cloud2

from pcl_cstm_msg.msg import PointCloudArray, TrackedCylinderArray, VCylindersFit, TrackedCylinderArray, VNormals


CLUSTER_COLORS = [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 1.0, 0.0],
    [1.0, 0.0, 1.0],
    [0.0, 1.0, 1.0],
    [0.5, 0.0, 0.0],
    [0.0, 0.5, 0.0],
    [0.0, 0.0, 0.5],
    [0.5, 0.5, 0.0],
    [0.5, 0.0, 0.5],
    [0.0, 0.5, 0.5],
]


def _make_cylinder_wireframe(center_x, center_y, center_z, radius, height, color, orientation_q):
    mesh = o3d.geometry.TriangleMesh.create_cylinder(
        radius=radius, height=height, resolution=20)
    r = o3d.geometry.get_rotation_matrix_from_quaternion(
        [orientation_q.w, orientation_q.x, orientation_q.y, orientation_q.z])
    mesh.rotate(r, center=[0.0, 0.0, 0.0])
    mesh.translate([center_x, center_y, center_z])

    lines = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    lines.paint_uniform_color(color)
    return lines


def _make_drone_arrow(position, orientation_q):
    qw, qx, qy, qz = orientation_q
    r_align = o3d.geometry.get_rotation_matrix_from_axis_angle(
        [0.0, np.pi / 2.0, 0.0])
    r_odom = o3d.geometry.get_rotation_matrix_from_quaternion(
        [qw, qx, qy, qz])
    r = np.dot(r_odom, r_align)

    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=0.1,
        cone_radius=0.25,
        cylinder_height=0.5,
        cone_height=0.3,
        resolution=20)
    arrow.rotate(r, center=[0.0, 0.0, 0.0])
    arrow.translate([position.x, position.y, position.z])
    arrow.paint_uniform_color([1.0, 0.8, 0.0])
    return arrow


class VoxelVisualizer(Node):

    def __init__(self):
        super().__init__('voxel_visualizer')

        self.declare_parameter('voxel_size', 0.1)
        self.voxel_size = self.get_parameter('voxel_size').value

        self._lock = threading.Lock()
        
        # --- PAUSE STATE ---
        self.is_paused = False

        self._latest_cloud = None
        self._latest_clusters = None
        self._latest_cylinders = None
        self._latest_tracked_cylinders = None
        self._latest_normals = None
        self._latest_origins = None
        self._latest_odom = None
        self._first_frame = True

        self._vis = o3d.visualization.VisualizerWithKeyCallback()
        self._vis.create_window(window_name='Point Cloud Voxels (Press ENTER to Pause)',
                                width=1024, height=768)

        # Register ENTER key in Open3D window (GLFW codes: 257 = Enter, 335 = Numpad Enter)
        self._vis.register_key_callback(257, self._toggle_pause)
        self._vis.register_key_callback(335, self._toggle_pause)

        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        self._vis.add_geometry(axes)
        self._geometries = [axes]

        self._drone_arrow = None

        self._sub = self.create_subscription(
            PointCloud2,
            '/input_cloud',
            self._cloud_callback,
            qos_profile_sensor_data)

        self._sub_clusters = self.create_subscription(
            PointCloudArray,
            '/clusters',
            self._clusters_callback,
            qos_profile_sensor_data)

        self._sub_cylinders = self.create_subscription(
            VCylindersFit,
            '/cylinders',
            self._cylinders_callback,
            qos_profile_sensor_data)

        self._sub_odom = self.create_subscription(
            Odometry,
            '/odom',
            self._odom_callback,
            qos_profile_services_default)

        self._sub_tracked_cylinders = self.create_subscription(
            TrackedCylinderArray,
            '/global_cylinders',
            self._tracked_cylinders_callback,
            qos_profile_sensor_data)

        self._sub_normals = self.create_subscription(
            VNormals,
            '/normals',
            self._normals_callback,
            qos_profile_sensor_data)

        # --- TERMINAL INPUT LISTENER THREAD ---
        self._input_thread = threading.Thread(target=self._wait_for_terminal_enter, daemon=True)
        self._input_thread.start()
        
        self.get_logger().info('Visualizer started. Press ENTER in the terminal or Open3D window to pause/resume.')

    def _toggle_pause(self, vis=None):
        """ Toggles the pause state and logs it. """
        self.is_paused = not self.is_paused
        state = "PAUSED (Ignoring incoming data)" if self.is_paused else "RESUMED (Listening to topics)"
        self.get_logger().info(f"\n\n---> [ENTER PRESSED] SUBSCRIPTIONS {state} <---\n")
        return False

    def _wait_for_terminal_enter(self):
        """ Background thread that waits for terminal 'Enter' presses. """
        while rclpy.ok():
            try:
                sys.stdin.readline()
                self._toggle_pause()
            except Exception:
                break

    def _cloud_callback(self, msg: PointCloud2):
        if self.is_paused:
            return
            
        self.get_logger().info('First point cloud received!', once=True)

        points = np.array([
            (p[0], p[1], p[2])
            for p in point_cloud2.read_points(msg, field_names=['x', 'y', 'z'])
        ], dtype=np.float32)

        if points.shape[0] == 0:
            self.get_logger().warn('Received an empty point cloud!')
            return

        o3d_cloud = o3d.geometry.PointCloud()
        o3d_cloud.points = o3d.utility.Vector3dVector(points)

        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
            o3d_cloud, voxel_size=self.voxel_size)

        with self._lock:
            self._latest_cloud = voxel_grid

    def _normals_callback(self, msg: VNormals):
        if self.is_paused:
            return
            
        if not msg.normals:
            return
        normals = []
        origins = []
        for i, normal in enumerate(msg.normals):
            normals.append(normal)
        for i, origin in enumerate(msg.origins):
            origins.append(origin)
        if normals and origins:
            with self._lock:
                self._latest_normals = normals
                self._latest_origins = origins

    def _clusters_callback(self, msg: PointCloudArray):
        if self.is_paused:
            return
            
        if not msg.clouds:
            return

        self.get_logger().info(
            f'Received {len(msg.clouds)} clusters', throttle_duration_sec=2.0)

        cluster_voxels = []
        for i, cloud_msg in enumerate(msg.clouds):
            points = np.array([
                (p[0], p[1], p[2])
                for p in point_cloud2.read_points(
                    cloud_msg, field_names=['x', 'y', 'z'])
            ], dtype=np.float32)

            if points.shape[0] == 0:
                continue

            o3d_cloud = o3d.geometry.PointCloud()
            o3d_cloud.points = o3d.utility.Vector3dVector(points)

            color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
            o3d_cloud.paint_uniform_color(color)

            voxel = o3d.geometry.VoxelGrid.create_from_point_cloud(
                o3d_cloud, voxel_size=self.voxel_size)

            cluster_voxels.append(voxel)

        if cluster_voxels:
            with self._lock:
                self._latest_clusters = cluster_voxels

    def _cylinders_callback(self, msg: VCylindersFit):
        if self.is_paused:
            return
            
        if not msg.cylinders:
            return

        wireframes = []
        clusters_voxel = []
        total = len(msg.cylinders)
        valid = 0
        for i, cyl in enumerate(msg.cylinders):
            if cyl.clouds.width > 0 and cyl.clouds.fields:
                self.get_logger().info(
                            f'Have cluster, {cyl.clouds.width}')   
                points = np.array([
                    (p[0], p[1], p[2])
                    for p in point_cloud2.read_points(
                        cyl.clouds, field_names=['x', 'y', 'z'])
                ], dtype=np.float32)
    
                if points.shape[0] == 0:
                    continue
                
                o3d_cloud = o3d.geometry.PointCloud()
                o3d_cloud.points = o3d.utility.Vector3dVector(points)
    
                color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
                o3d_cloud.paint_uniform_color(color)
    
                voxel = o3d.geometry.VoxelGrid.create_from_point_cloud(
                    o3d_cloud, voxel_size=self.voxel_size)
    
                # clusters_voxel.append(voxel)
            if cyl.is_valid:
                valid += 1
                color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
                lines = _make_cylinder_wireframe(
                    center_x=cyl.pose.position.x,
                    center_y=cyl.pose.position.y,
                    center_z=cyl.pose.position.z,
                    radius=cyl.radius,
                    height=cyl.height,
                    color=color,
                    orientation_q=cyl.pose.orientation)
                wireframes.append(lines)
            else:
                continue
        self.get_logger().info(
            f'Received {total} cylinders, valid: {valid}')         
        if wireframes:
            with self._lock:
                self._latest_cylinders = wireframes 
                # self._latest_clusters = clusters_voxel

    def _tracked_cylinders_callback(self, msg: TrackedCylinderArray):
        return
        if self.is_paused:
            return
            
        if not msg.cylinders:
            return

        wireframes = []
        for tc in msg.cylinders:
            cyl = tc.cylinder
            lines = _make_cylinder_wireframe(
                center_x=cyl.pose.position.x,
                center_y=cyl.pose.position.y,
                center_z=cyl.pose.position.z,
                radius=cyl.radius,
                height=cyl.height,
                color=[0.9, 0.9, 0.9],
                orientation_q=cyl.pose.orientation)
            wireframes.append(lines)

        if wireframes:
            with self._lock:
                self._latest_tracked_cylinders = wireframes

    def _odom_callback(self, msg: Odometry):
        if self.is_paused:
            return
            
        with self._lock:
            self._latest_odom = msg

    def render_step(self):
        clusters = None
        cylinders = None
        tracked_cylinders = None
        cloud = None
        odom = None
        normals = None
        origins = None
        with self._lock:
            if self._latest_clusters is not None:
                
                clusters = self._latest_clusters
                self._latest_clusters = None
            if self._latest_cloud is not None:
                cloud = self._latest_cloud
                self._latest_cloud = None
            if self._latest_odom is not None:
                odom = self._latest_odom
            if self._latest_tracked_cylinders is not None:
                tracked_cylinders = self._latest_tracked_cylinders
                self._latest_tracked_cylinders = None
            if self._latest_cylinders is not None:
                cylinders = self._latest_cylinders
                self._latest_cylinders = None
            if self._latest_normals is not None and self._latest_origins is not None:
                normals = self._latest_normals
                origins = self._latest_origins
                self._latest_normals = None
                self._latest_origins = None

        if odom is not None:
            self._render_drone(odom)

        if clusters is not None:
            self.get_logger().info(
                                        f'Render cluster')
            self._render_clusters(clusters)
        if cylinders is not None:
            self._render_cylinders(cylinders)
        if tracked_cylinders:
            self._render_tracked_cylinders(tracked_cylinders)
        if cloud is not None and False:
            self._render_point_cloud(cloud)
        if normals is not None and origins is not None:
            self._render_normals(normals, origins)
        
        self._vis.poll_events()
        self._vis.update_renderer()

    def _render_normals(self, normals, origins):
        while len(self._geometries) > 1:
            self._vis.remove_geometry(self._geometries.pop(), False)

        np_origins = np.array([[p.x, p.y, p.z] for p in origins], dtype=np.float64)
        np_normals = np.array([[n.x, n.y, n.z] for n in normals], dtype=np.float64)
        pcd = o3d.geometry.PointCloud()

        if len(np_origins) > 0:
            pcd.points = o3d.utility.Vector3dVector(np_origins)
            pcd.normals = o3d.utility.Vector3dVector(np_normals)
        opt = self._vis.get_render_option()
        opt.point_show_normal = True

        self._vis.add_geometry(pcd, reset_bounding_box=self._first_frame)
        self._first_frame = False

        self._geometries.append(pcd)

    def _render_point_cloud(self, voxel):
        while len(self._geometries) > 1:
            self._vis.remove_geometry(self._geometries.pop(), False)

        self._vis.add_geometry(voxel, reset_bounding_box=self._first_frame)
        self._first_frame = False

        self._geometries.append(voxel)

    def _render_clusters(self, cluster_voxels):
        while len(self._geometries) > 1:
            self._vis.remove_geometry(self._geometries.pop(), False)

        for voxel in cluster_voxels:
            self._vis.add_geometry(voxel, reset_bounding_box=self._first_frame)
            self._first_frame = False
            self._geometries.append(voxel)


    def _render_cylinders(self, cylinder_wireframes):
        while len(self._geometries) > 1:
            self._vis.remove_geometry(self._geometries.pop(), False)

        self.get_logger().info(
            f'Received {len(cylinder_wireframes)} cylinders')
        for cylinder in cylinder_wireframes:

            self._vis.add_geometry(cylinder, reset_bounding_box=self._first_frame)
            self._first_frame = False
            self._geometries.append(cylinder)


    def _render_tracked_cylinders(self, tracked_cylinder_wireframes=None):
        if tracked_cylinder_wireframes:
            self.get_logger().info(
                        f'Received {len(tracked_cylinder_wireframes)} tracked cylinders')
            for lines in tracked_cylinder_wireframes:
                self._vis.add_geometry(lines, reset_bounding_box=self._first_frame)
                self._first_frame = False
                self._geometries.append(lines)


    def _render_drone(self, odom_msg):
        if self._drone_arrow is not None:
            self._vis.remove_geometry(self._drone_arrow, False)

        q = odom_msg.pose.pose.orientation
        p = odom_msg.pose.pose.position
        self._drone_arrow = _make_drone_arrow(p, (q.w, q.x, q.y, q.z))

        self._vis.add_geometry(self._drone_arrow, False)

    def destroy_node(self):
        self._vis.destroy_window()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = VoxelVisualizer()

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)
            node.render_step()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()