#include <mutex>
#include <vector>
#include <memory>
#include <string>
#include <algorithm>
#include <cmath>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/odometry.hpp>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/console/print.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/random_sample.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "pcl_cstm_msg/msg/point_cloud_array.hpp"
#include "pcl_cstm_msg/msg/v_cylinders_fit.hpp"
#include "pcl_cstm_msg/msg/tracked_cylinder_array.hpp"
#include "point-cloud-test/pcl_processor.h"
#include "point-cloud-test/pcl_filter.h"
#include "point-cloud-test/global_cylinder_manager.hpp"

namespace point_cloud_test
{

  struct CloudOdomPair
  {
    sensor_msgs::msg::PointCloud2::ConstSharedPtr cloud;
    nav_msgs::msg::Odometry::ConstSharedPtr odom;
  };

  class PclProcNode : public rclcpp::Node
  {
  public:
    PclProcNode()
        : Node("pcl_proc_node")
    {
      output_frame_ = declare_parameter<std::string>("output_frame", "odom");
      process_period_ms_ = std::max(100, declare_parameter<int>("process_period_ms", 500));
      max_buffered_pairs_ = static_cast<std::size_t>(
          std::max(1, declare_parameter<int>("max_buffered_pairs", 2)));
      voxel_leaf_size_ = static_cast<float>(
          std::max(0.05, declare_parameter<double>("voxel_leaf_size", 0.20)));
      max_merged_points_ = std::max(10000, declare_parameter<int>("max_merged_points", 600000));
      sor_mean_k_ = std::max(5, declare_parameter<int>("sor_mean_k", 30));
      sor_stddev_ = std::max(0.1, declare_parameter<double>("sor_stddev", 1.0));

      const float mount_x = static_cast<float>(declare_parameter<double>("camera_mount_x", 0.0));
      const float mount_y = static_cast<float>(declare_parameter<double>("camera_mount_y", 0.0));
      const float mount_z = static_cast<float>(declare_parameter<double>("camera_mount_z", 0.0));
      const float mount_roll = static_cast<float>(declare_parameter<double>("camera_mount_roll", 0.0));
      const float mount_pitch = static_cast<float>(declare_parameter<double>("camera_mount_pitch", 0.0));
      const float mount_yaw = static_cast<float>(declare_parameter<double>("camera_mount_yaw", 0.0));

      camera_mount_translation_ = Eigen::Vector3f(mount_x, mount_y, mount_z);
      camera_mount_rotation_ =
          Eigen::AngleAxisf(mount_yaw, Eigen::Vector3f::UnitZ()).toRotationMatrix() *
          Eigen::AngleAxisf(mount_pitch, Eigen::Vector3f::UnitY()).toRotationMatrix() *
          Eigen::AngleAxisf(mount_roll, Eigen::Vector3f::UnitX()).toRotationMatrix();

      rclcpp::SubscriptionOptions sub_opts;
      rclcpp::CallbackGroup::SharedPtr sync_cb_group = create_callback_group(
          rclcpp::CallbackGroupType::Reentrant);
      sub_opts.callback_group = sync_cb_group;

      cloud_sub_.subscribe(this, "/input_cloud",
                           rclcpp::SensorDataQoS().get_rmw_qos_profile(), sub_opts);
      odom_sub_.subscribe(this, "/odom",
                          rclcpp::QoS(10).reliable().get_rmw_qos_profile(), sub_opts);

      sync_ = std::make_shared<Synchronizer>(
          SyncPolicy(100), cloud_sub_, odom_sub_);
      sync_->registerCallback(&PclProcNode::sync_callback, this);

      cloud_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
          "/output_cloud", rclcpp::SensorDataQoS());

      cluster_pub_ = create_publisher<pcl_cstm_msg::msg::PointCloudArray>(
          "/clusters", rclcpp::SensorDataQoS());

      cylinder_pub_ = create_publisher<pcl_cstm_msg::msg::VCylindersFit>(
          "/cylinders", rclcpp::SensorDataQoS());

      global_cylinder_pub_ = create_publisher<pcl_cstm_msg::msg::TrackedCylinderArray>(
          "/global/cylinders", rclcpp::SensorDataQoS());

      timer_ = create_wall_timer(
          std::chrono::milliseconds(process_period_ms_),
          [this]()
          { timer_callback(); });
    }

  private:
    void sync_callback(
        const sensor_msgs::msg::PointCloud2::ConstSharedPtr &cloud_msg,
        const nav_msgs::msg::Odometry::ConstSharedPtr &odom_msg)
    {
      std::lock_guard<std::mutex> lock(swap_mutex_);
      // Jangan biarkan frame menumpuk ketika backend PCL sedang sibuk.
      // Cukup simpan dua pasangan cloud-odometry terbaru.
      write_buffer_->push_back({cloud_msg, odom_msg});

      if (write_buffer_->size() > max_buffered_pairs_)
      {
        write_buffer_->erase(
            write_buffer_->begin(),
            write_buffer_->end() - max_buffered_pairs_);
      }
    }

    void timer_callback()
    {
      auto timer_cb_start = std::chrono::high_resolution_clock::now();
      std::shared_ptr<std::vector<CloudOdomPair>> to_process;
      {
        std::lock_guard<std::mutex> lock(swap_mutex_);
        std::swap(write_buffer_, process_buffer_);
        to_process = process_buffer_;
      }

      if (to_process->empty())
      {
        return;
      }

      // Reduce ke 700k jika lebih dari 1,2 juta point cloud

      int total_point_clouds = 0;
      for (const auto &pair : *to_process)
      {
        total_point_clouds += static_cast<int>(
            pair.cloud->width * pair.cloud->height);
      }

      pcl::PointCloud<pcl::PointXYZ>::Ptr merged_cloud(
          new pcl::PointCloud<pcl::PointXYZ>);

      auto time_filter_start = std::chrono::high_resolution_clock::now();

      for (const auto &pair : *to_process)
      {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*pair.cloud, *cloud);
        pcl::RandomSample<pcl::PointXYZ> random_sampler;

        if (total_point_clouds > max_merged_points_ * 2)
        {
          //RCLCPP_INFO(get_logger(), "Resampling");
          pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_sampled(new pcl::PointCloud<pcl::PointXYZ>);
          unsigned int samples_per_pair = static_cast<unsigned int>(max_merged_points_ / to_process->size());
          random_sampler.setInputCloud(cloud);
          random_sampler.setSample(samples_per_pair);

          random_sampler.filter(*cloud_sampled);
          cloud = cloud_sampled;
        }

        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::VoxelGrid<pcl::PointXYZ> voxel;
        voxel.setInputCloud(cloud);
        voxel.setLeafSize(voxel_leaf_size_, voxel_leaf_size_, voxel_leaf_size_);
        voxel.filter(*filtered);

        pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
        sor.setInputCloud(filtered);
        sor.setMeanK(sor_mean_k_);
        sor.setStddevMulThresh(sor_stddev_);
        sor.filter(*filtered);

        const auto &pos = pair.odom->pose.pose.position;
        const auto &q = pair.odom->pose.pose.orientation;
        Eigen::Quaternionf rotation(q.w, q.x, q.y, q.z);
        Eigen::Matrix3f R = rotation.toRotationMatrix();
        Eigen::Vector3f t(pos.x, pos.y, pos.z);

        for (const auto &pt : filtered->points)
        {
          Eigen::Vector3f optical_pt(pt.x, pt.y, pt.z);
          Eigen::Vector3f robot_pt = camera_mount_rotation_ * (R_optical_to_robot_ * optical_pt) + camera_mount_translation_;
          Eigen::Vector3f global_pt = R * robot_pt + t;
          pcl::PointXYZ p;
          p.x = global_pt.x();
          p.y = global_pt.y();
          p.z = global_pt.z();
          merged_cloud->push_back(p);
        }
      }

      auto time_filter_end = std::chrono::high_resolution_clock::now();
      double time_filter_ms =
          std::chrono::duration_cast<std::chrono::microseconds>(time_filter_end - time_filter_start)
              .count() /
          1000.0;
      RCLCPP_INFO(get_logger(), "Total Time Filter: %lf ms, array size: %ld", time_filter_ms, to_process->size());

      merged_cloud->width = merged_cloud->size();
      merged_cloud->height = 1;
      merged_cloud->is_dense = true;

      auto time_ransac_start = std::chrono::high_resolution_clock::now();
      auto non_ground = processRANSAC(merged_cloud);
      if (!non_ground || non_ground->empty())
      {
        RCLCPP_WARN(get_logger(), "Ground removal failed");
        to_process->clear();
        return;
      }
      auto time_ransac_end = std::chrono::high_resolution_clock::now();
      double time_ransac_ms =
          std::chrono::duration_cast<std::chrono::microseconds>(time_ransac_end - time_ransac_start)
              .count() /
          1000.0;
      RCLCPP_INFO(get_logger(), "Total Time RANSAC: %lf ms", time_ransac_ms);

      auto trunk_filter = removeNonNormals(non_ground);

      sensor_msgs::msg::PointCloud2 output_msg;
      pcl::toROSMsg(*trunk_filter, output_msg);
      output_msg.header.stamp = now();
      output_msg.header.frame_id = output_frame_;

      cloud_pub_->publish(output_msg);

      auto time_cluster_start = std::chrono::high_resolution_clock::now();
      auto clusters = clusterTrees_RegionGrowing(trunk_filter, 5, 1);
      auto time_cluster_end = std::chrono::high_resolution_clock::now();

      double time_cluster_ms =
          std::chrono::duration_cast<std::chrono::microseconds>(time_cluster_end - time_cluster_start)
              .count() /
          1000.0;
      RCLCPP_INFO(get_logger(), "Total Time Cluster: %lf ms, total_clusters: %ld", time_cluster_ms, clusters.size());

      if (!clusters.empty())
      {
        pcl_cstm_msg::msg::PointCloudArray cluster_msg;
        cluster_msg.header.stamp = now();
        cluster_msg.header.frame_id = output_frame_;

        pcl_cstm_msg::msg::VCylindersFit cyl_array_msg;
        cyl_array_msg.header.stamp = now();
        cyl_array_msg.header.frame_id = output_frame_;

        std::vector<CylinderParams> params_vec;
        params_vec.reserve(clusters.size());

        int is_valid = 0;
        int is_point = 0;
        auto time_fit_start = std::chrono::high_resolution_clock::now();
        for (const auto &cluster : clusters)
        {
          sensor_msgs::msg::PointCloud2 cluster_cloud;
          pcl::toROSMsg(*cluster, cluster_cloud);
          cluster_cloud.header.stamp = now();
          cluster_cloud.header.frame_id = output_frame_;
          cluster_msg.clouds.push_back(std::move(cluster_cloud));

          auto params = fitCylinderZAxis(cluster);
          params_vec.push_back(params);

          pcl_cstm_msg::msg::CylinderFit cyl_msg;
          cyl_msg.header.stamp = now();
          cyl_msg.header.frame_id = output_frame_;
          cyl_msg.radius = params.radius;
          cyl_msg.height = params.height;
          cyl_msg.confidence = params.confidence;

          cyl_msg.pose.position.x = params.center_x;
          cyl_msg.pose.position.y = params.center_y;
          cyl_msg.pose.position.z = params.center_z;

          Eigen::Vector3f z_axis(0.0f, 0.0f, 1.0f);
          Eigen::Vector3f cylinder_axis(params.dir_x, params.dir_y, params.dir_z);
          if (cylinder_axis.z() < 0)
          {
            cylinder_axis = -cylinder_axis;
          }

          Eigen::Quaternionf q;
          q.setFromTwoVectors(z_axis, cylinder_axis);

          cyl_msg.pose.orientation.x = q.x();
          cyl_msg.pose.orientation.y = q.y();
          cyl_msg.pose.orientation.z = q.z();
          cyl_msg.pose.orientation.w = q.w();

          cyl_msg.is_valid = params.isValid;

          if (params.isValid)
          {
            is_valid++;
          }
          if (params.clouds && !params.clouds->points.empty())
          {
            // Convert the point cloud
            is_point++;
            pcl::toROSMsg(*params.clouds, cyl_msg.clouds);

            cyl_msg.clouds.header = cyl_msg.header;
          }

          cyl_array_msg.cylinders.push_back(std::move(cyl_msg));
        }
        auto time_fit_end = std::chrono::high_resolution_clock::now();

        double time_fit_ms =
            std::chrono::duration_cast<std::chrono::microseconds>(time_fit_end - time_fit_start)
                .count() /
            1000.0;
        RCLCPP_INFO(get_logger(), "Total Fit: %lf ms, valid: %d %d", time_fit_ms, is_valid, is_point);

        cluster_pub_->publish(cluster_msg);
        if (!cyl_array_msg.cylinders.empty())
        {
          cylinder_pub_->publish(cyl_array_msg);
        }

        std::vector<TrackedCylinder> tracked;

        auto time_manager_start = std::chrono::high_resolution_clock::now();
        global_manager_.process(params_vec, tracked);
        auto time_manager_end = std::chrono::high_resolution_clock::now();

        double time_manager_ms =
            std::chrono::duration_cast<std::chrono::microseconds>(time_manager_end - time_manager_start)
                .count() /
            1000.0;
        RCLCPP_INFO(get_logger(), "Total Manager: %lf ms", time_manager_ms);

        pcl_cstm_msg::msg::TrackedCylinderArray tracked_msg;
        tracked_msg.header.stamp = now();
        tracked_msg.header.frame_id = output_frame_;

        for (const auto &t : tracked)
        {
          pcl_cstm_msg::msg::TrackedCylinder tc_msg;
          tc_msg.id = t.id;
          tc_msg.seen_count = t.seen_count;
          tc_msg.missed_count = t.missed_count;
          tc_msg.cylinder.header.stamp = now();
          tc_msg.cylinder.header.frame_id = output_frame_;
          tc_msg.cylinder.radius = t.radius;
          tc_msg.cylinder.height = t.height;
          tc_msg.cylinder.confidence = t.confidence;
          tc_msg.cylinder.pose.position.x = t.center_x;
          tc_msg.cylinder.pose.position.y = t.center_y;
          tc_msg.cylinder.pose.position.z = t.center_z;

          Eigen::Vector3f z_axis(0.0f, 0.0f, 1.0f);
          Eigen::Vector3f cylinder_axis(t.dir_x, t.dir_y, t.dir_z);
          if (cylinder_axis.z() < 0)
          {
            cylinder_axis = -cylinder_axis;
          }

          Eigen::Quaternionf q;
          q.setFromTwoVectors(z_axis, cylinder_axis);

          tc_msg.cylinder.pose.orientation.x = q.x();
          tc_msg.cylinder.pose.orientation.y = q.y();
          tc_msg.cylinder.pose.orientation.z = q.z();
          tc_msg.cylinder.pose.orientation.w = q.w();

          tc_msg.cylinder.is_valid = true;
          tracked_msg.cylinders.push_back(std::move(tc_msg));
        }

        global_cylinder_pub_->publish(tracked_msg);
      }

      to_process->clear();
      auto timer_cb_end = std::chrono::high_resolution_clock::now();

      double timer_cb_ms =
          std::chrono::duration_cast<std::chrono::microseconds>(timer_cb_end - timer_cb_start)
              .count() /
          1000.0;
      RCLCPP_INFO(get_logger(), "Total Point Processed: %d", total_point_clouds);
      RCLCPP_INFO(get_logger(), "Total Times: %lf ms", timer_cb_ms);
    }

    message_filters::Subscriber<sensor_msgs::msg::PointCloud2> cloud_sub_;
    message_filters::Subscriber<nav_msgs::msg::Odometry> odom_sub_;

    using SyncPolicy = message_filters::sync_policies::ApproximateTime<
        sensor_msgs::msg::PointCloud2, nav_msgs::msg::Odometry>;
    using Synchronizer = message_filters::Synchronizer<SyncPolicy>;
    std::shared_ptr<Synchronizer> sync_;

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_pub_;
    rclcpp::Publisher<pcl_cstm_msg::msg::PointCloudArray>::SharedPtr cluster_pub_;
    rclcpp::Publisher<pcl_cstm_msg::msg::VCylindersFit>::SharedPtr cylinder_pub_;
    rclcpp::Publisher<pcl_cstm_msg::msg::TrackedCylinderArray>::SharedPtr global_cylinder_pub_;
    rclcpp::TimerBase::SharedPtr timer_;

    std::shared_ptr<std::vector<CloudOdomPair>> write_buffer_{
        std::make_shared<std::vector<CloudOdomPair>>()};
    std::shared_ptr<std::vector<CloudOdomPair>> process_buffer_{
        std::make_shared<std::vector<CloudOdomPair>>()};
    std::mutex swap_mutex_;

    Eigen::Matrix3f R_optical_to_robot_{
        (Eigen::Matrix3f() << 0, 0, 1,
         -1, 0, 0,
         0, -1, 0)
            .finished()};
    Eigen::Matrix3f camera_mount_rotation_{Eigen::Matrix3f::Identity()};
    Eigen::Vector3f camera_mount_translation_{Eigen::Vector3f::Zero()};
    std::string output_frame_{"odom"};
    int process_period_ms_{500};
    std::size_t max_buffered_pairs_{2};
    float voxel_leaf_size_{0.20f};
    int max_merged_points_{600000};
    int sor_mean_k_{30};
    double sor_stddev_{1.0};

    GlobalCylinderManager global_manager_{2.0f};
  };

} // namespace point_cloud_test

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);

  pcl::console::setVerbosityLevel(pcl::console::L_ALWAYS);

  rclcpp::executors::MultiThreadedExecutor executor;
  auto node = std::make_shared<point_cloud_test::PclProcNode>();
  executor.add_node(node);
  executor.spin();

  rclcpp::shutdown();
  return 0;
}