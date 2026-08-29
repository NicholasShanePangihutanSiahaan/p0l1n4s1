#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <builtin_interfaces/msg/time.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <tf2/exceptions.h>
#include <tf2/time.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <zed_msgs/msg/objects_stamped.hpp>

#include "pcl_cstm_msg/msg/tracked_cylinder_array.hpp"
#include "pcl_cstm_msg/msg/v_cylinders_fit.hpp"

namespace point_cloud_test
{

std::string trim_copy(const std::string & value)
{
  auto first = std::find_if_not(
    value.begin(), value.end(),
    [](unsigned char character) {return std::isspace(character);});
  auto last = std::find_if_not(
    value.rbegin(), value.rend(),
    [](unsigned char character) {return std::isspace(character);}).base();
  return first < last ? std::string(first, last) : std::string{};
}

struct ObjectsPosePair
{
  zed_msgs::msg::ObjectsStamped::ConstSharedPtr objects;
  geometry_msgs::msg::PoseStamped::ConstSharedPtr pose;
};

struct RigidTransform
{
  Eigen::Quaternionf rotation{Eigen::Quaternionf::Identity()};
  Eigen::Vector3f translation{Eigen::Vector3f::Zero()};
};

struct CylinderEstimate
{
  float center_x{0.0f};
  float center_y{0.0f};
  float center_z{0.0f};
  float dir_x{0.0f};
  float dir_y{0.0f};
  float dir_z{1.0f};
  float radius{0.0f};
  float height{0.0f};
  float confidence{0.0f};
  bool is_valid{false};
};

struct CylinderTrack
{
  std::int32_t id{0};
  CylinderEstimate estimate;
  std::int32_t seen_count{0};
  std::int32_t missed_count{0};
};

class CylinderTrackManager
{
public:
  explicit CylinderTrackManager(float max_match_distance)
  : max_match_distance_(max_match_distance)
  {
    tracks_.reserve(128);
  }

  void process(
    const std::vector<CylinderEstimate> & detections,
    std::vector<CylinderTrack> & output)
  {
    const std::size_t original_size = tracks_.size();
    std::vector<bool> matched(original_size, false);
    const float squared_threshold =
      max_match_distance_ * max_match_distance_;

    for (const auto & detection : detections) {
      if (!detection.is_valid) {
        continue;
      }

      int best_index = -1;
      float best_distance = squared_threshold;
      for (std::size_t index = 0; index < original_size; ++index) {
        if (matched[index]) {
          continue;
        }
        const float dx = tracks_[index].estimate.center_x - detection.center_x;
        const float dy = tracks_[index].estimate.center_y - detection.center_y;
        const float squared_distance = dx * dx + dy * dy;
        if (squared_distance < best_distance) {
          best_distance = squared_distance;
          best_index = static_cast<int>(index);
        }
      }

      if (best_index < 0) {
        CylinderTrack track;
        track.id = next_id_++;
        track.estimate = detection;
        track.seen_count = 1;
        tracks_.push_back(track);
        continue;
      }

      const std::size_t index = static_cast<std::size_t>(best_index);
      matched[index] = true;
      auto & track = tracks_[index];
      constexpr float old_weight = 0.7f;
      constexpr float new_weight = 0.3f;
      track.estimate.center_x =
        old_weight * track.estimate.center_x + new_weight * detection.center_x;
      track.estimate.center_y =
        old_weight * track.estimate.center_y + new_weight * detection.center_y;
      track.estimate.center_z =
        old_weight * track.estimate.center_z + new_weight * detection.center_z;
      track.estimate.radius =
        old_weight * track.estimate.radius + new_weight * detection.radius;
      track.estimate.height =
        old_weight * track.estimate.height + new_weight * detection.height;
      Eigen::Vector3f direction(
        old_weight * track.estimate.dir_x + new_weight * detection.dir_x,
        old_weight * track.estimate.dir_y + new_weight * detection.dir_y,
        old_weight * track.estimate.dir_z + new_weight * detection.dir_z);
      if (direction.norm() > 1.0e-6f) {
        direction.normalize();
        track.estimate.dir_x = direction.x();
        track.estimate.dir_y = direction.y();
        track.estimate.dir_z = direction.z();
      }
      track.estimate.confidence = detection.confidence;
      track.estimate.is_valid = true;
      ++track.seen_count;
      track.missed_count = 0;
    }

    for (std::size_t index = 0; index < original_size; ++index) {
      if (!matched[index]) {
        ++tracks_[index].missed_count;
      }
    }
    tracks_.erase(
      std::remove_if(
        tracks_.begin(), tracks_.end(),
        [](const CylinderTrack & track) {return track.missed_count >= 15;}),
      tracks_.end());
    output = tracks_;
  }

private:
  float max_match_distance_{0.8f};
  std::int32_t next_id_{1};
  std::vector<CylinderTrack> tracks_;
};

class BbPclProcNode : public rclcpp::Node
{
public:
  BbPclProcNode()
  : Node("bb_pcl_proc_node"),
    tf_buffer_(get_clock()),
    tf_listener_(tf_buffer_)
  {
    declare_parameter("use_odom_pose", false);
    declare_parameter("callback_time", 500);
    declare_parameter("object_label_target", "pohon");
    declare_parameter("min_object_confidence", 25.0);
    declare_parameter("accept_searching_tracks", false);
    declare_parameter("global_frame_id", "map");
    declare_parameter("base_frame_id", "base_link");
    declare_parameter("max_match_distance", 0.8);

    use_odom_pose_ = get_parameter("use_odom_pose").as_bool();
    object_label_target_ = get_parameter("object_label_target").as_string();
    min_object_confidence_ =
      static_cast<float>(get_parameter("min_object_confidence").as_double());
    accept_searching_tracks_ =
      get_parameter("accept_searching_tracks").as_bool();
    global_frame_id_ = get_parameter("global_frame_id").as_string();
    base_frame_id_ = get_parameter("base_frame_id").as_string();
    global_manager_ = std::make_unique<CylinderTrackManager>(
      static_cast<float>(get_parameter("max_match_distance").as_double()));

    const auto qos = rclcpp::QoS(10).reliable().get_rmw_qos_profile();
    objects_sub_.subscribe(this, "/objects", qos);

    if (use_odom_pose_) {
      odom_sub_.subscribe(this, "/odom", qos);
      odom_sync_ = std::make_shared<ObjectsOdomSynchronizer>(
        ObjectsOdomPolicy(20), objects_sub_, odom_sub_);
      odom_sync_->registerCallback(
        std::bind(
          &BbPclProcNode::objects_odom_callback, this,
          std::placeholders::_1, std::placeholders::_2));
    } else {
      pose_sub_.subscribe(this, "/pose", qos);
      pose_sync_ = std::make_shared<ObjectsPoseSynchronizer>(
        ObjectsPosePolicy(20), objects_sub_, pose_sub_);
      pose_sync_->registerCallback(
        std::bind(
          &BbPclProcNode::objects_pose_callback, this,
          std::placeholders::_1, std::placeholders::_2));
    }

    cylinder_pub_ = create_publisher<pcl_cstm_msg::msg::VCylindersFit>(
      "/cylinders", rclcpp::SensorDataQoS());
    global_cylinder_pub_ =
      create_publisher<pcl_cstm_msg::msg::TrackedCylinderArray>(
      "/global/cylinders", rclcpp::SensorDataQoS());

    const int callback_time = std::max(
      50, static_cast<int>(get_parameter("callback_time").as_int()));
    timer_ = create_wall_timer(
      std::chrono::milliseconds(callback_time),
      std::bind(&BbPclProcNode::timer_callback, this));

    RCLCPP_INFO(
      get_logger(),
      "BB perception ready: label='%s', pose_source=%s, frame=%s, base=%s",
      object_label_target_.c_str(), use_odom_pose_ ? "odom" : "ZED pose",
      global_frame_id_.c_str(), base_frame_id_.c_str());
  }

private:
  using ObjectsPosePolicy = message_filters::sync_policies::ApproximateTime<
    zed_msgs::msg::ObjectsStamped, geometry_msgs::msg::PoseStamped>;
  using ObjectsPoseSynchronizer =
    message_filters::Synchronizer<ObjectsPosePolicy>;
  using ObjectsOdomPolicy = message_filters::sync_policies::ApproximateTime<
    zed_msgs::msg::ObjectsStamped, nav_msgs::msg::Odometry>;
  using ObjectsOdomSynchronizer =
    message_filters::Synchronizer<ObjectsOdomPolicy>;

  void store_pair(
    const zed_msgs::msg::ObjectsStamped::ConstSharedPtr & objects,
    const geometry_msgs::msg::PoseStamped::ConstSharedPtr & pose)
  {
    std::lock_guard<std::mutex> lock(pair_mutex_);
    pending_pair_ = std::make_shared<ObjectsPosePair>(
      ObjectsPosePair{objects, pose});
  }

  void objects_pose_callback(
    const zed_msgs::msg::ObjectsStamped::ConstSharedPtr & objects,
    const geometry_msgs::msg::PoseStamped::ConstSharedPtr & pose)
  {
    store_pair(objects, pose);
  }

  void objects_odom_callback(
    const zed_msgs::msg::ObjectsStamped::ConstSharedPtr & objects,
    const nav_msgs::msg::Odometry::ConstSharedPtr & odom)
  {
    auto pose = std::make_shared<geometry_msgs::msg::PoseStamped>();
    pose->header = odom->header;
    pose->pose = odom->pose.pose;
    store_pair(objects, pose);
  }

  bool object_is_usable(const zed_msgs::msg::Object & object) const
  {
    // Some ZED wrapper releases preserve the whitespace after ``0: pohon``
    // in a COCO label YAML. Normalize it before the exact class match.
    if (trim_copy(object.label) != trim_copy(object_label_target_) ||
      object.confidence < min_object_confidence_)
    {
      return false;
    }
    if (object.tracking_available && object.tracking_state != 1 &&
      !(accept_searching_tracks_ && object.tracking_state == 2))
    {
      return false;
    }
    return std::all_of(
      object.position.begin(), object.position.end(),
      [](float value) {return std::isfinite(value);});
  }

  bool object_to_base_transform(
    const std::string & object_frame, RigidTransform & result)
  {
    if (object_frame.empty()) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "ObjectsStamped.header.frame_id kosong; deteksi diabaikan");
      return false;
    }
    if (object_frame == base_frame_id_) {
      result = RigidTransform{};
      return true;
    }

    try {
      const auto transform = tf_buffer_.lookupTransform(
        base_frame_id_, object_frame, tf2::TimePointZero);
      result.rotation = Eigen::Quaternionf(
        static_cast<float>(transform.transform.rotation.w),
        static_cast<float>(transform.transform.rotation.x),
        static_cast<float>(transform.transform.rotation.y),
        static_cast<float>(transform.transform.rotation.z));
      if (result.rotation.norm() < 1.0e-6f) {
        return false;
      }
      result.rotation.normalize();
      result.translation = Eigen::Vector3f(
        static_cast<float>(transform.transform.translation.x),
        static_cast<float>(transform.transform.translation.y),
        static_cast<float>(transform.transform.translation.z));
      return result.translation.allFinite();
    } catch (const tf2::TransformException & error) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "TF %s <- %s belum tersedia: %s", base_frame_id_.c_str(),
        object_frame.c_str(), error.what());
      return false;
    }
  }

  CylinderEstimate bounding_box_to_global(
    const zed_msgs::msg::Object & object,
    const geometry_msgs::msg::PoseStamped & pose,
    const RigidTransform & object_to_base) const
  {
    CylinderEstimate result;

    Eigen::Quaternionf rotation(
      static_cast<float>(pose.pose.orientation.w),
      static_cast<float>(pose.pose.orientation.x),
      static_cast<float>(pose.pose.orientation.y),
      static_cast<float>(pose.pose.orientation.z));
    if (rotation.norm() < 1.0e-6f) {
      return result;
    }
    rotation.normalize();

    const Eigen::Vector3f translation(
      static_cast<float>(pose.pose.position.x),
      static_cast<float>(pose.pose.position.y),
      static_cast<float>(pose.pose.position.z));
    const Eigen::Vector3f local_center(
      object.position[0], object.position[1], object.position[2]);
    const Eigen::Vector3f base_center =
      object_to_base.rotation * local_center + object_to_base.translation;
    const Eigen::Vector3f global_center = rotation * base_center + translation;

    const auto corner = [&object](std::size_t index) {
        return Eigen::Vector3f(
          object.bounding_box_3d.corners[index].kp[0],
          object.bounding_box_3d.corners[index].kp[1],
          object.bounding_box_3d.corners[index].kp[2]);
      };
    const float height = (corner(0) - corner(4)).norm();
    const float width = (corner(3) - corner(0)).norm();
    const float depth = (corner(1) - corner(0)).norm();
    const float radius = 0.25f * (width + depth);
    Eigen::Vector3f local_direction = corner(0) - corner(4);
    if (local_direction.norm() < 1.0e-6f) {
      return result;
    }
    local_direction.normalize();
    const Eigen::Vector3f direction =
      rotation * object_to_base.rotation * local_direction;

    if (!global_center.allFinite() || !direction.allFinite() ||
      !std::isfinite(height) || !std::isfinite(radius) ||
      height <= 0.0f || radius <= 0.0f)
    {
      return result;
    }

    result.center_x = global_center.x();
    result.center_y = global_center.y();
    result.center_z = global_center.z();
    result.dir_x = direction.x();
    result.dir_y = direction.y();
    result.dir_z = direction.z();
    result.radius = radius;
    result.height = height;
    result.confidence = object.confidence;
    result.is_valid = true;
    return result;
  }

  pcl_cstm_msg::msg::CylinderFit cylinder_message(
    const CylinderEstimate & cylinder,
    const builtin_interfaces::msg::Time & stamp) const
  {
    pcl_cstm_msg::msg::CylinderFit msg;
    msg.header.stamp = stamp;
    msg.header.frame_id = global_frame_id_;
    msg.pose.position.x = cylinder.center_x;
    msg.pose.position.y = cylinder.center_y;
    msg.pose.position.z = cylinder.center_z;
    msg.radius = cylinder.radius;
    msg.height = cylinder.height;
    msg.confidence = cylinder.confidence;
    msg.is_valid = cylinder.is_valid;

    Eigen::Vector3f direction(
      cylinder.dir_x, cylinder.dir_y, cylinder.dir_z);
    if (direction.z() < 0.0f) {
      direction = -direction;
    }
    Eigen::Quaternionf orientation;
    orientation.setFromTwoVectors(Eigen::Vector3f::UnitZ(), direction);
    msg.pose.orientation.x = orientation.x();
    msg.pose.orientation.y = orientation.y();
    msg.pose.orientation.z = orientation.z();
    msg.pose.orientation.w = orientation.w();
    return msg;
  }

  void timer_callback()
  {
    std::shared_ptr<ObjectsPosePair> pair;
    {
      std::lock_guard<std::mutex> lock(pair_mutex_);
      std::swap(pair, pending_pair_);
    }
    if (!pair) {
      return;
    }

    RigidTransform object_to_base;
    if (!object_to_base_transform(
        pair->objects->header.frame_id, object_to_base))
    {
      return;
    }

    std::vector<CylinderEstimate> detections;
    detections.reserve(pair->objects->objects.size());
    pcl_cstm_msg::msg::VCylindersFit cylinders_msg;
    cylinders_msg.header.stamp = pair->objects->header.stamp;
    cylinders_msg.header.frame_id = global_frame_id_;

    for (const auto & object : pair->objects->objects) {
      if (!object_is_usable(object)) {
        continue;
      }
      auto cylinder = bounding_box_to_global(
        object, *pair->pose, object_to_base);
      if (!cylinder.is_valid) {
        continue;
      }
      detections.push_back(cylinder);
      cylinders_msg.cylinders.push_back(
        cylinder_message(cylinder, pair->objects->header.stamp));
    }
    cylinder_pub_->publish(cylinders_msg);

    std::vector<CylinderTrack> tracked;
    global_manager_->process(detections, tracked);
    pcl_cstm_msg::msg::TrackedCylinderArray tracked_msg;
    tracked_msg.header.stamp = pair->objects->header.stamp;
    tracked_msg.header.frame_id = global_frame_id_;
    tracked_msg.cylinders.reserve(tracked.size());
    for (const auto & track : tracked) {
      pcl_cstm_msg::msg::TrackedCylinder msg;
      msg.id = track.id;
      msg.seen_count = track.seen_count;
      msg.missed_count = track.missed_count;
      msg.cylinder = cylinder_message(
        track.estimate, pair->objects->header.stamp);
      tracked_msg.cylinders.push_back(std::move(msg));
    }
    global_cylinder_pub_->publish(tracked_msg);

    RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "BB objects=%zu accepted=%zu tracks=%zu",
      pair->objects->objects.size(), detections.size(), tracked.size());
  }

  bool use_odom_pose_{false};
  bool accept_searching_tracks_{false};
  float min_object_confidence_{25.0f};
  std::string object_label_target_{"pohon"};
  std::string global_frame_id_{"map"};
  std::string base_frame_id_{"base_link"};

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  message_filters::Subscriber<zed_msgs::msg::ObjectsStamped> objects_sub_;
  message_filters::Subscriber<geometry_msgs::msg::PoseStamped> pose_sub_;
  message_filters::Subscriber<nav_msgs::msg::Odometry> odom_sub_;
  std::shared_ptr<ObjectsPoseSynchronizer> pose_sync_;
  std::shared_ptr<ObjectsOdomSynchronizer> odom_sync_;

  std::mutex pair_mutex_;
  std::shared_ptr<ObjectsPosePair> pending_pair_;
  std::unique_ptr<CylinderTrackManager> global_manager_;
  rclcpp::Publisher<pcl_cstm_msg::msg::VCylindersFit>::SharedPtr cylinder_pub_;
  rclcpp::Publisher<pcl_cstm_msg::msg::TrackedCylinderArray>::SharedPtr
    global_cylinder_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace point_cloud_test

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::executors::MultiThreadedExecutor executor;
  auto node = std::make_shared<point_cloud_test::BbPclProcNode>();
  executor.add_node(node);
  executor.spin();
  rclcpp::shutdown();
  return 0;
}
