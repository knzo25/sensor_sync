
#pragma once

#include "sensor_sync_optimizer/point_types.hpp"

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>

#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_eigen/tf2_eigen.hpp>

#include <pcl/point_cloud.h>
#include <pcl_ros/transforms.hpp>
#include <pcl_conversions/pcl_conversions.h>

#include <ceres/ceres.h>
#include <ceres/loss_function.h>

#include <image_transport/image_transport.hpp>
#include <opencv2/opencv.hpp>

#include <image_geometry/pinhole_camera_model.h>

#include <string>
#include <unordered_map>

class SensorSyncOptimizer final : public rclcpp::Node
{protected:

  std::unordered_map<std::string, rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr> pointcloud_subscribers_map_;
  std::unordered_map<std::string, pcl::PointCloud<PointXYZAbsStamp>::Ptr> pointcloud_map_;

  rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr image_subscriber_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_subscriber_;

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr initial_pointcloud_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr optimized_pointcloud_publisher_;

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr initial_camera_points_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr optimized_camera_points_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr initial_image_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr optimized_image_publisher_;

  std::vector<std::string> pointcloud_topics_;
  std::string common_frame_id_;
  std::string main_frame_id_;
  double max_corr_distance_;
  double line_readout_time_us_;
  double exposure_time_ms_;
  bool use_rectified_image_;
  std::string optimization_modality_;

  std::deque<sensor_msgs::msg::CompressedImage> image_deque_; 
  sensor_msgs::msg::CameraInfo camera_info_;
  image_geometry::PinholeCameraModel pinhole_camera_model_;

  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
 
public:
  explicit SensorSyncOptimizer(const rclcpp::NodeOptions & options);

  void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg, const std::string & topic);
  void imageCallback(const sensor_msgs::msg::CompressedImage::SharedPtr msg);
  void cameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg);

  void optimizeLidars();
  void optimizeCamera(const sensor_msgs::msg::CompressedImage msg);

};
