#include "sensor_sync_optimizer/sensor_sync_optimizer.hpp"
#include "sensor_sync_optimizer/stamp_residual.hpp"

#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/registration.h>

#include <cv_bridge/cv_bridge.h> 

SensorSyncOptimizer::SensorSyncOptimizer(const rclcpp::NodeOptions & options)
: rclcpp::Node("sensor_sync_optimizer", options)
{

  pointcloud_topics_ = this->declare_parameter<std::vector<std::string>>("pointcloud_topics", std::vector<std::string>());
  for (const auto & topic : pointcloud_topics_)
  {
    pointcloud_subscribers_map_[topic] = create_subscription<sensor_msgs::msg::PointCloud2>(
      topic, rclcpp::SensorDataQoS(), 
      [this, topic](const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        this->pointCloudCallback(msg, topic);
      });
  }

  common_frame_id_ = this->declare_parameter<std::string>("common_frame_id", "base_link");
  main_frame_id_ = pointcloud_topics_[0];
  max_corr_distance_ = this->declare_parameter<double>("max_corr_distance", 0.05);
  line_readout_time_us_ = this->declare_parameter<double>("line_readout_time_us", 10.0);
  exposure_time_ms_ = this->declare_parameter<double>("exposure_time_ms", 10.0);
  use_rectified_image_ = this->declare_parameter<bool>("use_rectified_image", true);

  optimization_modality_ = this->declare_parameter<std::string>("optimization_modality", "lidar");

  if (optimization_modality_ == "lidar")
  {
    initial_pointcloud_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("initial_pointcloud", 10);
    optimized_pointcloud_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("optimized_pointcloud", 10);
  }
  else if (optimization_modality_ == "camera")
  {
    image_subscriber_ = this->create_subscription<sensor_msgs::msg::CompressedImage>(
      "image", rclcpp::SensorDataQoS(), std::bind(&SensorSyncOptimizer::imageCallback, this, std::placeholders::_1));
    camera_info_subscriber_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
      "camera_info", rclcpp::SensorDataQoS(), std::bind(&SensorSyncOptimizer::cameraInfoCallback, this, std::placeholders::_1));

    initial_camera_points_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("initial_camera_points", 10);
    optimized_camera_points_publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("optimized_camera_points", 10);
    initial_image_publisher_ = this->create_publisher<sensor_msgs::msg::Image>("initial_image", 10);
    optimized_image_publisher_ = this->create_publisher<sensor_msgs::msg::Image>("optimized_image", 10);
  }

  tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
}

void SensorSyncOptimizer::pointCloudCallback(
  const sensor_msgs::msg::PointCloud2::SharedPtr msg, const std::string & topic)
{
  RCLCPP_INFO(get_logger(), "Received point cloud message on topic %s", msg->header.frame_id.c_str());
  RCLCPP_INFO(get_logger(), "Common frame id: %s", common_frame_id_.c_str());
  
  sensor_msgs::msg::PointCloud2 cloud_transformed;

  // Attempt to lookup transforms and transform the clouds
  try {
    auto transform = tf_buffer_->lookupTransform(
      common_frame_id_, msg->header.frame_id,
      tf2::TimePointZero);

    Eigen::Matrix4f mat = tf2::transformToEigen(transform.transform).matrix().cast<float>();
    pcl_ros::transformPointCloud(mat, *msg, cloud_transformed);
  } catch (tf2::TransformException &ex) {
    RCLCPP_WARN(this->get_logger(), "Failed to transform point cloud: %s", ex.what());
    return;
  }

  // Convert ROS PointCloud2 messages to PCL point clouds
  pcl::PointCloud<PointXYZRelStamp> pcl_cloud_rel;
  pcl::fromROSMsg(cloud_transformed, pcl_cloud_rel);

  pcl::PointCloud<PointXYZAbsStamp>::Ptr pcl_cloud_abs_ptr(new pcl::PointCloud<PointXYZAbsStamp>{});
  pcl_cloud_abs_ptr->resize(pcl_cloud_rel.size());

  // Copy the point cloud data and set the timestamp
  double abs_min_stamp = std::numeric_limits<double>::max();
  double abs_max_stamp = std::numeric_limits<double>::min();
  double rel_min_stamp = std::numeric_limits<double>::max();
  double rel_max_stamp = std::numeric_limits<double>::min();

  for (size_t i = 0; i < pcl_cloud_rel.size(); ++i)
  {
    auto & point = pcl_cloud_abs_ptr->points[i];
    point.x = pcl_cloud_rel[i].x;
    point.y = pcl_cloud_rel[i].y;
    point.z = pcl_cloud_rel[i].z;
    double rel_stamp = static_cast<double>(pcl_cloud_rel[i].time_stamp);
    double abs_stamp = static_cast<double>(msg->header.stamp.sec) + 1e-9*static_cast<double>(msg->header.stamp.nanosec + pcl_cloud_rel[i].time_stamp);
    point.time_stamp = abs_stamp;

    abs_min_stamp = std::min(abs_min_stamp, abs_stamp);
    abs_max_stamp = std::max(abs_max_stamp, abs_stamp);

    rel_min_stamp = std::min(rel_min_stamp, rel_stamp);
    rel_max_stamp = std::max(rel_max_stamp, rel_stamp);
  }

  RCLCPP_INFO(get_logger(), "Topic: %s, Abs Min timestamp: %.6f, Abs Max timestamp: %.6f", topic.c_str(), abs_min_stamp, abs_max_stamp);
  RCLCPP_INFO(get_logger(), "Topic: %s, Rel Min timestamp: %.6f, Rel Max timestamp: %.6f", topic.c_str(), rel_min_stamp, rel_max_stamp);

  pointcloud_map_[topic] = pcl_cloud_abs_ptr;

  /* bool all_clouds_received = std::all_of(
    pointcloud_map_.begin(), pointcloud_map_.end(),
    [](const auto & pair) { return !pair.second->empty(); }); */
  bool all_clouds_received = pointcloud_map_.size() == pointcloud_topics_.size();

  if (all_clouds_received && optimization_modality_ == "lidar")
  {
    RCLCPP_INFO(get_logger(), "All point clouds received");
    optimizeLidars();
  }
}

void SensorSyncOptimizer::imageCallback(const sensor_msgs::msg::CompressedImage::SharedPtr msg)
{
  RCLCPP_INFO(get_logger(), "Received image message");

  if (!pinhole_camera_model_.initialized())
  {
    return;
  }

  // Check if all the point clouds have been received and the camera is newer

  bool all_clouds_received = pointcloud_map_.size() == pointcloud_topics_.size();

  double min_cloud_time = std::numeric_limits<double>::max();
  for (const auto & pair : pointcloud_map_)
  {
    min_cloud_time = std::min(min_cloud_time, pair.second->points[0].time_stamp);
  }

  if (!all_clouds_received)
  {
    return;
  }

  image_deque_.push_back(*msg);

  if (image_deque_.size() > 10)
  {
    image_deque_.pop_front();
  }

  for (const auto & image : image_deque_)
  {
    double image_time = static_cast<double>(image.header.stamp.sec) + 1e-9*static_cast<double>(image.header.stamp.nanosec);
    if (std::abs(image_time - min_cloud_time) < 1e-3 * 25) // TODO(knzo25): this is just a harcoded value for tolerance. we do not want to pair incorrect frames
    {
      optimizeCamera(image);
      return;
    }
  }
}

void SensorSyncOptimizer::cameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
{
  RCLCPP_INFO(get_logger(), "Received camera info message");

  camera_info_ = *msg;

  if (use_rectified_image_) {
    camera_info_.k[0] = camera_info_.p[0];
    camera_info_.k[2] = camera_info_.p[2];
    camera_info_.k[4] = camera_info_.p[5];
    camera_info_.k[5] = camera_info_.p[6];
    std::fill(camera_info_.d.begin(), camera_info_.d.end(), 0.0);
  }
  
  pinhole_camera_model_.fromCameraInfo(camera_info_);
}


void SensorSyncOptimizer::optimizeLidars()
{
  pcl::PointCloud<PointXYZAbsStamp> combined_cloud;

  RCLCPP_INFO(get_logger(), "Combining point clouds");
  for (const auto & pair : pointcloud_map_)
  {
    combined_cloud += *pair.second;
  }

  // Subtract the timestamp of the first point cloud to make the timestamps relative
  double first_timestamp = pointcloud_map_[main_frame_id_]->points[0].time_stamp;

  RCLCPP_INFO(get_logger(), "Substracting first timestamp");
  double min_stamp = std::numeric_limits<double>::max();
  double max_stamp = std::numeric_limits<double>::min();
  for (size_t i = 0; i < combined_cloud.size(); ++i)
  {
    combined_cloud[i].time_stamp -= first_timestamp;
    min_stamp = std::min(min_stamp, combined_cloud[i].time_stamp);
    max_stamp = std::max(max_stamp, combined_cloud[i].time_stamp);
  }

  RCLCPP_INFO(get_logger(), "First timestamp: %.3f", first_timestamp);
  RCLCPP_INFO(get_logger(), "Min timestamp: %.3f, Max timestamp: %.3f", min_stamp, max_stamp);

  // Publish the initial combined point cloud
  RCLCPP_INFO(get_logger(), "Publish initial pointcloud");
  sensor_msgs::msg::PointCloud2 initial_pointcloud_msg;
  pcl::toROSMsg(combined_cloud, initial_pointcloud_msg);
  initial_pointcloud_msg.header.frame_id = common_frame_id_;
  initial_pointcloud_msg.header.stamp = this->now();
  initial_pointcloud_publisher_->publish(initial_pointcloud_msg);

  // Perform optimization
  // ...
  ceres::Problem problem;


  RCLCPP_INFO(get_logger(), "Computing correspondences between point clouds");
  pcl::registration::CorrespondenceEstimation<pcl::PointXYZ, pcl::PointXYZ> estimator;

  std::unordered_map<std::string, pcl::PointCloud<pcl::PointXYZ>::Ptr> pointcloud_xyz_map;
  std::vector<double> offsets_placeholder(pointcloud_topics_.size(), 0.0);

  for (const auto & [topic, cloud] : pointcloud_map_) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>{});
    pcl::copyPointCloud(*cloud, *pcl_cloud_ptr);
    pointcloud_xyz_map[topic] = pcl_cloud_ptr;
  }

  for (std::size_t source_index = 0; source_index < pointcloud_topics_.size(); source_index++) {
    const auto & source_topic = pointcloud_topics_[source_index];
    const auto & source_cloud = pointcloud_xyz_map[source_topic];
    
    for (std::size_t target_index = 0; target_index < pointcloud_topics_.size(); target_index++) {
      const auto & target_topic = pointcloud_topics_[target_index];
      const auto & target_cloud = pointcloud_xyz_map[target_topic];
      
      if (source_index == target_index) {
        continue;
      }
      estimator.setInputSource(source_cloud);
      estimator.setInputTarget(target_cloud);

      pcl::Correspondences correspondences;
      estimator.determineCorrespondences(correspondences, max_corr_distance_);
      RCLCPP_INFO(get_logger(), "Found %ld correspondences between source=%s and target=%s", correspondences.size(), source_topic.c_str(), target_topic.c_str());

      for (const auto & correspondence : correspondences) {
        auto & source_point = pointcloud_map_[source_topic]->points[correspondence.index_query];
        auto & target_point = pointcloud_map_[target_topic]->points[correspondence.index_match];

        problem.AddResidualBlock(
          StampResidual::createResidual(source_point.time_stamp, target_point.time_stamp),
          nullptr,  // L2
          &offsets_placeholder[source_index],
          &offsets_placeholder[target_index]);
      }
    }
  }

  problem.SetParameterBlockConstant(&offsets_placeholder[0]);

  double initial_cost = 0.0;
  std::vector<double> residuals;
  ceres::Problem::EvaluateOptions eval_opt;
  eval_opt.num_threads = 1;
  problem.GetResidualBlocks(&eval_opt.residual_blocks);
  problem.Evaluate(eval_opt, &initial_cost, &residuals, nullptr, nullptr);

  RCLCPP_INFO(get_logger(), "Initial cost: %.6f", initial_cost);
  
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_SCHUR;  // cSpell:ignore SCHUR
  options.minimizer_progress_to_stdout = true;
  options.max_num_iterations = 500;
  options.function_tolerance = 1e-10;
  options.gradient_tolerance = 1e-14;
  options.num_threads = 8;
  options.max_num_consecutive_invalid_steps = 1000;
  options.use_inner_iterations = true;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  RCLCPP_INFO(get_logger(), "Initial cost: %.6f", summary.final_cost);
  RCLCPP_INFO_STREAM(get_logger(), "Report: " << summary.FullReport());

  for (std::size_t i = 0; i < pointcloud_topics_.size(); i++) {
    RCLCPP_INFO(get_logger(), "Offset for %s: %.6f", pointcloud_topics_[i].c_str(), offsets_placeholder[i]);
  }

  combined_cloud.clear();

  for (std::size_t i = 0; i < pointcloud_topics_.size(); i++) {
    const auto & topic = pointcloud_topics_[i];
    const auto & cloud = pointcloud_map_[topic];
    const double offset = offsets_placeholder[i];

    for (size_t j = 0; j < cloud->size(); ++j) {
      auto & point = combined_cloud.points.emplace_back();
      point.x = cloud->points[j].x;
      point.y = cloud->points[j].y;
      point.z = cloud->points[j].z;
      point.time_stamp = cloud->points[j].time_stamp + offset - first_timestamp;
      combined_cloud.push_back(point);
    }
  }
  

  RCLCPP_INFO(get_logger(), "Publish optimized pointcloud");
  // Publish the optimized combined point cloud
  sensor_msgs::msg::PointCloud2 optimized_pointcloud_msg;
  pcl::toROSMsg(combined_cloud, optimized_pointcloud_msg);
  optimized_pointcloud_msg.header.frame_id = common_frame_id_;
  optimized_pointcloud_msg.header.stamp = this->now();
  optimized_pointcloud_publisher_->publish(optimized_pointcloud_msg);
}

void SensorSyncOptimizer::optimizeCamera(const sensor_msgs::msg::CompressedImage msg) 
{
  cv::Mat raw_data(1, msg.data.size(), CV_8UC1, (void*)msg.data.data());
  cv::Mat decoded_image = cv::imdecode(raw_data, cv::IMREAD_COLOR);
  cv::Mat initial_sync_image = decoded_image.clone();
  cv::Mat optimized_sync_image = decoded_image.clone();

  double first_timestamp = pointcloud_map_[main_frame_id_]->points[0].time_stamp;
  double image_stamp = msg.header.stamp.sec + 1e-9*msg.header.stamp.nanosec;
  double raw_camera_lidar_offset = image_stamp - first_timestamp;

  RCLCPP_INFO(get_logger(), "First point timestamp: %.6f", first_timestamp);
  RCLCPP_INFO(get_logger(), "Image timestamp: %.6f", image_stamp);
  RCLCPP_INFO(get_logger(), "Raw camera-lidar offset: %.6f", raw_camera_lidar_offset);


  double initial_offset = 0.0;
  double offset_placeholder = initial_offset;
  double lidar_placeholder = 0.0;

  pcl::PointCloud<PointXYZAbsStamp> combined_cloud, camera_combined_cloud;

  std::vector<double> combined_lidar_stamps;
  std::vector<cv::Point2d> combined_projected_points;
  std::vector<cv::Point3d> combined_camera_points;

  auto camera_matrix = pinhole_camera_model_.intrinsicMatrix();
  auto distortion_coeffs = pinhole_camera_model_.distortionCoeffs();



  for (const auto & [topic, cloud] : pointcloud_map_) {
    
    pcl::PointCloud<PointXYZAbsStamp>::Ptr pcl_cloud_abs_ptr(new pcl::PointCloud<PointXYZAbsStamp>{});

    // Attempt to lookup transforms and transform the clouds
    try {
      auto transform = tf_buffer_->lookupTransform(
        msg.header.frame_id, common_frame_id_,
        tf2::TimePointZero);

      Eigen::Matrix4f mat = tf2::transformToEigen(transform.transform).matrix().cast<float>();
      pcl::transformPointCloud(*cloud, *pcl_cloud_abs_ptr, mat);

      std::vector<cv::Point3d> camera_points;
      std::vector<cv::Point2d> projected_points;
      std::vector<double> lidar_stamps;

      for (size_t i = 0; i < pcl_cloud_abs_ptr->size(); ++i) {
        auto & point = pcl_cloud_abs_ptr->points[i];
        camera_points.push_back(cv::Point3d(point.x, point.y, point.z));    
        lidar_stamps.push_back(point.time_stamp);
      }

      cv::Matx31d rvec, tvec;

      cv::projectPoints(
        camera_points, rvec, tvec, camera_matrix, distortion_coeffs, projected_points);

      for (size_t i = 0; i < projected_points.size(); ++i) {
        if (camera_points[i].z > 0 && projected_points[i].x >= 0 && projected_points[i].x < decoded_image.cols && projected_points[i].y >= 0 && projected_points[i].y < decoded_image.rows) {
          combined_projected_points.push_back(projected_points[i]);
          combined_camera_points.push_back(camera_points[i]);
          combined_lidar_stamps.push_back(lidar_stamps[i]);
        }
      }
    } catch (tf2::TransformException &ex) {
      RCLCPP_WARN(this->get_logger(), "Failed to transform point cloud: %s", ex.what());
      return;
    }
  }



  ceres::Problem problem;

  for (std::size_t i = 0; i < combined_projected_points.size(); i++) {
    const auto & image_point = combined_projected_points[i];
    const auto & point_stamp = combined_lidar_stamps[i];    
    double image_stamp = msg.header.stamp.sec + 1e-9*msg.header.stamp.nanosec + image_point.y*1e-6*line_readout_time_us_ + 0.5*1e-3*exposure_time_ms_;

    problem.AddResidualBlock(
        StampResidual::createResidual(point_stamp, image_stamp),
        nullptr,  // L2
        &lidar_placeholder,
        &offset_placeholder);
  }

  problem.SetParameterBlockConstant(&lidar_placeholder);

  double initial_cost = 0.0;
  std::vector<double> residuals;
  ceres::Problem::EvaluateOptions eval_opt;
  eval_opt.num_threads = 1;
  problem.GetResidualBlocks(&eval_opt.residual_blocks);
  problem.Evaluate(eval_opt, &initial_cost, &residuals, nullptr, nullptr);

  RCLCPP_INFO(get_logger(), "Initial cost: %.6f", initial_cost);
  
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_SCHUR;  // cSpell:ignore SCHUR
  options.minimizer_progress_to_stdout = true;
  options.max_num_iterations = 500;
  options.function_tolerance = 1e-10;
  options.gradient_tolerance = 1e-14;
  options.num_threads = 8;
  options.max_num_consecutive_invalid_steps = 1000;
  options.use_inner_iterations = true;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  RCLCPP_INFO(get_logger(), "Final cost: %.6f", summary.final_cost);
  RCLCPP_INFO_STREAM(get_logger(), "Report: " << summary.FullReport());

  RCLCPP_INFO(get_logger(), "Initial camera-lidar offset: %.6f", initial_offset);
  RCLCPP_INFO(get_logger(), "Optimized camera-lidar offset: %.6f", offset_placeholder);

  double initial_sync_error_sum = 0.0;
  double optimized_sync_error_sum = 0.0;

  double max_error = 1e-3*exposure_time_ms_;

  for (std::size_t i = 0; i < combined_projected_points.size(); i++) {
    const auto & image_point = combined_projected_points[i];
    const auto & point_stamp = combined_lidar_stamps[i];    
    
    double initial_image_stamp = msg.header.stamp.sec + 1e-9*msg.header.stamp.nanosec + initial_offset + image_point.y*1e-6*line_readout_time_us_ + 0.5*1e-3*exposure_time_ms_;
    double optimized_image_stamp = msg.header.stamp.sec + 1e-9*msg.header.stamp.nanosec + offset_placeholder + image_point.y*1e-6*line_readout_time_us_ + 0.5*1e-3*exposure_time_ms_;
    double initial_sync_error = std::abs(point_stamp - initial_image_stamp);
    double optimized_sync_error = std::abs(point_stamp - optimized_image_stamp);
    max_error = std::max(initial_sync_error, std::max(optimized_sync_error, max_error));
  }

  RCLCPP_INFO(get_logger(), "Max error: %.6f", max_error);

  auto rainbow_fn = [](double value) -> cv::Scalar {
    double h = value * 5.0 + 1.0;
    int i = floor(h);
    double f = h - i;
    
    if(i % 2 == 0)
        f = 1.0 - f;

    double n = 1 - f;

    std::uint8_t n_int = static_cast<std::uint8_t>(n * 255);

    if(i <= 1)
      return cv::Scalar(n_int, 0, 255);
    else if(i == 2)
      return cv::Scalar(0, n_int, 255);
    else if(i == 3)
      return cv::Scalar(0, 255, n_int);
    else if (i == 4)
      return cv::Scalar(n_int, 255, 0);
    else
      return cv::Scalar(255, n_int, 0);
  };

  for (std::size_t i = 0; i < combined_projected_points.size(); i++) {
    const auto & image_point = combined_projected_points[i];
    const auto & point_stamp = combined_lidar_stamps[i];    
    
    double initial_image_stamp = msg.header.stamp.sec + 1e-9*msg.header.stamp.nanosec + initial_offset + image_point.y*1e-6*line_readout_time_us_ + 0.5*1e-3*exposure_time_ms_;
    double optimized_image_stamp = msg.header.stamp.sec + 1e-9*msg.header.stamp.nanosec + offset_placeholder + image_point.y*1e-6*line_readout_time_us_ + 0.5*1e-3*exposure_time_ms_;
    double initial_sync_error = std::abs(point_stamp - initial_image_stamp);
    double optimized_sync_error = std::abs(point_stamp - optimized_image_stamp);

    initial_sync_error_sum += initial_sync_error;
    optimized_sync_error_sum += optimized_sync_error;

    cv::Scalar initial_sync_error_color = rainbow_fn(1.0 - initial_sync_error / max_error);
    cv::Scalar optimized_sync_error_color = rainbow_fn(1.0 - optimized_sync_error / max_error);

    if (i % 1000 == 0) {
      RCLCPP_INFO(get_logger(), "Optimized sync error: %.6f color.r: %.6f color.g: %.6f color.b: %.6f", optimized_sync_error, optimized_sync_error_color[0], optimized_sync_error_color[1], optimized_sync_error_color[2]);
    }

    cv::circle(initial_sync_image, image_point, 5, initial_sync_error_color, -1);
    cv::circle(optimized_sync_image, image_point, 5, optimized_sync_error_color, -1);
  }

  RCLCPP_INFO(get_logger(), "Initial sync error: %.6fs", initial_sync_error_sum / combined_projected_points.size());
  RCLCPP_INFO(get_logger(), "Optimized sync error: %.6fs", optimized_sync_error_sum / combined_projected_points.size());

  pcl::PointCloud<PointXYZAbsStamp> initial_camera_points_cloud;
  pcl::PointCloud<PointXYZAbsStamp> optimized_camera_points_cloud;
  initial_camera_points_cloud.reserve(combined_camera_points.size());
  optimized_camera_points_cloud.reserve(combined_camera_points.size());
  
  for (std::size_t i = 0; i < combined_projected_points.size(); i++) {
    const auto & camera_point = combined_camera_points[i];
    const auto & image_point = combined_projected_points[i];
    const auto & point_stamp = combined_lidar_stamps[i];
    double initial_image_stamp = msg.header.stamp.sec + 1e-9*msg.header.stamp.nanosec + initial_offset + image_point.y*1e-6*line_readout_time_us_ + 0.5*1e-3*exposure_time_ms_;
    double optimized_image_stamp = msg.header.stamp.sec + 1e-9*msg.header.stamp.nanosec + offset_placeholder + image_point.y*1e-6*line_readout_time_us_ + 0.5*1e-3*exposure_time_ms_;
    PointXYZAbsStamp point;
    point.x = camera_point.x;
    point.y = camera_point.y;
    point.z = camera_point.z;
    //point.time_stamp = point_stamp - first_timestamp;
    point.time_stamp = std::abs(point_stamp - initial_image_stamp);
    initial_camera_points_cloud.push_back(point);

    point.time_stamp = std::abs(point_stamp - optimized_image_stamp);
    optimized_camera_points_cloud.push_back(point);
  }

  sensor_msgs::msg::PointCloud2 initial_camera_points_msg;
  pcl::toROSMsg(initial_camera_points_cloud, initial_camera_points_msg);
  initial_camera_points_msg.header.frame_id = msg.header.frame_id;
  initial_camera_points_msg.header.stamp = this->now();
  initial_camera_points_publisher_->publish(initial_camera_points_msg);

  sensor_msgs::msg::PointCloud2 optimized_camera_points_msg;
  pcl::toROSMsg(optimized_camera_points_cloud, optimized_camera_points_msg);
  optimized_camera_points_msg.header.frame_id = msg.header.frame_id;
  optimized_camera_points_msg.header.stamp = this->now();
  optimized_camera_points_publisher_->publish(optimized_camera_points_msg);

  sensor_msgs::msg::Image initial_image_msg;
  sensor_msgs::msg::Image optimized_image_msg;

  cv_bridge::CvImage initial_sync_image_bridge;
  cv_bridge::CvImage optimized_sync_image_bridge;

  initial_sync_image_bridge.encoding = sensor_msgs::image_encodings::RGB8;
  initial_sync_image_bridge.image = initial_sync_image;
  initial_image_msg = *initial_sync_image_bridge.toImageMsg();
  initial_image_publisher_->publish(initial_image_msg);

  optimized_sync_image_bridge.encoding = sensor_msgs::image_encodings::RGB8;
  optimized_sync_image_bridge.image = optimized_sync_image;
  optimized_image_msg = *optimized_sync_image_bridge.toImageMsg();
  optimized_image_publisher_->publish(optimized_image_msg);
}

RCLCPP_COMPONENTS_REGISTER_NODE(SensorSyncOptimizer)