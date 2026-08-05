#ifndef POINT_CLOUD_TEST__PCL_PROCESSOR_H_
#define POINT_CLOUD_TEST__PCL_PROCESSOR_H_

#include <iostream>
#include <chrono>

#include <Eigen/Dense>

#include <rclcpp/rclcpp.hpp>

#include <pcl/point_types.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/features/normal_3d.h>
#include <pcl/common/common.h>

#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/region_growing.h>

#include <pcl/segmentation/progressive_morphological_filter.h>

#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>

#include "pcl_cstm_msg/msg/v_normals.hpp"
#include "pcl_cstm_msg/msg/xyz.hpp"

inline pcl::PointCloud<pcl::PointXYZ>::Ptr processPMF(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
  std::cout << "Starting Progressive Morphological Filter (PMF) ground removal..."
            << std::endl;
  pcl::PointIndices::Ptr ground_inliers(new pcl::PointIndices());
  pcl::ProgressiveMorphologicalFilter<pcl::PointXYZ> pmf;

  pmf.setInputCloud(cloud);

  pmf.setMaxWindowSize(20);
  pmf.setSlope(1.0f);
  pmf.setInitialDistance(0.5f);
  pmf.setMaxDistance(2.5f);

  pmf.extract(ground_inliers->indices);

  pcl::ExtractIndices<pcl::PointXYZ> extract;
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_trees(
      new pcl::PointCloud<pcl::PointXYZ>);

  extract.setInputCloud(cloud);
  extract.setIndices(ground_inliers);
  extract.setNegative(true);
  extract.filter(*cloud_trees);

  std::cout << "Remaining tree points: "
            << cloud_trees->width * cloud_trees->height << std::endl;

  return cloud_trees;
}

inline pcl::PointCloud<pcl::PointXYZ>::Ptr processRANSAC(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{

  pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
  pcl::PointIndices::Ptr inliers(new pcl::PointIndices());

  pcl::PointXYZ minPt, maxPt;
  pcl::getMinMax3D(*cloud, minPt, maxPt);

  /* PASSTHROUGH FILTER BETWEEN z min and z max * 0,5 */
  float z_half = minPt.z + (maxPt.z - minPt.z) / 2.0f;

  pcl::PointIndices::Ptr lower_half_indices(new pcl::PointIndices());
  pcl::PassThrough<pcl::PointXYZ> pass;
  pass.setInputCloud(cloud);
  pass.setFilterFieldName("z");
  pass.setFilterLimits(minPt.z, z_half);

  pass.filter(lower_half_indices->indices);

  pcl::SACSegmentation<pcl::PointXYZ> seg;
  seg.setOptimizeCoefficients(true);
  seg.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);
  seg.setMethodType(pcl::SAC_RANSAC);
  seg.setMaxIterations(1000);

  seg.setAxis(Eigen::Vector3f::UnitZ());
  seg.setEpsAngle(0.17f);

  seg.setDistanceThreshold(0.4);
  seg.setInputCloud(cloud);

  seg.setIndices(lower_half_indices); // bataskan seg untuk indices yang terfilter

  // auto ransac_start = std::chrono::high_resolution_clock::now();
  seg.segment(*inliers, *coefficients);
  // auto ransac_end = std::chrono::high_resolution_clock::now();

  if (inliers->indices.size() == 0)
  {
    std::cerr << "Could not estimate a planar model for the given dataset."
              << std::endl;
    return nullptr;
  }

  pcl::ExtractIndices<pcl::PointXYZ> extract;
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_trees(
      new pcl::PointCloud<pcl::PointXYZ>);

  extract.setInputCloud(cloud);
  extract.setIndices(inliers);
  extract.setNegative(true);
  extract.filter(*cloud_trees);

  // double ransac_ms =
  //   std::chrono::duration_cast<std::chrono::microseconds>(ransac_end - ransac_start)
  //     .count() / 1000.0;
  // std::cout << "Ground removed in " << ransac_ms << " ms" << std::endl;
  // std::cout << "Remaining tree points: "
  //           << cloud_trees->width * cloud_trees->height << std::endl;

  return cloud_trees;
}

inline std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clusterTrees(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_trees,
    double max_z_mean = 0.4,
    double max_z_variance = 0.1)
{
  // std::cout << "Starting Euclidean Clustering to isolate trees..." << std::endl;

  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree_search(
      new pcl::search::KdTree<pcl::PointXYZ>);
  tree_search->setInputCloud(cloud_trees);

  std::vector<pcl::PointIndices> cluster_indices;
  pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;

  ec.setClusterTolerance(0.5);

  ec.setMinClusterSize(40);
  ec.setMaxClusterSize(2000);

  ec.setSearchMethod(tree_search);
  ec.setInputCloud(cloud_trees);

  ec.extract(cluster_indices);

  std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> isolated_trees;

  size_t i = 0;
  for (const auto &indices : cluster_indices)
  {
    i++;
    pcl::PointCloud<pcl::PointXYZ>::Ptr single_tree_cloud(
        new pcl::PointCloud<pcl::PointXYZ>);
    for (const auto &idx : indices.indices)
    {
      single_tree_cloud->push_back((*cloud_trees)[idx]);
    }

    single_tree_cloud->width = single_tree_cloud->size();
    single_tree_cloud->height = 1;
    single_tree_cloud->is_dense = true;

    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr cluster_tree(new pcl::search::KdTree<pcl::PointXYZ>());
    ne.setSearchMethod(cluster_tree);
    ne.setInputCloud(single_tree_cloud);

    ne.setRadiusSearch(0.7);

    pcl::PointCloud<pcl::Normal>::Ptr cluster_normals(new pcl::PointCloud<pcl::Normal>);
    ne.compute(*cluster_normals);

    double sum_abs_z = 0.0;
    double sum_abs_z_sq = 0.0;
    int valid_normals = 0;

    for (const auto &normal : cluster_normals->points)
    {
      if (std::isnan(normal.normal_x) ||
          std::isnan(normal.normal_y) ||
          std::isnan(normal.normal_z))
      {
        continue;
      }

      // Use absolute Z to handle normal flips (pointing inward vs outward)
      double nz = std::abs(normal.normal_z);

      sum_abs_z += nz;
      sum_abs_z_sq += (nz * nz);
      valid_normals++;
    }

    if (valid_normals < 10)
    {
      continue;
    }

    double mean_z = sum_abs_z / valid_normals;
    double variance_z = (sum_abs_z_sq / valid_normals) - (mean_z * mean_z);

    if (mean_z > max_z_mean || variance_z > max_z_variance)
    {
      // std::cout << "Discarded cluster - Mean Z: " << mean_z << ", Variance Z: " << variance_z << "\n";
      // RCLCPP_INFO(rclcpp::get_logger("tree_cluster"), "Discarded cluster - Mean Z: %lf, Variance Z: %lf, index: %ld", mean_z, variance_z, (int)i - 1);

      continue; // Ignore this cluster
    }
    isolated_trees.push_back(single_tree_cloud);
  }

  return isolated_trees;
}

inline std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clusterTrees_RegionGrowing(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_trees,
    float maxDegrees = 3.0,
    float curvatureThreshold = 1.0
  )
{
  pcl::search::Search<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);

  pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator;
  normal_estimator.setSearchMethod(tree);
  normal_estimator.setInputCloud(cloud_trees);
  normal_estimator.setRadiusSearch(1.0); ;
  normal_estimator.compute(*normals);

  pcl::IndicesPtr indices(new std::vector<int>);
  pcl::removeNaNFromPointCloud(*cloud_trees, *indices);

  pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> reg;
  reg.setMinClusterSize(40);   // Retained your original min size
  reg.setMaxClusterSize(2000); // Retained your original max size
  reg.setSearchMethod(tree);
  reg.setNumberOfNeighbours(30);
  reg.setInputCloud(cloud_trees);
  reg.setIndices(indices);
  reg.setInputNormals(normals);
  reg.setSmoothnessThreshold(maxDegrees / 180.0 * M_PI);
  reg.setCurvatureThreshold(curvatureThreshold);

  std::vector<pcl::PointIndices> cluster_indices;
  reg.extract(cluster_indices);

  // 5. Extract Individual Clusters
  std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> isolated_trees;

  for (const auto &indices_it : cluster_indices)
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr single_tree_cloud(
        new pcl::PointCloud<pcl::PointXYZ>);
        
    for (const auto &idx : indices_it.indices)
    {
      single_tree_cloud->push_back((*cloud_trees)[idx]);
    }

    single_tree_cloud->width = single_tree_cloud->size();
    single_tree_cloud->height = 1;
    single_tree_cloud->is_dense = true;

    isolated_trees.push_back(single_tree_cloud);
  }

  return isolated_trees;
}

struct CylinderParams
{
  float center_x = 0.0f;
  float center_y = 0.0f;
  float center_z = 0.0f;

  float dir_x = 0.0f;
  float dir_y = 0.0f;
  float dir_z = 0.0f;
  
  float radius = 0.0f;
  float height = 0.0f;
  float confidence = 0.0f;
  pcl::PointCloud<pcl::PointXYZ>::Ptr clouds;
  bool isValid = false;
};

inline CylinderParams fitCylinderZAxis(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr &cluster)
{
  CylinderParams result;

  if (!cluster || cluster->points.size() < 10)
  {
    return result;
  }

  pcl::PointXYZ minPt, maxPt;
  pcl::getMinMax3D(*cluster, minPt, maxPt);
  
  if (minPt.z >= 0.5f)
  {
    // Poin terendah tidak boleh melayang
    return result; 
  }

  // 1. Estimate Normals for the entire cluster
  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(
      new pcl::search::KdTree<pcl::PointXYZ>());
  pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);

  ne.setSearchMethod(tree);
  ne.setInputCloud(cluster);
  ne.setRadiusSearch(1.0); 
  ne.compute(*normals);

  // 2. Setup RANSAC for a free cylinder
  pcl::SACSegmentationFromNormals<pcl::PointXYZ, pcl::Normal> seg;
  pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

  seg.setOptimizeCoefficients(true);
  seg.setModelType(pcl::SACMODEL_CYLINDER);
  seg.setMethodType(pcl::SAC_RANSAC);
  seg.setNormalDistanceWeight (0.1);
  seg.setMaxIterations(5000);
  seg.setDistanceThreshold(0.5);  
  seg.setRadiusLimits(0.05, 0.4); 

  seg.setAxis(Eigen::Vector3f(0.0, 0.0, 1.0));
  double toleransi_derajat = 30.0;
  double toleransi_radian = toleransi_derajat * (M_PI / 180.0);
  seg.setEpsAngle(toleransi_radian); 


  seg.setInputCloud(cluster);
  seg.setInputNormals(normals);
  seg.segment(*inliers, *coefficients);

  if (inliers->indices.empty())
  {
    return result;
  }

  // 3. Extract Mathematical Coefficients
  Eigen::Vector3f axis_pt(coefficients->values[0], coefficients->values[1], coefficients->values[2]);
  Eigen::Vector3f axis_dir(coefficients->values[3], coefficients->values[4], coefficients->values[5]);
  axis_dir.normalize();

  result.radius = coefficients->values[6];
  result.dir_x = 0.0;
  result.dir_y = 0.0;
  result.dir_z = 1.0;

  // 4. Calculate Length and True Center
  result.clouds.reset(new pcl::PointCloud<pcl::PointXYZ>(*cluster));

  float min_proj = std::numeric_limits<float>::max();
  float max_proj = -std::numeric_limits<float>::max();

  // Use ONLY the inliers to determine the physical bounds (height and center) of the cylinder shape
  for (const auto &idx : inliers->indices)
  {
    const auto &pt = (*cluster)[idx]; // Read directly from the input cluster

    // Project point onto the cylinder axis to find its bounds
    Eigen::Vector3f p(pt.x, pt.y, pt.z);
    float proj = (p - axis_pt).dot(axis_dir);
    
    if (proj < min_proj) min_proj = proj;
    if (proj > max_proj) max_proj = proj;
  }

  result.height = max_proj - min_proj;

  // Shift the center point to the actual physical midpoint of the inlier cloud along the axis
  float mid_proj = (max_proj + min_proj) / 2.0f;
  Eigen::Vector3f true_center = axis_pt + mid_proj * axis_dir;
  
  result.center_x = true_center.x();
  result.center_y = true_center.y();
  result.center_z = true_center.z();

  result.confidence = static_cast<float>(inliers->indices.size()) /
                      static_cast<float>(cluster->points.size());

  result.isValid = true;
  return result;
}

inline CylinderParams fitCylinder_RegionGrowing(
  const pcl::PointCloud<pcl::PointXYZ>::Ptr &cluster
) {
  CylinderParams result;

  if (!cluster || cluster->points.size() < 10)
  {
    return result;
  }

  pcl::PointXYZ minPt, maxPt;
  pcl::getMinMax3D(*cluster, minPt, maxPt);
  
  if (minPt.z >= 0.5f)
  {
    // Poin terendah tidak boleh melayang
    return result; 
  }
}

inline std::shared_ptr<pcl_cstm_msg::msg::VNormals> get_normals(
    const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> &clusters,
    int k_neighbors = 10)
{
  auto normals_msg = std::make_shared<pcl_cstm_msg::msg::VNormals>();

  for (const auto &cluster_cloud : clusters)
  {
    if (!cluster_cloud || cluster_cloud->points.size() < static_cast<size_t>(k_neighbors))
    {
      continue;
    }

    // 2. Setup normal estimation
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());

    ne.setSearchMethod(tree);
    ne.setInputCloud(cluster_cloud);
    ne.setKSearch(k_neighbors);

    // 3. Compute normals
    pcl::PointCloud<pcl::Normal>::Ptr cluster_normals(new pcl::PointCloud<pcl::Normal>);
    ne.compute(*cluster_normals);

    // 4. Push into normals_msg
    for (size_t i = 0; i < cluster_normals->points.size(); ++i)
    {
      const auto &pcl_normal = cluster_normals->points[i];
      const auto &pcl_point = cluster_cloud->points[i];

      if (std::isnan(pcl_normal.normal_x) ||
          std::isnan(pcl_normal.normal_y) ||
          std::isnan(pcl_normal.normal_z))
      {
        continue;
      }
      pcl_cstm_msg::msg::XYZ normal;
      normal.x = static_cast<double>(pcl_normal.normal_x);
      normal.y = static_cast<double>(pcl_normal.normal_y);
      normal.z = static_cast<double>(pcl_normal.normal_z);

      normals_msg->normals.push_back(normal);

      // Populate origin point
      pcl_cstm_msg::msg::XYZ origin;
      origin.x = static_cast<double>(pcl_point.x);
      origin.y = static_cast<double>(pcl_point.y);
      origin.z = static_cast<double>(pcl_point.z);

      normals_msg->origins.push_back(origin);
    }
  }

  return normals_msg;
}

#endif // POINT_CLOUD_TEST__PCL_PROCESSOR_H_
