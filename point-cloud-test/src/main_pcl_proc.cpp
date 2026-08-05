#include <iostream>
#include <chrono>
#include <iomanip>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>

// Ransac Filter
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>

// PMF
#include <pcl/segmentation/progressive_morphological_filter.h>

// Clustering
#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>

// Cylinder Fitting
#include <pcl/common/common.h>
#include <pcl/filters/passthrough.h>
#include <pcl/features/normal_3d.h>

#include "point-cloud-test/pcl_processor.h"

void fitTreeTrunkCylinder(const pcl::PointCloud<pcl::PointXYZ>::Ptr &tree_cluster, int tree_id);

int main(int argc, char **argv)
{
  // Check if the user provided a PCD file as a command-line argument
  if (argc < 3)
  {
    std::cerr << "Usage: " << argv[0] << " <path_to_pcd_file.pcd> <path_to_pcd_save_file.pcd>" << std::endl;
    return -1;
  }

  pcl::PCLPointCloud2::Ptr cloud(new pcl::PCLPointCloud2());
  pcl::PCLPointCloud2::Ptr cloud_filtered(new pcl::PCLPointCloud2());

  // Load the PCD file
  std::string filename = argv[1];
  std::string filename_save = argv[2];
  std::cout << "Loading " << filename << "..." << std::endl;

  pcl::PCDReader reader;
  // Replace the path below with the path where you saved your file
  reader.read(filename, *cloud); // Remember to download the file first!

  std::cerr << "PointCloud before filtering: " << cloud->width * cloud->height
            << " data points (" << pcl::getFieldsList(*cloud) << ")." << std::endl;

  std::cout << "Loaded " << cloud->width * cloud->height
            << " data points from " << filename << std::endl;

  // Filter
  pcl::VoxelGrid<pcl::PCLPointCloud2> sor;
  sor.setInputCloud(cloud);
  sor.setLeafSize(0.05f, 0.05f, 0.05f);

  auto start_time = std::chrono::high_resolution_clock::now();

  sor.filter(*cloud_filtered);

  auto end_time = std::chrono::high_resolution_clock::now();

  // Calc processing time for filter
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
  double duration_ms = duration / 1000.0;

  std::cout << "Downsampling complete!" << std::endl;
  std::cout << "Remaining data points: " << cloud_filtered->width * cloud_filtered->height << std::endl;
  std::cout << "Time taken: " << duration_ms << " ms (" << duration << " microseconds)" << std::endl;

  std::cerr << "PointCloud after filtering: " << cloud_filtered->width * cloud_filtered->height
            << " data points (" << pcl::getFieldsList(*cloud_filtered) << ")." << std::endl;

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::fromPCLPointCloud2(*cloud_filtered, *cloud_xyz);

  pcl::PointCloud<pcl::PointXYZ>::Ptr final_tree_cloud = processRANSAC(cloud_xyz);

  if (final_tree_cloud == nullptr)
  {
    std::cout << "Process Failed to save to " << filename_save << std::endl;
    return -1;
  }

  std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> tree_cluster = clusterTrees(final_tree_cloud);

  // SAVE CLUSTERS
  std::string base_name = filename_save;
  size_t ext_pos = base_name.find_last_of(".");
  if (ext_pos != std::string::npos)
  {
    base_name = base_name.substr(0, ext_pos);
  }

  pcl::PCDWriter writer;
  // writer.write<pcl::PointXYZ>(filename_save, *final_tree_cloud, false);
  // std::cout << "Saved trees to " << filename_save << std::endl;

  int cluster_id = 0;
  for (const auto &cluster : tree_cluster)
  {
    fitTreeTrunkCylinder(cluster, cluster_id);
    
    // Construct a unique filename for this cluster
    std::string cluster_filename = base_name + "_tree_" + std::to_string(cluster_id) + ".pcd";

    // Save the isolated tree
    writer.write<pcl::PointXYZ>(cluster_filename, *cluster, false);
    std::cout << "Saved: " << cluster_filename << " (" << cluster->points.size() << " points)" << std::endl;

    cluster_id++;
  }

  // 3. (Optional) Save the master cloud containing ALL trees without the ground
  writer.write<pcl::PointXYZ>(filename_save, *final_tree_cloud, false);
  std::cout << "\nSaved combined master cloud to: " << filename_save << std::endl;

  return 0;
}

void fitTreeTrunkCylinder(const pcl::PointCloud<pcl::PointXYZ>::Ptr &tree_cluster, int tree_id)
{
  std::cout << "\n--- Fitting Cylinder to Tree " << tree_id << " ---" << std::endl;

  pcl::PointXYZ minPt, maxPt;
  pcl::getMinMax3D(*tree_cluster, minPt, maxPt);

  // Slice canopy
  pcl::PointCloud<pcl::PointXYZ>::Ptr trunk_only(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PassThrough<pcl::PointXYZ> pass;
  pass.setInputCloud(tree_cluster);

  pass.setFilterFieldName("z");

  float trunk_height = 2.5f;
  pass.setFilterLimits(minPt.z, minPt.z + trunk_height);
  pass.filter(*trunk_only);

  if (trunk_only->points.size() < 10)
  {
    std::cerr << "Tree " << tree_id << " does not have enough points in the trunk region." << std::endl;
    return;
  }

  // Normal Estimation

  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
  pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);

  ne.setSearchMethod(tree);
  ne.setInputCloud(trunk_only); // Feed the sliced trunk!
  ne.setKSearch(50);
  ne.compute(*cloud_normals);

  // RANSAC CYLINDER FIT
  pcl::SACSegmentationFromNormals<pcl::PointXYZ, pcl::Normal> seg;
  pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

  seg.setOptimizeCoefficients(true);
  seg.setModelType(pcl::SACMODEL_CYLINDER);
  seg.setMethodType(pcl::SAC_RANSAC);

  seg.setNormalDistanceWeight(0.1);
  seg.setMaxIterations(10000);
  seg.setDistanceThreshold(0.10);
  seg.setRadiusLimits(0.1, 1.5);

  seg.setInputCloud(trunk_only); // Feed the sliced trunk!
  seg.setInputNormals(cloud_normals);

  seg.segment(*inliers, *coefficients);

  if (inliers->indices.size() == 0)
  {
    std::cerr << "Failed to find a valid cylinder in Tree " << tree_id << "." << std::endl;
    return;
  }

  // Extract and Print the Data
  // Coefficients array: [Point_X, Point_Y, Point_Z, Axis_X, Axis_Y, Axis_Z, Radius]
  float center_x = coefficients->values[0];
  float center_y = coefficients->values[1];
  float center_z = minPt.z; // Override RANSAC's Z with the actual bottom of the tree
  float radius = coefficients->values[6];

  std::cout << "Successfully fitted cylinder!" << std::endl;
  std::cout << "- Trunk Radius: " << radius << " m" << std::endl;
  std::cout << "- Base Location: X: " << center_x << ", Y: " << center_y << ", Z: " << center_z << std::endl;
}