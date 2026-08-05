#ifndef POINT_CLOUD_TEST__PCL_FILTER_H_
#define POINT_CLOUD_TEST__PCL_FILTER_H_

#include "point-cloud-test/pcl_processor.h"


inline pcl::PointCloud<pcl::PointXYZ>::Ptr removeNonNormals(
  const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float max_z_components = 0.5f
) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr trunk_candidates(new pcl::PointCloud<pcl::PointXYZ>());

  if (!cloud || cloud->empty()) 
  {
    return trunk_candidates;
  }

  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());

  ne.setSearchMethod(tree);
  ne.setInputCloud(cloud);

  ne.setRadiusSearch(0.8);

  pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
  ne.compute(*normals);

  trunk_candidates->reserve(cloud->points.size());

  for (size_t i = 0; i < cloud->points.size(); ++i)
  {
    const auto& pt = cloud->points[i];
    const auto& normal = normals->points[i];

    // Check 1: Does it have enough points to represent a normal?
    // If PCL can't find enough neighbors in the 0.9m radius, it sets the normal to NaN.
    if (std::isnan(normal.normal_x) || 
        std::isnan(normal.normal_y) || 
        std::isnan(normal.normal_z))
    {
      continue; // Skip isolated points
    }

    // Check z angle threshold
    if (std::abs(normal.normal_z) < max_z_components)
    {
      trunk_candidates->points.push_back(pt);
    }
  }

  trunk_candidates->width = trunk_candidates->points.size();
  trunk_candidates->height = 1;
  trunk_candidates->is_dense = true;

  return trunk_candidates;
} 

#endif // POINT_CLOUD_TEST__PCL_PROCESSOR_H_