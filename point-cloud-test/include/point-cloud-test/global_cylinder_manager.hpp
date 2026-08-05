#ifndef POINT_CLOUD_TEST__GLOBAL_CYLINDER_MANAGER_HPP_
#define POINT_CLOUD_TEST__GLOBAL_CYLINDER_MANAGER_HPP_

#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>
#include "point-cloud-test/pcl_processor.h"

namespace point_cloud_test
{

struct TrackedCylinder
{
  int32_t id = 0;
  
  // 3D Geometric Center
  float center_x = 0.0f;
  float center_y = 0.0f;
  float center_z = 0.0f; 

  // 3D Orientation (Direction Vector)
  float dir_x = 0.0f;
  float dir_y = 0.0f;
  float dir_z = 1.0f; 

  float radius = 0.0f;
  float height = 0.0f;
  float confidence = 0.0f;
  
  int32_t seen_count = 0;
  int32_t missed_count = 0;

  // We still use 2D XY distance for matching because tree canopies 
  // fluctuate in height, but ground positions remain stable.
  float sq_xy_dist(float ox, float oy) const
  {
    float dx = center_x - ox;
    float dy = center_y - oy;
    return dx * dx + dy * dy;
  }
};

class GlobalCylinderManager
{
public:
  GlobalCylinderManager(float max_match_dist = 4.0f)
  : max_match_dist_(max_match_dist)
  {
    tracks_.reserve(128);
  }

  void process(const std::vector<CylinderParams> & detections,
               std::vector<TrackedCylinder> & tracked_out)
  {
    if (detections.empty()) {
      // FIX: If no detections arrive, we still must age the existing tracks!
      for (auto & t : tracks_) {
        t.missed_count++;
      }
      
      // Clean up old ones
      tracks_.erase(
        std::remove_if(tracks_.begin(), tracks_.end(),
                       [](const TrackedCylinder & t) { return t.missed_count >= 15; }),
        tracks_.end());
        
      tracked_out = tracks_;
      return;
    }

    const float sq_thresh = max_match_dist_ * max_match_dist_;

    // FIX: Capture original size so we don't try to match against newly created tracks in the same frame
    size_t original_track_count = tracks_.size();
    std::vector<bool> track_matched(original_track_count, false);

    for (const auto & det : detections) {
      if (!det.isValid) {
        continue;
      }

      int best_idx = -1;
      float best_dist = sq_thresh;

      for (size_t i = 0; i < original_track_count; ++i) {
        if (track_matched[i]) {
          continue;
        }

        float d2 = tracks_[i].sq_xy_dist(det.center_x, det.center_y);
        if (d2 < best_dist) {
          best_dist = d2;
          best_idx = static_cast<int>(i);
        }
      }

      if (best_idx >= 0) {
        /* Old Trunk */
        track_matched[best_idx] = true;
        auto & t = tracks_[best_idx];

        // Smooth spatial data
        t.center_x = t.center_x * 0.7f + det.center_x * 0.3f;
        t.center_y = t.center_y * 0.7f + det.center_y * 0.3f;
        t.center_z = t.center_z * 0.7f + det.center_z * 0.3f;
        t.radius   = t.radius * 0.7f + det.radius * 0.3f;
        t.height   = t.height * 0.7f + det.height * 0.3f;
        
        // Smooth orientation vectors
        t.dir_x = t.dir_x * 0.7f + det.dir_x * 0.3f;
        t.dir_y = t.dir_y * 0.7f + det.dir_y * 0.3f;
        t.dir_z = t.dir_z * 0.7f + det.dir_z * 0.3f;

        // FIX: Re-normalize the direction vector after blending so it remains mathematically valid
        float norm = std::sqrt(t.dir_x * t.dir_x + t.dir_y * t.dir_y + t.dir_z * t.dir_z);
        if (norm > 0.0001f) {
            t.dir_x /= norm;
            t.dir_y /= norm;
            t.dir_z /= norm;
        }

        t.confidence = det.confidence;
        t.seen_count++;
        t.missed_count = 0;
      } else {
        /* New Trunk */
        TrackedCylinder new_track;
        new_track.id = next_id_++;
        new_track.center_x = det.center_x;
        new_track.center_y = det.center_y;
        new_track.center_z = det.center_z;
        new_track.dir_x = det.dir_x;
        new_track.dir_y = det.dir_y;
        new_track.dir_z = det.dir_z;
        new_track.radius = det.radius;
        new_track.height = det.height;
        new_track.confidence = det.confidence;
        new_track.seen_count = 1;
        new_track.missed_count = 0;
        
        tracks_.push_back(new_track);
      }
    }

    // Increment missed counts only for original tracks that didn't get matched
    for (size_t i = 0; i < original_track_count; ++i) {
      if (!track_matched[i]) {
        tracks_[i].missed_count++;
      }
    }

    // Erase ghosts
    tracks_.erase(
    std::remove_if(tracks_.begin(), tracks_.end(),
                   [](const TrackedCylinder & t) { return t.missed_count >= 15; }),
    tracks_.end());

    tracked_out = tracks_;
  }

private:
  std::vector<TrackedCylinder> tracks_;
  int32_t next_id_ = 1; // Good practice to start IDs at 1 rather than 0
  float max_match_dist_ = 2.0f;
};

}  // namespace point_cloud_test

#endif  // POINT_CLOUD_TEST__GLOBAL_CYLINDER_MANAGER_HPP_