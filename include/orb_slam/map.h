/**
 * Declares the Map class.
 */

#pragma once

#include <memory>
#include <mutex>
#include <set>

namespace orb_slam
{

class KeyFrame;
using KeyFramePtr = std::shared_ptr<KeyFrame>;
class MapPoint;
using MapPointPtr = std::shared_ptr<MapPoint>;

class Map
{
public:
    Map() {}
    ~Map() {}

private:
    // map data
    long unsigned int max_key_frame_id_;
    std::set<MapPointPtr> map_points_; // A set of all 3d points in map
    // A set of all the key frames generated from camera observations
    std::set<KeyFramePtr> key_frames_;


    // mutexes
    std::mutex map_mutex_;
};

} // namespace orb_slam