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

#define LOCK_MAP std::unique_lock<std::mutex> lock(map_mutex_)

class Map
{
public:
    Map() {}
    ~Map() {}

    /**
     * Getters
     */
    const size_t nMapPoints() {
        LOCK_MAP;
        return map_points_.size();
    }

    const std::vector<MapPointPtr> refMapPoints() {
        LOCK_MAP;
        return ref_map_points_;
    }

    const std::vector<KeyFramePtr> refKeyFrames() {
        LOCK_MAP;
        return ref_key_frames_;
    }

    std::mutex& mapUpdateMutex() { return map_update_mutex_; }

    /**
     * Adds a new key frame to map
     * @param key_frame: Key frame to be added
     */
    void addKeyFrame(const KeyFramePtr& key_frame);

    /**
     * Removes the given key frame from the map
     * @param key_frame: Key frame to be removed
     */
    void removeKeyFrame(const KeyFramePtr& kf);

    /**
     * Adds a new map point to map
     * @param mp: Map point to be added
     */
    void addMapPoint(const MapPointPtr& mp);

    /**
     * Removes the given map point from the map
     * @param mp: Map point to be removed
     */
    void removeMapPoint(const MapPointPtr& mp);

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