/**
 * Declares the Map class.
 */

#pragma once

#include <memory>
#include <mutex>
#include <set>
#include <vector>

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

    const void lock() const {
        LOCK_MAP;
    }

    /**
     * Getters
     */
    const size_t nMapPoints() const {
        LOCK_MAP;
        return map_points_.size();
    }

    const size_t nKeyFrames() const {
        LOCK_MAP;
        return key_frames_.size();
    }

    const std::set<MapPointPtr> mapPoints() const {
        LOCK_MAP;
        return map_points_;
    }

    const std::set<KeyFramePtr> keyFrames() const {
        LOCK_MAP;
        return key_frames_;
    }

    const std::vector<MapPointPtr> refMapPoints() const {
        LOCK_MAP;
        return ref_map_points_;
    }

    const std::vector<KeyFramePtr> refKeyFrames() const {
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
     * Adds a new key frame to map that acts as global reference
     * @param key_frame: Key frame to be added
     */
    void addRefKeyFrame(const KeyFramePtr& ref_key_frame);

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

    /**
     * Adds a new map point to reference map
     * @param mp: Map point to be added
     */
    void addRefMapPoint(const MapPointPtr& mp);

    /**
     * Removes the given map point from the reference map
     * @param mp: Map point to be removed
     */
    void removeRefMapPoint(const MapPointPtr& mp);

private:
    // map data
    long unsigned int max_key_frame_id_;
    std::set<MapPointPtr> map_points_; // A set of all 3d points in map
    // A set of all the key frames generated from camera observations
    std::set<KeyFramePtr> key_frames_;

    // Map points of reference key frame
    std::vector<MapPointPtr> ref_map_points_;
    std::vector<KeyFramePtr> ref_key_frames_;

    // mutexes. mutable for use in const functions
    mutable std::mutex map_mutex_; // for updating map

    // for freezing map state between different threads
    std::mutex map_update_mutex_;
};

using MapPtr = std::shared_ptr<Map>;
using MapConstPtr = std::shared_ptr<const Map>;

} // namespace orb_slam