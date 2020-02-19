/**
 * Declares the MapPoint class.
 */

#pragma once

#include <atomic>
#include <map>
#include <mutex>
#include <memory>
#include <opencv2/core/core.hpp>
#include <orb_slam/geometry/orb_extractor.h>
#include <orb_slam/geometry/orb_matcher.h>

namespace orb_slam
{

#define MIN_REQ_OBSERVATIONS 2

class KeyFrame;
using KeyFramePtr = std::shared_ptr<KeyFrame>;
class Map;
using MapPtr = std::shared_ptr<Map>;

class MapPoint : public std::enable_shared_from_this<MapPoint>
{
public:
    MapPoint(
        const cv::Mat& world_pos,
        const KeyFramePtr& ref_key_frame,
        const MapPtr& map);
    ~MapPoint();

    std::map<KeyFramePtr, size_t> observations();

    /**
     * Getters
     */
    const size_t nObservations();
    /**
     * Adds an observation for this point.
     * @param key_frame: The key frame in which the point is observed
     * @param idx: The id of the feature point in frame associated with this
     *     map point.
     */
    void addObservation(const KeyFramePtr& key_frame, const size_t idx);

    /**
     * Removes a given key frame observation for this point.
     * @param key_frame: The key frame in which the point is observed
     */
    void removeObservation(const KeyFramePtr& key_frame);

private:
    // Map point details
    cv::Mat world_pos_; // Point position in world coordinates

    long unsigned int id_; // Point id
    static std::atomic_uint64_t global_id_; // Thread safe points id accumulator

    // Key frames observing the point and associated index in keyframe
    std::map<KeyFramePtr, size_t> observations_;

    // Reference key frame
    KeyFramePtr ref_key_frame_;

    // Pointer to the map
    MapPtr map_;

    // Mutexes
    std::mutex pos_mutex_;
    std::mutex observations_mutex_;
};

} // namespace orb_slam