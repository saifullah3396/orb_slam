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

struct TrackProperties {
    void reset() {
        in_view_ = false;
    }

    bool in_view_; // whether this point is in view or not
    cv::Point2f proj_xy_; // projection x-y in frame
    int proj_xr_; // projection x in right camera for stereo
    // predicted scale level in which this point is tracked
    int pred_scale_level_;
    float view_cosine_; // cosine of angle
};

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
    const bool& isBad() { return bad_point_; }
    const cv::Mat& worldPos() const { return world_pos_; }
    const cv::Mat& viewVector() const { return view_vector_; }
    const cv::Mat& bestDescriptor() const { return best_descriptor_; }

    /**
     * Setters
     */
    static void setORBMatcher(const geometry::ORBMatcherConstPtr& orb_matcher)
        { orb_matcher_ = orb_matcher; }
    static void setORBExtractor(const geometry::ORBExtractorConstPtr& orb_extractor)
        { orb_extractor_ = orb_extractor; }
    void resetTrackProperties() { track_properties_.reset(); }
    void setTrackProperties(const TrackProperties& track_properties)
        { track_properties_ = track_properties; }

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

    /**
     * Removes this point from the map
     */
    void removeFromMap();

    /**
     * Computes descriptor distances between each key frame observation of this
     * point and finds the best descriptor for this point.
     *
     * Based on original orb-slam repository
     */
    void computeBestDescriptor();

    /**
     * Computes the normal view vector from the frame to point and scale
     * invariance min/max distances of the point.
     *
     * Based on original orb-slam repository
     */
    void updateNormalAndScale();
private:
    // Properties of map point related to tracking in a frame
    TrackProperties track_properties_;
    // Map point details
    cv::Mat world_pos_; // Point position in world coordinates
    cv::Mat view_vector_; // Mean viewing direction
    // Best descriptor from all the key frames that matches closest to the rest
    cv::Mat best_descriptor_;

    // Scale invariance distances
    float min_scale_distance_;
    float max_scale_distance_;

    long unsigned int id_; // Point id
    static std::atomic_uint64_t global_id_; // Thread safe points id accumulator
    // If true, the point is bad and should be removed from the map
    bool bad_point_ = {false};

    // Key frames observing the point and associated index in keyframe
    std::map<KeyFramePtr, size_t> observations_;

    // Reference key frame
    KeyFramePtr ref_key_frame_;

    // Pointer to the map
    MapPtr map_;

    // Mutexes
    std::mutex pos_mutex_;
    std::mutex observations_mutex_;

    // Orb matcher for descriptor distances
    static geometry::ORBMatcherConstPtr orb_matcher_;
    static geometry::ORBExtractorConstPtr orb_extractor_;
};

} // namespace orb_slam