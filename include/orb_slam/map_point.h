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

#define LOCK_OBSERVATIONS \
    std::unique_lock<std::mutex> observations_lock(observations_mutex_);
#define LOCK_POS \
    std::unique_lock<std::mutex> pos_lock(pos_mutex_);

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
    const long unsigned int& id() const { return id_; }
    const size_t nObservations();
    const bool isBad() {
        LOCK_OBSERVATIONS;
        LOCK_POS;
        return bad_point_;
    }
    const bool isInLocalMapOf(const long unsigned int& id) const
        { return this->is_in_local_map_of_ == id; }
    const bool trackedInFrame(const long unsigned int& id) const
        { return this->tracked_in_frame_ == id; }
    const bool isFuseCandidateOf(const long unsigned int& id) const
        { return this->is_fuse_candidate_of_ == id; }
    const bool inLocalAdjustmentOf(const long unsigned int& id) const
        { return this->in_local_adjustment_of_ == id; }
    const cv::Mat worldPos() const {
        LOCK_POS;
        return world_pos_.clone(); // return a copy
    }
    const cv::Mat viewVector() const {
        LOCK_POS;
        return view_vector_.clone(); // return a copy
    }
    const cv::Mat bestDescriptor() const {
        LOCK_OBSERVATIONS;
        return best_descriptor_.clone();
    }
    const float maxScaleInvDist() {
        LOCK_POS;
        return 1.2f * max_scale_distance_;
    }
    const float minScaleInvDist() {
        LOCK_POS;
        return 0.8f * min_scale_distance_;
    }
    const TrackProperties& trackProperties() const { return track_properties_; }
    const KeyFramePtr refKeyFrame() {
        LOCK_OBSERVATIONS;
        return ref_key_frame_;
    }
    const int found() {
        return found_;
    }

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
    void setTrackedInFrame(const long unsigned int& id)
        { tracked_in_frame_ = id; }
    void setIsInLocalMapOf(const long unsigned int& id)
        { is_in_local_map_of_ = id; }
    void setFuseCandidateOf(const long unsigned int& id)
        { is_fuse_candidate_of_ = id; }
    void setInLocalAdjustmentOf(const long unsigned int& id)
        { this->in_local_adjustment_of_ = id; }
    void setWorldPos(const cv::Mat& world_pos) {
        LOCK_POS;
        world_pos.copyTo(world_pos_);
    }
    void increaseVisibility(const int& n = 1) {
        LOCK_OBSERVATIONS;
        visibility_ += n;
    }
    void increaseFound(const int& n = 1) {
        LOCK_OBSERVATIONS;
        found_ += n;
    }

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
     * Returns true if this map point is observed in the given key frame.
     * @param key_frame: Key frame to check the point in
     * @returns boolean
     */
    bool isObservedInKeyFrame(const KeyFramePtr& key_frame);

    /**
     * Returns the index of this point in given key frame
     * @param key_frame: Key frame to check the point in
     */
    int getIndexInKeyFrame(const KeyFramePtr& key_frame);

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

    /**
     * Replace this map point with another map point.
     * @param other: Other map point
     */
    void replace(const MapPointPtr& other);

    int predictScale(const float& dist);

    float foundRatio() {
        LOCK_OBSERVATIONS;
        return static_cast<float>(found_) / static_cast<float>(visibility_);
    }
private:
    // Properties of map point related to tracking in a frame
    TrackProperties track_properties_;
    // id of frame whose local bundle adjustment this point is in
    int in_local_adjustment_of_;
    // id of frame whose local map this map point is in
    long int is_in_local_map_of_;
    // id of the frame in which it is last seen
    long int tracked_in_frame_;
    // id of frame whose fuse candidate this map point is
    long int is_fuse_candidate_of_;

    // Map point details
    cv::Mat world_pos_; // Point position in world coordinates
    cv::Mat view_vector_; // Mean viewing direction
    // Best descriptor from all the key frames that matches closest to the rest
    cv::Mat best_descriptor_;

    // Scale invariance distances
    float min_scale_distance_;
    float max_scale_distance_;

    // Number of times it is seen
    int visibility_ = 1;
    int found_ = 1;

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

    // Poiter to replacing point if required
    MapPointPtr replaced_by;

    // Mutexes
    mutable std::mutex pos_mutex_;
    mutable std::mutex observations_mutex_;

    // Orb matcher for descriptor distances
    static geometry::ORBMatcherConstPtr orb_matcher_;
    static geometry::ORBExtractorConstPtr orb_extractor_;

    std::string name_tag_ = {"MapPoint"};
};

} // namespace orb_slam