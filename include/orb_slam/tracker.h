/**
 * Declares the Tracker class.
 */

#pragma once

#include <cv_bridge/cv_bridge.h>
#include <opencv2/core/core.hpp>
#include <memory.h>
#include <orb_slam/orb_vocabulary.h>

namespace orb_slam
{

namespace geometry {
    template <typename T>
    class Camera;
    template <typename T>
    using CameraPtr = std::shared_ptr<Camera<T>>;

    class ORBExtractor;
    using ORBExtractorPtr = std::shared_ptr<ORBExtractor>;
    class ORBMatcher;
    using ORBMatcherPtr = std::shared_ptr<ORBMatcher>;
}

class Initializer;
using InitializerPtr = std::unique_ptr<Initializer>;

class Frame;
using FramePtr = std::shared_ptr<Frame>;

class Map;
using MapPtr = std::shared_ptr<Map>;


class Tracker {
public:
    /**
     * Constructor
     * @param nh: ROS node handle
     */
    Tracker(const ros::NodeHandle& nh, const int& camera_type);

    /**
     * Destructor
     */
    ~Tracker();

    /**
     * Creates a tracker for a given camera type
     */
    static std::unique_ptr<Tracker> createTracker(
        const ros::NodeHandle& nh, const int& camera_type);

    /**
     * Updates the visual odometry tracking
     */
    virtual void update() = 0;
    virtual void initializeTracking() = 0;

protected:
    virtual void trackFrame();
    virtual bool trackReferenceFrame();

    // latest frame
    FramePtr current_frame_;
    std::vector<cv::Mat> camera_pose_history_; // vector of poses

    // initialization
    FramePtr ref_frame_;
    FramePtr last_frame_;

    // after triangulation
    cv::Mat best_rot_mat, best_trans_mat;

    // ros node handle
    ros::NodeHandle nh_;

    // pointer to camera
    geometry::CameraPtr<float> camera_;

    geometry::ORBExtractorPtr orb_extractor_; // orb features extractor
    geometry::ORBMatcherPtr orb_matcher_; // orb features matcher
    ORBVocabularyPtr orb_vocabulary_;
    MapPtr map_;


    // Tracking states same as in original orb slam package
    enum TrackingState {
        SYSTEM_NOT_READY = -1,
        NO_IMAGES_YET = 0,
        NOT_INITIALIZED = 1,
        OK = 2,
        LOST = 3
    };
    TrackingState state_;
    TrackingState last_proc_state_;
};
using TrackerPtr = std::shared_ptr<Tracker>;

} // namespace orb_slam