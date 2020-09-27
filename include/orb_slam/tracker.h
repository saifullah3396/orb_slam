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

class KeyFrame;
using KeyFramePtr = std::shared_ptr<KeyFrame>;

class Map;
using MapPtr = std::shared_ptr<Map>;

class MapPoint;
using MapPointPtr = std::shared_ptr<MapPoint>;

class LocalMapper;
using LocalMapperPtr = std::shared_ptr<LocalMapper>;

template <typename T>
class MotionModel;
template <typename T>
using MotionModelPtr = std::shared_ptr<MotionModel<T>>;
template <typename T>
using MotionModelConstPtr = std::shared_ptr<const MotionModel<T>>;

class PoseOptimizer_;
using PoseOptimizerPtr = std::shared_ptr<PoseOptimizer_>;

class Viewer;
using ViewerPtr = std::shared_ptr<Viewer>;

#define MIN_REQ_MATCHES_INIT 100
#define MIN_REQ_MATCHES 15
#define MIN_REQ_MATCHES_PROJ 20
#define MIN_REQ_KEY_FRAMES_RELOC 5

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

    /**
     * Initializes the tracking with first good frame
     */
    virtual void initializeTracking() = 0;

    /**
     * Resets the tracker completely.
     */
    virtual void reset();

protected:
    virtual void trackFrame();
    virtual bool trackReferenceKeyFrame();
    virtual bool trackWithMotionModel();

    /**
     * Counts key frames that are observing map points found in frame.
     * @param frame: Frame from which map points are taken from
     * @param obs_key_frame_map: A mapping from key_frame to how many map points
     *     each key frame observes.
     */
    void findObservingKeyFrames(
        const FramePtr& frame, map<KeyFramePtr, int>& obs_key_frame_map) const;

    // local mapping of frame
    bool updateLocalMap();
    void updateLocalMapKeyFrames();
    void updateLocalMapPoints();
    bool projectLocalPoints();

    // addition of key frame to local map
    bool needNewKeyFrame();
    void createNewKeyFrame();

    // relocalization
    bool relocalize();

    std::string name_tag_ = {"Tracker"};

    // latest frame
    FramePtr current_frame_;
    std::vector<cv::Mat> camera_pose_history_; // vector of poses

    // initialization
    FramePtr last_frame_;

    // local mapping
    KeyFramePtr ref_key_frame_;
    KeyFramePtr last_key_frame_;
    int n_min_frames_; // minimum frames passed before new key frame insertion
    int n_max_frames_; // maximum frames passed before new key frame insertion
    int local_map_matches_; // matches of points with local map in current frame
    std::vector<KeyFramePtr> key_frames_local_map_;
    std::vector<MapPointPtr> map_points_local_map_;
    // The depth range within which points are considered close to frame
    float close_depth_threshold_;
    LocalMapperPtr local_mapper_;

    // motion model
    MotionModelPtr<float> motion_model_;

    // after triangulation
    cv::Mat best_rot_mat, best_trans_mat;
    std::vector<cv::Point3d> inlier_points_3d;
    std::vector<size_t> inliers_idxs;
    std::vector<cv::Point2d> inlier_points, inlier_ref_points;

    // ros node handle
    ros::NodeHandle nh_;

    // pointer to camera
    geometry::CameraPtr<float> camera_;

    geometry::ORBExtractorPtr orb_extractor_; // orb features extractor
    geometry::ORBMatcherPtr orb_matcher_; // orb features matcher
    ORBVocabularyPtr orb_vocabulary_;
    MapPtr map_;

    // g2o optimization
    PoseOptimizerPtr pose_optimizer_;

    // viewer
    ViewerPtr viewer_;

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