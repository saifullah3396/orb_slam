/**
 * Defines the Tracker class.
 */

#include <ros/ros.h>
#include <ros/package.h>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "orb_slam/frame.h"
#include "orb_slam/key_frame.h"
#include "orb_slam/map.h"
#include "orb_slam/map_point.h"
#include "orb_slam/tracker.h"
#include "orb_slam/mono_tracker.h"
#include "orb_slam/rgbd_tracker.h"
#include "orb_slam/initializer.h"
#include "orb_slam/geometry/camera.h"
#include "orb_slam/geometry/orb_extractor.h"
#include "orb_slam/geometry/orb_matcher.h"
#include "orb_slam/g2o/pose_optimizer.h"

namespace orb_slam
{

Tracker::Tracker(const ros::NodeHandle& nh, const int& camera_type) : nh_(nh)
{
    // initialize the camera
    ROS_DEBUG("Initializing camera...");
    if (camera_type == static_cast<int>(geometry::CameraType::MONO)) {
        camera_ =
            geometry::CameraPtr<float>(new geometry::MonoCamera<float>(nh_));
    } else if (camera_type == static_cast<int>(geometry::CameraType::RGBD)) {
        camera_ =
            geometry::CameraPtr<float>(new geometry::RGBDCamera<float>(nh_));
    }
    camera_->readParams();
    camera_->setup();
    camera_->setupCameraStream();

    ROS_DEBUG("Initializing orb features vocabulary...");
    auto pkg_path = ros::package::getPath("orb_slam");
    orb_vocabulary_ = ORBVocabularyPtr(new ORBVocabulary());
    std::string vocabulary_path;
    ROS_DEBUG_STREAM("pkg_path:" << pkg_path);
    nh_.getParam("/orb_slam/tracker/vocabulary_path", vocabulary_path);
    try {
        orb_vocabulary_->loadFromTextFile(
            "/home/sai/visual_slam/orb_slam_ws/src/orb_slam/vocabulary/orb_vocabulary.txt");
    } catch (std::exception& e) {
        ROS_FATAL_STREAM(e.what());
        ROS_FATAL_STREAM(
            "Failed to load ORB vocabulary from path:" <<
                pkg_path + "/" + vocabulary_path);
        exit(-1);
    }
    ROS_DEBUG("ORB vocabulary successfully loaded...");

    ROS_DEBUG("Initializing orb features extractor...");
    orb_extractor_ =
        geometry::ORBExtractorPtr(new geometry::ORBExtractor(nh_));

    ROS_DEBUG("Initializing orb features matcher...");
    orb_matcher_ =
        geometry::ORBMatcherPtr(new geometry::ORBMatcher(nh_));

    ROS_DEBUG("Initializing frame base variables...");
    Frame::setCamera(
        std::const_pointer_cast<const geometry::Camera<float>>(camera_));
    Frame::setORBExtractor(
        std::const_pointer_cast<const geometry::ORBExtractor>(orb_extractor_));
    Frame::setORBMatcher(
        std::const_pointer_cast<const geometry::ORBMatcher>(orb_matcher_));
    Frame::setORBVocabulary(
        std::const_pointer_cast<const ORBVocabulary>(orb_vocabulary_));
    Frame::setupGrid(nh_);

    ROS_DEBUG("Initializing the global map...");
    map_ = MapPtr(new Map());

    ROS_DEBUG("Setting orb extractors and matchers...");
    MapPoint::setORBExtractor(
        std::const_pointer_cast<const geometry::ORBExtractor>(orb_extractor_));
    MapPoint::setORBMatcher(
        std::const_pointer_cast<const geometry::ORBMatcher>(orb_matcher_));

    ROS_DEBUG("Initializing pose optimizer...");
    pose_optimizer_ = PoseOptimizerPtr(new PoseOptimizer());

    state_ = NO_IMAGES_YET;
    ROS_DEBUG("Tracker node successfully initialized...");
}

Tracker::~Tracker()
{
}

std::unique_ptr<Tracker> Tracker::createTracker(
    const ros::NodeHandle& nh, const int& camera_type) {
    if (camera_type == static_cast<int>(geometry::CameraType::MONO)) {
        return std::unique_ptr<Tracker>(new MonoTracker(nh, camera_type));
    } else if (camera_type == static_cast<int>(geometry::CameraType::RGBD)) {
        return std::unique_ptr<Tracker>(new RGBDTracker(nh, camera_type));
    }
}

void Tracker::trackFrame()
{
    // means first image is yet to be processed
    if (state_ == NO_IMAGES_YET) {
        state_ = NOT_INITIALIZED;
    }
    last_proc_state_ = state_;

    // map is frozen and cannot be accessed by other threads
    std::unique_lock<std::mutex> lock(map_->mapUpdateMutex());

    if(state_ == NOT_INITIALIZED) {
        initializeTracking();
        if(state_ != OK)
            return;
    } else {
        // initialization done, starting tracking...
        if(state_ == OK) {
            void* motion_model = NULL; // no motion model or relocalization yet
            if (!motion_model) {
                trackReferenceFrame();
            }
        }
    }
}

bool Tracker::trackReferenceFrame()
{
    // compute bag of words vector for current frame
    current_frame_->computeBow();

    // find matches between current and reference frame. If enough matches are
    // found we setup a PnP solver
    std::vector<cv::DMatch> matches;
    orb_matcher_->matchByBowFeatures( // 0.7 taken from original orb slam code
        current_frame_, ref_frame_, matches, true, 0.7);

    if (matches.size() < MIN_REQ_MATCHES)
        return false;

    const auto ref_map_points = ref_frame_->obsMapPoints();
    for (int i = 0; i < matches.size(); ++i) {
        // add matched map points from reference to current frame
        current_frame_->addMapPoint(
            ref_map_points[matches[i].trainIdx], matches[i].queryIdx);
    }

    // set initial pose of this frame to last frame. This acts as starting point
    // for pose optimization using graph
    current_frame_->setPose(last_frame_->getCamInWorldT());
    pose_optimizer_->solve(current_frame_);

    // Discard outliers
    int map_matches = 0;
    const auto& map_points = current_frame_->obsMapPoints();
    const auto& outliers = current_frame_->outliers();
    for (int i = 0; i < current_frame_->nFeaturesUndist(); i++) {
        const auto& mp = map_points[i];
        if(!mp) continue;
        if(outliers[i]) {
            current_frame_->removeMapPointAt(i);
            current_frame_->setOutlier(i, false);
            //mp->setTrackInView(false);
            //mp->setLastSeenFrame(current_frame_->id());
        } else if (mp->nObservations() > 0) {
            map_matches++;
        }
    }

    return map_matches >= 10;
}

} // namespace orb_slam