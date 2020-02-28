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
#include "orb_slam/motion_model.h"
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

    ROS_DEBUG("Initializing motion model...");
    motion_model_ = MotionModelPtr<float>(new MotionModel<float>());

    ROS_DEBUG("Initializing pose optimizer...");
    pose_optimizer_ = PoseOptimizerPtr(new PoseOptimizer());

    state_ = NO_IMAGES_YET;
    ROS_DEBUG("Tracker node successfully initialized...");
}

Tracker::~Tracker()
{
}

void Tracker::reset()
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
        ROS_DEBUG_STREAM("Initializing tracking...");
        initializeTracking();
        if(state_ != OK)
            return;
    } else {
        bool tracking_good = false;
        // initialization done, starting tracking...
        if(state_ == OK) {
            if (motion_model_ && motion_model_->initialized()) {
                tracking_good = trackWithMotionModel();
            } else {
                tracking_good = trackReferenceKeyFrame();
            }
        }

        // @todo: update frame drawing here...

        if (tracking_good) {

            // @todo: track local map here...

            state_ = TrackingState::OK;

            motion_model_->updateModel(
                current_frame_->getWorldInCamT(), ros::Time::now());

            // @todo: draw current camera pose here

            // @todo: check if we can insert a new key frame to local mapper

        } else {
            state_ = TrackingState::LOST;

            if(map_->nKeyFrames() <= MIN_REQ_KEY_FRAMES_RELOC) {
                ROS_WARN_STREAM(
                    "Tracking lost with very few key frames. \
                    Cannot relocalize...");
                reset();
                return;
            }
        }

        last_frame_ = current_frame_;
    }
}

bool Tracker::trackReferenceKeyFrame()
{
    ROS_DEBUG_STREAM("Computing orb bow features...");
    // compute bag of words vector for current frame
    current_frame_->computeBow();
    ROS_DEBUG_STREAM("Bow: " << current_frame_->bow().size());
    ROS_DEBUG_STREAM("Bow Features: " << current_frame_->bowFeatures().size());

    // find matches between current and reference frame.
    ROS_DEBUG_STREAM("Matching bow features between frames...");
    // 0.7 taken from original orb slam code
    current_frame_->matchByBowFeatures(ref_key_frame_->frame(), true, 0.7);
    const auto& matches = current_frame_->matches();

    ROS_DEBUG_STREAM("Matches: " << matches.size());

    if (matches.size() < MIN_REQ_MATCHES)
        return false;

    //current_frame_->showMatchesWithRef("Matched points.");
    //cv::waitKey(0);
    //ROS_DEBUG_STREAM("Adding resultant map points to map.");

    const auto ref_map_points = ref_key_frame_->frame()->obsMapPoints();
    for (int i = 0; i < matches.size(); ++i) {
        // add matched map points from reference to current frame
        current_frame_->addMapPoint(
            ref_map_points[matches[i].trainIdx], matches[i].queryIdx);
    }

    ROS_DEBUG_STREAM("Optimizing current frame pose...");
    // set initial pose of this frame to last frame. This acts as starting point
    // for pose optimization using graph
    current_frame_->setWorldInCam(ref_key_frame_->frame()->getWorldInCamT());
    cv::Mat opt_pose;
    pose_optimizer_->solve(current_frame_, opt_pose);
    current_frame_->setWorldInCam(opt_pose);

    // discard outliers
    ROS_DEBUG_STREAM("Discarding outliers points...");
    int map_matches = 0;
    const auto& map_points = current_frame_->obsMapPoints();
    const auto& outliers = current_frame_->outliers();
    ROS_DEBUG_STREAM("map points:" << map_points.size());
    ROS_DEBUG_STREAM("outliers points:" << outliers.size());
    for (int i = 0; i < current_frame_->nFeaturesUndist(); i++) {
        const auto& mp = map_points[i];
        if(!mp) continue;
        if(outliers[i]) {
            // remove the matched map point since it is an outlier
            current_frame_->removeMapPointAt(i);
            // reset the feature is inlier for usage next time with maybe
            // another reference matching
            current_frame_->setOutlier(i, false);
            //mp->setTrackInView(false); used in orb_slam
            //mp->setLastSeenFrame(current_frame_->id()); used in orb_slam
        } else if (mp->nObservations() > 0) {
            map_matches++;
        }
    }
    ROS_DEBUG_STREAM("map_matches:" << map_matches);
    return map_matches >= 10;
}

bool Tracker::trackWithMotionModel()
{
    ROS_DEBUG_STREAM("Tracking with motion model...");
    // compute predicted camera pose...
    cv::Mat predicted_pose;
    if (motion_model_->predict(predicted_pose, ros::Time::now())) {
        ROS_WARN_STREAM("Failed to predict next pose.");
    }
    current_frame_->setWorldInCam(predicted_pose);
    current_frame_->resetMap();

    // project points seen in previous frame to current frame//
    int radius; // match search radius
    if(camera_->type() == geometry::CameraType::MONO)
        radius = 15;
    else
        radius = 7;

    current_frame_->matchByProjection(last_frame_, true, radius);
    const auto& matches = current_frame_->matches();
    ROS_DEBUG_STREAM("Matches: " << matches.size());
    if (matches.size() < MIN_REQ_MATCHES_PROJ) {
        radius *= 2.0;
        current_frame_->resetMap();
        current_frame_->matchByProjection(last_frame_, true, radius);
    }

    if (matches.size() < MIN_REQ_MATCHES_PROJ) {
        return false;
    }

    const auto ref_map_points = ref_key_frame_->frame()->obsMapPoints();
    for (int i = 0; i < matches.size(); ++i) {
        // add matched map points from reference to current frame
        current_frame_->addMapPoint(
            ref_map_points[matches[i].trainIdx], matches[i].queryIdx);
    }

    ROS_DEBUG_STREAM("Optimizing current frame pose...");
    // set initial pose of this frame to last frame. This acts as starting point
    // for pose optimization using graph
    current_frame_->setWorldInCam(ref_key_frame_->frame()->getWorldInCamT());
    cv::Mat opt_pose;
    pose_optimizer_->solve(current_frame_, opt_pose);
    current_frame_->setWorldInCam(opt_pose);

    // discard outliers
    ROS_DEBUG_STREAM("Discarding outliers points...");
    int map_matches = 0;
    const auto& map_points = current_frame_->obsMapPoints();
    const auto& outliers = current_frame_->outliers();
    ROS_DEBUG_STREAM("map points:" << map_points.size());
    ROS_DEBUG_STREAM("outliers points:" << outliers.size());
    for (int i = 0; i < current_frame_->nFeaturesUndist(); i++) {
        const auto& mp = map_points[i];
        if(!mp) continue;
        if(outliers[i]) {
            // remove the matched map point since it is an outlier
            current_frame_->removeMapPointAt(i);
            // reset the feature is inlier for usage next time with maybe
            // another reference matching
            current_frame_->setOutlier(i, false);
            //mp->setTrackInView(false); used in orb_slam
            //mp->setLastSeenFrame(current_frame_->id()); used in orb_slam
        } else if (mp->nObservations() > 0) {
            map_matches++;
        }
    }
    ROS_DEBUG_STREAM("map_matches:" << map_matches);
    return map_matches >= 10;
}

} // namespace orb_slam