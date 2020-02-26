/**
 * Defines the Tracker class.
 */

#include <ros/ros.h>
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

    if(state_ == NOT_INITIALIZED) {
        if (camera_->type() == geometry::CameraType::MONO) {
            monocularInitialization();
        }

        if(state_ != OK)
            return;
    }
}

void Tracker::monocularInitialization()
{
    if(!initializer_) { // if no initializer, set the frame as reference
        ROS_DEBUG("Initializing the SLAM system with monocular camera...");
        // if enough features are available
        if(current_frame_->nFeatures() > MIN_REQ_MATCHES)
        {
            // set is reference frame
            current_frame_->setupFirstFrame();
            // assign first and last frame as current frame for initialization
            ref_frame_ = last_frame_ = current_frame_;
            const auto& key_points = current_frame_->featuresUndist();

            // reset the initializer with current frame
            initializer_ =
                InitializerPtr(
                    new Initializer(
                        current_frame_,
                        camera_,
                        initializer_sigma_,
                        initializer_iterations_));
            return;
        } else {
            ROS_WARN("Not enough features to initialize. Resetting...");
        }
    } else {
        // now we have the reference frame so try to initialize the map
        // try to initialize with the current frame, here we will already have
        // a reference frame assigned.
        if(current_frame_->nFeatures() <= MIN_REQ_MATCHES) {// not enough key points?
            ROS_WARN(
                "Not enough features between in first frame \
                    after initialization");
            // discard the initializer so that we can retry
            initializer_.reset();
            return;
        }

        ROS_DEBUG("Matching first frame with the reference frame...");
        //ref_frame_->showImageWithFeatures("ref_frame");
        //cv::waitKey(0);
        //current_frame_->showImageWithFeatures("current_frame");
        //cv::waitKey(0);

        // find correspondences between current frame and first reference frame
        current_frame_->match(ref_frame_);

        // check if there are enough matches
        if(current_frame_->nMatches() < MIN_REQ_MATCHES)
        {
            ROS_DEBUG("Not enough matches between first and reference frame");

            // not enough matches, retry
            initializer_.reset();
            return;
        }

        // try to initialize the monocular slam with current frame and already
        // assigned reference frame
        cv::Mat best_rot_mat, best_trans_mat;
        ROS_DEBUG("Trying to initialize between first and reference frame...");
        initializer_->tryToInitialize(
            current_frame_, best_rot_mat, best_trans_mat);
        ROS_DEBUG_STREAM("R from current to reference:\n" << best_rot_mat);
        ROS_DEBUG_STREAM("t from current to reference:\n" << best_trans_mat);
        current_frame_->showMatchesWithRef("current_frame");
        cv::waitKey(0);
    }
}

} // namespace orb_slam