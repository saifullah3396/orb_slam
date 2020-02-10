/**
 * Defines the Tracker class.
 */

#include <ros/ros.h>
#include <vector>
#include <opencv2/core/core.hpp>
#include "orb_slam/frame.h"
#include "orb_slam/tracker.h"
#include "orb_slam/initializer.h"
#include "orb_slam/geometry/camera.h"
#include "orb_slam/geometry/orb_extractor.h"
#include "orb_slam/geometry/orb_matcher.h"

namespace orb_slam
{

Tracker::Tracker(const ros::NodeHandle& nh) : nh_(nh)
{
    // initialize the camera
    ROS_DEBUG("Initializing camera...");
    camera_ = geometry::CameraPtr<float>(new geometry::MonoCamera<float>(nh_));
    camera_->readParams();
    camera_->setup();
    camera_->setupCameraStream();
    camera_->setTracker(TrackerPtr(this));

    ROS_DEBUG("Initializing orb features extractor...");
    orb_extractor_ =
        geometry::ORBExtractorPtr(new geometry::ORBExtractor(nh_));

    ROS_DEBUG("Initializing orb features matcher...");
    orb_matcher_ =
        geometry::ORBMatcherPtr(new geometry::ORBMatcher(nh_));

    ROS_DEBUG("Initializing frame base variables...");
    Frame::setCamera(camera_);
    Frame::setORBExtractor(orb_extractor_);
    Frame::setORBMatcher(orb_matcher_);
    Frame::setupGrid(nh_);
    ROS_DEBUG("Tracker node successfully initialized...");
}

Tracker::~Tracker()
{
}

void Tracker::update()
{
    ROS_DEBUG("Updating tracking...");

    // create a frame from the image
    ROS_DEBUG("Creating frame...");
    current_frame_ =
        FramePtr(new MonoFrame(ros::Time::now()));

    ROS_DEBUG("Extracting features from frame...");
    // extract features from the frame
    current_frame_->extractFeatures();

    ROS_DEBUG("Tracking frame...");
    // track the frame
    trackFrame();
    camera_pose_history_.push_back(current_frame_->getWorldToCamT().clone());
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

            // set all key points to prev matched points ?
            vb_prev_matched_.resize(key_points.size());
            for(size_t i = 0; i < key_points.size(); i++)
                vb_prev_matched_[i] = key_points[i].pt;

            // reset the initializer with current frame
            initializer_ =
                InitializerPtr(new Initializer(current_frame_, 1.0, 200));

            // reset the initial matches ?
            std::fill(v_ini_matches_.begin(), v_ini_matches_.end(), -1);
            return;
        } else {
            ROS_WARN("Not enough features to initialize. Resetting...");
        }
    } else { // now we have the reference frame so try to initialize the map
        ROS_DEBUG("Matching first frame with a reference frame...");
        // try to initialize with the current frame, here we will already have
        // a reference frame assigned.
        if(current_frame_->nFeatures() <= MIN_REQ_MATCHES) {// not enough key points?
            ROS_WARN(
                "Not enough features between first frame and reference frame");
            // discard the initializer so that we can retry
            initializer_.reset();

            // reset the initial matches ?
            std::fill(v_ini_matches_.begin(), v_ini_matches_.end(), -1);
            return;
        }

        // find correspondences between current frame and first reference frame
        current_frame_->match(ref_frame_);

        // check if there are enough matches
        if(current_frame_->nMatches() < MIN_REQ_MATCHES)
        {
            // not enough matches, retry
            initializer_.reset();
            return;
        }

        // try to initialize the monocular slam with current frame and already
        // assigned reference frame
        cv::Mat best_rot_mat, best_trans_mat;
        initializer_->tryToInitialize(
            current_frame_, best_rot_mat, best_trans_mat);
    }
}

} // namespace orb_slam