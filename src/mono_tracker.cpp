/**
 * Implements the MonoTracker class.
 */

#include <ros/ros.h>
#include <ros/package.h>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "orb_slam/frame.h"
#include "orb_slam/mono_frame.h"
#include "orb_slam/key_frame.h"
#include "orb_slam/map.h"
#include "orb_slam/map_point.h"
#include "orb_slam/mono_tracker.h"
#include "orb_slam/g2o/pose_optimizer.h"
#include "orb_slam/initializer.h"
#include "orb_slam/geometry/camera.h"
#include "orb_slam/geometry/orb_extractor.h"
#include "orb_slam/geometry/orb_matcher.h"

namespace orb_slam
{

MonoTracker::MonoTracker(
    const ros::NodeHandle& nh, const int& camera_type) :
    Tracker(nh, camera_type)
{
    nh_.getParam("/orb_slam/tracker/initializer_sigma", initializer_sigma_);
    nh_.getParam(
        "/orb_slam/tracker/initializer_iterations", initializer_iterations_);

    ROS_DEBUG("Initializing pose optimizer...");
    pose_optimizer_ = PoseOptimizerMonoPtr(new PoseOptimizerMono());
}

void MonoTracker::update()
{
    auto image = camera_->image();
    if (!image)
        return;

    if (last_image_ && last_image_->header.seq == image->header.seq) {
        // no new images
        return;
    }

    ROS_DEBUG("Updating tracking...");

    // create a frame from the image
    ROS_DEBUG("Creating frame...");
    current_frame_ =
        FramePtr(new MonoFrame(image, ros::Time::now()));

    ROS_DEBUG("Extracting features from frame...");
    // extract features from the frame
    current_frame_->extractFeatures();

    ROS_DEBUG("Tracking frame...");
    // track the frame
    trackFrame();
    camera_pose_history_.push_back(current_frame_->cameraInWorldT());

    last_image_ = image;
}

void MonoTracker::initializeTracking()
{
    if(!initializer_) { // if no initializer, set the frame as reference
        ROS_DEBUG("Initializing the SLAM system with monocular camera...");
        // if enough features are available
        if(current_frame_->nFeatures() > MIN_REQ_MATCHES_INIT)
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
        if(current_frame_->nFeatures() <= MIN_REQ_MATCHES_INIT) {// not enough key points?
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
        if(current_frame_->nMatches() < MIN_REQ_MATCHES_INIT)
        {
            ROS_DEBUG("Not enough matches between first and reference frame");

            // not enough matches, retry
            initializer_.reset();
            return;
        }

        // try to initialize the monocular slam with current frame and already
        // assigned reference frame
        ROS_DEBUG("Trying to initialize between first and reference frame...");
        if (initializer_->tryToInitialize(
            current_frame_,
            best_rot_mat,
            best_trans_mat,
            inlier_points,
            inlier_ref_points,
            inlier_points_3d,
            inliers_idxs))
        {
            ROS_DEBUG_STREAM("R from current to reference:\n" << best_rot_mat);
            ROS_DEBUG_STREAM("t from current to reference:\n" << best_trans_mat);
            //current_frame_->showMatchesWithRef("current_frame");
            //cv::waitKey(0);

            // invert the rotations and translations since opencv gives us
            // transformation of points instead of camera
            best_rot_mat = best_rot_mat.t();
            best_trans_mat = -best_rot_mat * best_trans_mat;

            // set pose to the current frame. This pose is up-to-scale since
            // we do not have any depth information yet.
            ROS_DEBUG_STREAM("Setting frame pose...");
            cv::Mat c_T_w = cv::Mat::eye(4, 4, CV_32F); // for current frame
            best_rot_mat.copyTo(c_T_w.rowRange(0,3).colRange(0,3));
            best_trans_mat.copyTo(c_T_w.rowRange(0,3).col(3));
            current_frame_->setWorldInCam(c_T_w);
            createInitialMonocularMap(
                inlier_points,
                inlier_ref_points,
                inlier_points_3d,
                inliers_idxs);
        }
    }
}

void MonoTracker::createInitialMonocularMap(
    const std::vector<cv::Point2d>& inlier_points,
    const std::vector<cv::Point2d>& inlier_ref_points,
    const std::vector<cv::Point3d>& inlier_points_3d,
    const std::vector<size_t>& inliers_idxs)
{
    ROS_DEBUG_STREAM("Creating initial monocular map...");
    // compute bag of word features for reference and current frames
    ROS_DEBUG_STREAM("Computing frame bag of words...");
    ref_frame_->computeBow();
    current_frame_->computeBow();

    // create key frames from frames
    ROS_DEBUG_STREAM("Defining key frames...");
    auto ref_key_frame =
        KeyFramePtr(
            new KeyFrame(ref_frame_, map_));

    auto key_frame =
        KeyFramePtr(
            new KeyFrame(current_frame_, map_));

    ROS_DEBUG_STREAM("Adding key frames to map...");
    // add key frames to map
    map_->addKeyFrame(ref_key_frame);
    map_->addKeyFrame(key_frame);

    const auto n = inlier_points.size();

    // create MapPoints and assign associated key frames
    ROS_DEBUG_STREAM("Creating map points and assigning to frames...");
    for(size_t i = 0; i < n; ++i) { // only add inliers in map
        cv::Mat world_pos(inlier_points_3d[i]);
        // The key_frame acts as reference for map point, not to be confused
        // with ref_key_frame which is the reference for transformation of
        // key_frame.
        ROS_DEBUG_STREAM("Creating map point...");
        ROS_DEBUG_STREAM("world_pos:" << world_pos);
        auto mp = MapPointPtr(new MapPoint(world_pos, key_frame, map_));

        ROS_DEBUG_STREAM("Adding point to frames...");
        // add the map point to both key frames
        ref_key_frame->setMapPointAt(mp, i);
        key_frame->setMapPointAt(mp, i);

        ROS_DEBUG_STREAM("Adding frames to point...");
        // add both key frames as observers to the map point
        ROS_DEBUG_STREAM("inliers_idxs[" << i << "]: " << inliers_idxs[i]);
        mp->addObservation(ref_key_frame, inliers_idxs[i]);
        mp->addObservation(key_frame, inliers_idxs[i]);

        ROS_DEBUG_STREAM("Computing map point best descriptor...");
        // compute map point parameters
        mp->computeBestDescriptor();

        ROS_DEBUG_STREAM("Computing map point normal...");
        mp->updateNormalAndScale();

        ROS_DEBUG_STREAM("Adding map point to map...");
        // add map point to map
        map_->addMapPoint(mp);
    }

    // Update Connections
    ref_key_frame->updateConnections();
    key_frame->updateConnections();
}

} // namespace orb_slam