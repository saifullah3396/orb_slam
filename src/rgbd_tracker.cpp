/**
 * Implements the RGBDTracker class.
 */

#include <ros/ros.h>
#include <ros/package.h>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "orb_slam/frame.h"
#include "orb_slam/rgbd_frame.h"
#include "orb_slam/key_frame.h"
#include "orb_slam/map.h"
#include "orb_slam/map_point.h"
#include "orb_slam/rgbd_tracker.h"
#include "orb_slam/initializer.h"
#include "orb_slam/motion_model.h"
#include "orb_slam/g2o/pose_optimizer.h"
#include "orb_slam/geometry/camera.h"
#include "orb_slam/geometry/orb_extractor.h"
#include "orb_slam/geometry/orb_matcher.h"
#include "orb_slam/viewer/viewer.h"

namespace orb_slam
{

#define MIN_REQ_FEATURES 500

RGBDTracker::RGBDTracker(
    const ros::NodeHandle& nh, const int& camera_type) :
    Tracker(nh, camera_type)
{
    ROS_DEBUG("Initializing pose optimizer...");
    pose_optimizer_ = PoseOptimizerRGBDPtr(new PoseOptimizerRGBD());
}

void RGBDTracker::update()
{
    ROS_DEBUG_STREAM("Getting image...");
    auto image = camera_->image();
    auto depth = camera_->imageDepth();

    if (!image) {
        ROS_DEBUG_STREAM("No image received");
        return;
    }

    if (last_image_ && last_image_->header.seq == image->header.seq) {
        // no new images
        ROS_DEBUG_STREAM("No new image received");
        return;
    }

    // create a frame from the image
    ROS_DEBUG("Creating frame...");
    current_frame_ =
        FramePtr(new RGBDFrame(image, depth, image->header.stamp));

    ROS_DEBUG("Extracting features from frame...");
    // extract features from the frame
    current_frame_->extractFeatures();

    ROS_DEBUG("Tracking frame...");

    // track the frame
    trackFrame();

    ROS_DEBUG_STREAM("Outside track frame...");

    last_image_ = image;
    last_depth_ = depth;
}

void RGBDTracker::initializeTracking()
{
    ROS_DEBUG_STREAM("Initializing tracking...");
    const auto& n_key_points = current_frame_->nFeaturesUndist();
    if(n_key_points > MIN_REQ_FEATURES) {
        ROS_DEBUG_STREAM("Number of features found: " << n_key_points);

        // find bow for ref frame
        current_frame_->computeBow();

        // set frame pose to the origin
        current_frame_->setWorldInCam(cv::Mat::eye(4, 4, CV_32F));

        // create KeyFrame
        ref_key_frame_ =
            KeyFramePtr(
                new KeyFrame(current_frame_, map_));

        // resize local key frame map
        ref_key_frame_->resizeMap(n_key_points);

        // insert KeyFrame in the global map
        map_->addKeyFrame(ref_key_frame_);
        map_->addRefKeyFrame(ref_key_frame_);

        // create MapPoints and asscoiate to KeyFrame
        const auto& key_points = current_frame_->featuresUndist();
        const auto& key_point_depths = current_frame_->featureDepthsUndist();

        for(int i = 0; i < n_key_points; i++) {
            const auto& depth = key_point_depths[i];
            if (depth < 0) continue;
            // we consider synced and rectified images only. So undistorted
            // depths correspond to undistorted rgb in pixels...
            const auto& kp = key_points[i];

            // transform key point to 3d world space
            auto world_pos =
                cv::Mat(current_frame_->frameToWorld<float, float>(kp.pt, depth));

            // create a map point given 2d-3d correspondence
            auto mp =
                MapPointPtr(
                    new MapPoint(world_pos, ref_key_frame_, map_));

            // add the map point in local observation map of key frame
            // i corresponds to index of the map point here
            ref_key_frame_->setMapPointAt(mp, i);

            // add the key frame to map point in which it is observed
            // i corresponds to index of the key point in frame class
            mp->addObservation(ref_key_frame_, i);

            // compute map point best descriptor out of all observing key frames
            mp->computeBestDescriptor();

            // compute normal vector, and scale distance for map point
            mp->updateNormalAndScale();

            // add map points to map and to reference map
            map_->addMapPoint(mp);
            map_->addRefMapPoint(mp);
        }
        ROS_DEBUG_STREAM(
            "New map created with " << map_->nMapPoints() << " points.");

        ROS_DEBUG_STREAM("Updating motion model...");
        // initialize the motion model
        auto current_pose = current_frame_->worldInCameraT();
        motion_model_->updateModel(current_pose, current_frame_->timeStamp());

        // add the frame to viewer
        viewer_->addFrame(current_frame_);
        //mpLocalMapper->InsertKeyFrame(pKFini);

        ROS_DEBUG_STREAM("Setting key frame...");

        last_frame_ = current_frame_;
        last_key_frame_ = ref_key_frame_;
        last_key_frame_->setRefKeyFrame(ref_key_frame_); // reference is itself
        camera_pose_history_.push_back(current_frame_->cameraInWorldT());

        state_ = TrackingState::OK;
        ROS_DEBUG_STREAM("state_: OK");
    }
}

} // namespace orb_slam