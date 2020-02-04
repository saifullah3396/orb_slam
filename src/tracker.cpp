/**
 * Defines the Tracker class.
 */

#include <ros/ros.h>
#include <vector>
#include <opencv2/core/core.hpp>
#include "orb_slam/frame.h"
#include "orb_slam/tracker.h"
#include "orb_slam/geometry/camera.h"
#include "orb_slam/geometry/orb_extractor.h"

namespace orb_slam
{

Tracker::Tracker(const ros::NodeHandle& nh) : nh_(nh)
{
    // initialize the camera
    camera_ = geometry::CameraPtr<float>(new geometry::MonoCamera<float>(nh_));
    orb_extractor_ =
        geometry::ORBExtractorPtr(new geometry::ORBExtractor(nh_));
    Frame::setCamera(camera_);
    Frame::setORBExtractor(orb_extractor_);
    Frame::setupUniformKeyPointsExtractor(nh_);
}

Tracker::~Tracker()
{

}

void Tracker::run() {
    std::vector<cv::Mat> camera_pose_history;
    while (true) {
        // create a key frame from the image
        FramePtr frame =
            FramePtr(new MonoFrame(camera_, orb_extractor_, ros::Time::now()));
        frame->extractFeatures();
        camera_pose_history.push_back(frame->getWorldToCamT().clone());
    }
}

cv::Mat Tracker::getLatestImage()
{
    return cv::Mat();
}

} // namespace orb_slam