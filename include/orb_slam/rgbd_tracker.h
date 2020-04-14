/**
 * Declares the RGBDTracker class.
 */

#pragma once

#include <ros/ros.h>
#include <orb_slam/tracker.h>

namespace orb_slam
{

class KeyFrame;
using KeyFramePtr = std::shared_ptr<KeyFrame>;

class RGBDTracker : public Tracker {
public:
    /**
     * Constructor
     * @param nh: ROS node handle
     */
    RGBDTracker(const ros::NodeHandle& nh, const int& camera_type);

    /**
     * Destructor
     */
    ~RGBDTracker();

    void update();
    void initializeTracking();

private:
    cv_bridge::CvImageConstPtr last_image_;
    cv_bridge::CvImageConstPtr last_depth_;

    float depth_map_scale_; // for some datasets depth maps are scaled
};
using RGBDTrackerPtr = std::shared_ptr<RGBDTracker>;

} // namespace orb_slam