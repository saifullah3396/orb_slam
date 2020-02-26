/**
 * Declares the MonoTracker class.
 */

#pragma once

#include <ros/ros.h>
#include <orb_slam/tracker.h>

namespace orb_slam
{

class MonoTracker : public Tracker {
public:
    /**
     * Constructor
     * @param nh: ROS node handle
     */
    MonoTracker(const ros::NodeHandle& nh, const int& camera_type);

    /**
     * Destructor
     */
    ~MonoTracker();

    void update();
    void initializeTracking();

private:
    void createInitialMonocularMap(
        const std::vector<cv::Point2d>& inlier_points,
        const std::vector<cv::Point2d>& inlier_ref_points,
        const std::vector<cv::Point3d>& inlier_points_3d,
        const std::vector<size_t>& inliers_idxs);

    // monocular initialization
    InitializerPtr initializer_;
    double initializer_sigma_;
    int initializer_iterations_;

    // last image
    cv_bridge::CvImageConstPtr last_image_;
};
using MonoTrackerPtr = std::shared_ptr<MonoTracker>;

} // namespace orb_slam