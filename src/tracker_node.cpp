/**
 * Initializes the ros node that updates the tracking
 * node of SLAM.
 */

#include <ros/ros.h>
#include <memory>
#include <orb_slam/tracker.h>
#include <ros/console.h>
using namespace orb_slam;

int main(int argc, char** argv)
{
    ros::init(argc, argv, "tracking_node");
    ros::NodeHandle nh;

    if(ros::console::set_logger_level(
            ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Debug)) {
        ros::console::notifyLoggerLevelsChanged();
    }

    int camera_type;
    nh.getParam("/orb_slam/tracker/camera_type", camera_type);
    auto tracker = Tracker::createTracker(nh, camera_type);
    auto rate = ros::Rate(1000);
    while (ros::ok()) {
        tracker->update();
        ros::spinOnce();
        rate.sleep();
    }
    return 0;
}