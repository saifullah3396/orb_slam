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

    auto tracker =
        std::unique_ptr<Tracker>(new Tracker(nh));
    auto rate = ros::Rate(30);
    while (ros::ok()) {
        tracker->update();
        ros::spinOnce();
        rate.sleep();
    }
    return 0;
}