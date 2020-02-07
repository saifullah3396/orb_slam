/**
 * Initializes the ros node that updates the tracking
 * node of SLAM.
 */

#include <ros/ros.h>
#include <memory>
#include <orb_slam/tracker.h>

using namespace orb_slam;

int main(int argc, char** argv)
{
    ros::init(argc, argv, "tracking_node");
    ros::NodeHandle nh;
    auto tracker =
        std::unique_ptr<Tracker>(new Tracker(nh));
    auto rate = ros::Rate(1000);
    while (ros::ok()) {
        tracker->run();
        ros::spinOnce();
        rate.sleep();
    }
    return 0;
}