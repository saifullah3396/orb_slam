/**
 * Initializes the ros node that updates the tracking
 * node of SLAM.
 */

#include <ros/ros.h>
#include "orb_slam/tracker.h"

namespace orb_slam {

int main(int argc, char** argv)
{
    ros::init(argc, argv, "tracking_node");
    ros::NodeHandle nh;
    Tracker(nh);
    auto rate = ros::Rate(1000);
    while (ros::ok()) {
        ros::spinOnce();
        rate.sleep();
    }
    return 0;
}

}