/**
 * Defines tests for Utility functions.
 */

#include <gtest/gtest.h>
#include <ros/ros.h>
#include <ros/package.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <orb_slam/geometry/utils.h>

using namespace orb_slam::geometry;
using namespace cv;

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    ros::init(argc, argv, "utils_tests");
    ros::NodeHandle nh;
    return RUN_ALL_TESTS();
}
