/**
 * Defines tests for ORBExtractor class.
 */

#include <gtest/gtest.h>
#include <ros/ros.h>
#include <ros/package.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <orb_slam/geometry/orb_extractor.h>

using namespace orb_slam::geometry;

TEST (ORBExtractorTester, TestFeatureDetection) {
    auto pkg_path = ros::package::getPath("orb_slam");
    auto image_1 =
        cv::imread(pkg_path + "/tests/test_images/1.png", CV_LOAD_IMAGE_COLOR);
    auto image_2 =
        cv::imread(pkg_path + "/tests/test_images/2.png", CV_LOAD_IMAGE_COLOR);
    ros::NodeHandle nh;
    auto orb_extractor = ORBExtractor(nh);
    std::vector<cv::KeyPoint> key_points_1, key_points_2;
    // find orb features in the image
    orb_extractor.detect(image_1, key_points_1);
    orb_extractor.detect(image_2, key_points_2);
    EXPECT_EQ(key_points_1.size(), 4583);
    EXPECT_EQ(key_points_2.size(), 4652);
    cv::drawKeypoints(
        image_1,
        key_points_1,
        image_1,
        cv::Scalar::all(-1),
        cv::DrawMatchesFlags::DEFAULT);
    cv::imshow("key_points_1", image_1);
}

TEST (ORBExtractorTester, TestFeatureDescription) {
    auto pkg_path = ros::package::getPath("orb_slam");
    auto image_1 =
        cv::imread(pkg_path + "/tests/test_images/1.png", CV_LOAD_IMAGE_COLOR);
    auto image_2 =
        cv::imread(pkg_path + "/tests/test_images/2.png", CV_LOAD_IMAGE_COLOR);
    ros::NodeHandle nh;
    auto orb_extractor = ORBExtractor(nh);
    std::vector<cv::KeyPoint> key_points_1, key_points_2;
    // find orb features in the image
    orb_extractor.detect(image_1, key_points_1);
    orb_extractor.detect(image_2, key_points_2);
    cv::Mat descriptors_1, descriptors_2;
    orb_extractor.compute(image_1, key_points_1, descriptors_1);
    orb_extractor.compute(image_2, key_points_2, descriptors_2);
    ROS_INFO_STREAM("descriptors_1:" << descriptors_1.rows);
    ROS_INFO_STREAM("descriptors_2:" << descriptors_2.rows);
    EXPECT_EQ(descriptors_1.rows, 4583);
    EXPECT_EQ(descriptors_2.rows, 4652);
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    ros::init(argc, argv, "orb_extractor_tests");
    return RUN_ALL_TESTS();
}
