/**
 * Defines tests for ORBMatcher class.
 */

#include <gtest/gtest.h>
#include <ros/ros.h>
#include <ros/package.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <orb_slam/geometry/orb_extractor.h>
#include <orb_slam/geometry/orb_matcher.h>

using namespace orb_slam::geometry;

TEST (ORBExtractorTester, TestFeatureMatching) {
    auto pkg_path = ros::package::getPath("orb_slam");
    auto image_1 =
        cv::imread(pkg_path + "/tests/test_images/1.png", CV_LOAD_IMAGE_COLOR);
    auto image_2 =
        cv::imread(pkg_path + "/tests/test_images/2.png", CV_LOAD_IMAGE_COLOR);
    ros::NodeHandle nh;
    auto orb_extractor = ORBExtractor(nh);
    std::vector<cv::KeyPoint> key_points_1, key_points_2;
    cv::Mat d_1, d_2;
    // find orb features in the image
    orb_extractor.detect(image_1, key_points_1);
    orb_extractor.detect(image_2, key_points_2);
    orb_extractor.compute(image_1, key_points_1, d_1);
    orb_extractor.compute(image_2, key_points_2, d_2);
    cv::drawKeypoints(
        image_1,
        key_points_1,
        image_1,
        cv::Scalar::all(-1),
        cv::DrawMatchesFlags::DEFAULT);
    auto orb_matcher = ORBMatcher(nh);
    std::vector<cv::DMatch> matches;
    orb_matcher.match(key_points_1, key_points_2, d_1, d_2, matches, false);
    EXPECT_TRUE(matches.size() >= 4500);
    orb_matcher.filterMatches(d_1, matches);
    EXPECT_TRUE((matches.size() >= 750) && (matches.size() <= 800));
    cv::Mat image_match;
    drawMatches(
        image_1, key_points_1, image_2, key_points_2, matches, image_match);

    int duplicates = 0;
    for (int i = 0; i < matches.size(); ++i) {
        for (int j = 0; j < matches.size(); ++j) {
            if (j == i) continue;
            if (matches[i].trainIdx == matches[j].trainIdx) duplicates++;
        }
    }
    EXPECT_TRUE(duplicates == 0);

    cv::imshow("Good matches", image_match);
    cv::waitKey(0);
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    ros::init(argc, argv, "orb_matcher_tests");
    return RUN_ALL_TESTS();
}
