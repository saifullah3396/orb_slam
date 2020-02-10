/**
 * Defines tests for Camera class.
 */

#include <gtest/gtest.h>
#include <ros/ros.h>
#include <orb_slam/geometry/camera.h>

using namespace orb_slam::geometry;

TEST (MonoCameraTester, TestMonoCameraFloatReadParams) {
    ros::NodeHandle nh;
    auto camera =
        CameraPtr<float>(new MonoCamera<float>());
    camera->readParams(nh);
    EXPECT_EQ(camera->type(), CameraType::MONO);
    EXPECT_EQ(camera->fps(), 30);
    EXPECT_EQ(camera->width(), 640);
    EXPECT_EQ(camera->height(), 480);
    EXPECT_EQ(camera->fovX(), 518);
    EXPECT_EQ(camera->fovY(), 519);
    EXPECT_NEAR(camera->focalX(), 517.3, 1e-3);
    EXPECT_NEAR(camera->focalY(), 516.5, 1e-3);
    EXPECT_NEAR(camera->invFocalX(), 1.0 / 517.3, 1e-3);
    EXPECT_NEAR(camera->invFocalY(), 1.0 / 516.5, 1e-3);
    EXPECT_NEAR(camera->centerX(), 318.6, 1e-3);
    EXPECT_NEAR(camera->centerY(), 255.3, 1e-3);
    EXPECT_NEAR(camera->distCoeffs()(0, 0), 0.2624, 1e-3);
    EXPECT_NEAR(camera->distCoeffs()(0, 1), -0.9531, 1e-3);
    EXPECT_NEAR(camera->distCoeffs()(0, 2), -0.0054, 1e-3);
    EXPECT_NEAR(camera->distCoeffs()(0, 3), 0.0026, 1e-3);
    EXPECT_NEAR(camera->distCoeffs()(0, 4), 1.1633, 1e-3);
}

TEST (MonoCameraTester, TestMonoCameraFloatSetup) {
    ros::NodeHandle nh;
    auto camera =
        CameraPtr<float>(new MonoCamera<float>());
    camera->readParams(nh);
    camera->setup();
    auto mat = camera->intrinsicMatrix();
    EXPECT_EQ(mat(0, 0), camera->focalX());
    EXPECT_EQ(mat(0, 1), 0);
    EXPECT_EQ(mat(0, 2), camera->centerX());
    EXPECT_EQ(mat(1, 0), 0);
    EXPECT_EQ(mat(1, 1), camera->focalY());
    EXPECT_EQ(mat(1, 2), camera->centerY());
    EXPECT_EQ(mat(2, 0), 0);
    EXPECT_EQ(mat(2, 1), 0);
    EXPECT_EQ(mat(2, 2), 1);

    EXPECT_NEAR(camera->minX(), float(10.803), 1e-3);
    EXPECT_NEAR(camera->maxX(), float(626.059), 1e-3);
    EXPECT_NEAR(camera->minY(), float(14.684), 1e-3);
    EXPECT_NEAR(camera->maxY(), float(473.324), 1e-3);
    EXPECT_EQ(camera->undistWidth(),
        int(camera->maxX() - camera->minX()));
    EXPECT_EQ(camera->undistHeight(),
        int(camera->maxY() - camera->minY()));
}

TEST (MonoCameraTester, TestMonoCameraDoubleReadParams) {
    ros::NodeHandle nh;
    auto camera =
        CameraPtr<double>(new MonoCamera<double>());
    camera->readParams(nh);
        EXPECT_EQ(camera->type(), CameraType::MONO);
    EXPECT_EQ(camera->fps(), 30);
    EXPECT_EQ(camera->width(), 640);
    EXPECT_EQ(camera->height(), 480);
    EXPECT_EQ(camera->fovX(), 518);
    EXPECT_EQ(camera->fovY(), 519);
    EXPECT_NEAR(camera->focalX(), 517.3, 1e-3);
    EXPECT_NEAR(camera->focalY(), 516.5, 1e-3);
    EXPECT_NEAR(camera->invFocalX(), 1.0 / 517.3, 1e-3);
    EXPECT_NEAR(camera->invFocalY(), 1.0 / 516.5, 1e-3);
    EXPECT_NEAR(camera->centerX(), 318.6, 1e-3);
    EXPECT_NEAR(camera->centerY(), 255.3, 1e-3);
    EXPECT_NEAR(camera->distCoeffs()(0, 0), 0.2624, 1e-3);
    EXPECT_NEAR(camera->distCoeffs()(0, 1), -0.9531, 1e-3);
    EXPECT_NEAR(camera->distCoeffs()(0, 2), -0.0054, 1e-3);
    EXPECT_NEAR(camera->distCoeffs()(0, 3), 0.0026, 1e-3);
    EXPECT_NEAR(camera->distCoeffs()(0, 4), 1.1633, 1e-3);
}

TEST (MonoCameraTester, TestMonoCameraDoubleSetup) {
    ros::NodeHandle nh;
    auto camera =
        CameraPtr<double>(new MonoCamera<double>());
    camera->readParams(nh);
    camera->setup();
    auto mat = camera->intrinsicMatrix();
    EXPECT_EQ(mat(0, 0), camera->focalX());
    EXPECT_EQ(mat(0, 1), 0);
    EXPECT_EQ(mat(0, 2), camera->centerX());
    EXPECT_EQ(mat(1, 0), 0);
    EXPECT_EQ(mat(1, 1), camera->focalY());
    EXPECT_EQ(mat(1, 2), camera->centerY());
    EXPECT_EQ(mat(2, 0), 0);
    EXPECT_EQ(mat(2, 1), 0);
    EXPECT_EQ(mat(2, 2), 1);

    EXPECT_NEAR(camera->minX(), double(10.803), 1e-3);
    EXPECT_NEAR(camera->maxX(), double(626.059), 1e-3);
    EXPECT_NEAR(camera->minY(), double(14.684), 1e-3);
    EXPECT_NEAR(camera->maxY(), double(473.324), 1e-3);
    EXPECT_EQ(camera->undistWidth(),
        int(camera->maxX() - camera->minX()));
    EXPECT_EQ(camera->undistHeight(),
        int(camera->maxY() - camera->minY()));
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    ros::init(argc, argv, "camera_tests");
    return RUN_ALL_TESTS();
}
