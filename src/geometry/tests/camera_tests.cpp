/**
 * Defines tests for Camera class.
 */

#include <gtest/gtest.h>
#include <ros/ros.h>
#include <orb_slam/geometry/camera.h>

using namespace orb_slam::geometry;

TEST (MonoCameraTester, TestMonoCameraFloatReadParams) {
    ros::NodeHandle nh;
    auto camera_float_ =
        CameraPtr<float>(new MonoCamera<float>());
    camera_float_->readParams(nh);
    EXPECT_EQ(camera_float_->type(), CameraType::MONO);
    EXPECT_EQ(camera_float_->fps(), 30);
    EXPECT_EQ(camera_float_->width(), 640);
    EXPECT_EQ(camera_float_->height(), 480);
    EXPECT_EQ(camera_float_->fovX(), 518);
    EXPECT_EQ(camera_float_->fovY(), 519);
    EXPECT_NEAR(camera_float_->focalX(), 517.3, 1e-3);
    EXPECT_NEAR(camera_float_->focalY(), 516.5, 1e-3);
    EXPECT_NEAR(camera_float_->invFocalX(), 1.0 / 517.3, 1e-3);
    EXPECT_NEAR(camera_float_->invFocalY(), 1.0 / 516.5, 1e-3);
    EXPECT_NEAR(camera_float_->centerX(), 318.6, 1e-3);
    EXPECT_NEAR(camera_float_->centerY(), 255.3, 1e-3);
    EXPECT_NEAR(camera_float_->distCoeffs()(0, 0), 0.2624, 1e-3);
    EXPECT_NEAR(camera_float_->distCoeffs()(0, 1), -0.9531, 1e-3);
    EXPECT_NEAR(camera_float_->distCoeffs()(0, 2), -0.0054, 1e-3);
    EXPECT_NEAR(camera_float_->distCoeffs()(0, 3), 0.0026, 1e-3);
    EXPECT_NEAR(camera_float_->distCoeffs()(0, 4), 1.1633, 1e-3);
}

TEST (MonoCameraTester, TestMonoCameraFloatSetup) {
    ros::NodeHandle nh;
    auto camera_float_ =
        CameraPtr<float>(new MonoCamera<float>());
    camera_float_->readParams(nh);
    camera_float_->setup();
    auto mat = camera_float_->intrinsicMatrix();
    EXPECT_EQ(mat(0, 0), camera_float_->focalX());
    EXPECT_EQ(mat(0, 1), 0);
    EXPECT_EQ(mat(0, 2), camera_float_->centerX());
    EXPECT_EQ(mat(1, 0), 0);
    EXPECT_EQ(mat(1, 1), camera_float_->focalY());
    EXPECT_EQ(mat(1, 2), camera_float_->centerY());
    EXPECT_EQ(mat(2, 0), 0);
    EXPECT_EQ(mat(2, 1), 0);
    EXPECT_EQ(mat(2, 2), 1);

    EXPECT_NEAR(camera_float_->minX(), float(10.803), 1e-3);
    EXPECT_NEAR(camera_float_->maxX(), float(626.059), 1e-3);
    EXPECT_NEAR(camera_float_->minY(), float(14.684), 1e-3);
    EXPECT_NEAR(camera_float_->maxY(), float(473.324), 1e-3);
    EXPECT_EQ(camera_float_->undistWidth(),
        int(camera_float_->maxX() - camera_float_->minX()));
    EXPECT_EQ(camera_float_->undistHeight(),
        int(camera_float_->maxY() - camera_float_->minY()));
}

TEST (MonoCameraTester, TestMonoCameraDoubleReadParams) {
    ros::NodeHandle nh;
    auto camera_double_ =
        CameraPtr<double>(new MonoCamera<double>());
    camera_double_->readParams(nh);
        EXPECT_EQ(camera_double_->type(), CameraType::MONO);
    EXPECT_EQ(camera_double_->fps(), 30);
    EXPECT_EQ(camera_double_->width(), 640);
    EXPECT_EQ(camera_double_->height(), 480);
    EXPECT_EQ(camera_double_->fovX(), 518);
    EXPECT_EQ(camera_double_->fovY(), 519);
    EXPECT_NEAR(camera_double_->focalX(), 517.3, 1e-3);
    EXPECT_NEAR(camera_double_->focalY(), 516.5, 1e-3);
    EXPECT_NEAR(camera_double_->invFocalX(), 1.0 / 517.3, 1e-3);
    EXPECT_NEAR(camera_double_->invFocalY(), 1.0 / 516.5, 1e-3);
    EXPECT_NEAR(camera_double_->centerX(), 318.6, 1e-3);
    EXPECT_NEAR(camera_double_->centerY(), 255.3, 1e-3);
    EXPECT_NEAR(camera_double_->distCoeffs()(0, 0), 0.2624, 1e-3);
    EXPECT_NEAR(camera_double_->distCoeffs()(0, 1), -0.9531, 1e-3);
    EXPECT_NEAR(camera_double_->distCoeffs()(0, 2), -0.0054, 1e-3);
    EXPECT_NEAR(camera_double_->distCoeffs()(0, 3), 0.0026, 1e-3);
    EXPECT_NEAR(camera_double_->distCoeffs()(0, 4), 1.1633, 1e-3);
}


int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    ros::init(argc, argv, "camera_tests");
    return RUN_ALL_TESTS();
}
