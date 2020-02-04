/**
 * This file declares the Camera class.
 *
 * @author <A href="mailto:saifullah3396@gmail.com">Saifullah</A>
 */

#pragma once

#include <iostream>
#include <opencv2/core/core.hpp>
#include <ros/ros.h>
#include <string>

namespace orb_slam
{

namespace geometry
{

/**
 * @struct Camera
 * @brief Holds information about a single camera
 */
template <typename T = float>
class Camera
{
public:
    /**
     * Camera type
     */
    enum class CameraType {
        MONO
    };

    /**
     * @brief Camera Constructor
     * @param nh ROS node handle
     */
    Camera(const ros::NodeHandle& nh)

    /**
     * @brief ~Camera Destructor
     */
    virtual ~Camera();

    /**
     * Getters
     */
    uint16_t fps() { return fps_; }
    uint32_t width() { return width_; }
    uint32_t height() { return height_; }
    T fovX() { return fov_x_; }
    T fovY() { return fov_y_; }
    T focalX() { return focal_x_; }
    T focalY() { return focal_y_; }
    T invFocalX() { return focal_x_inv_; }
    T invFocalY() { return focal_y_inv_; }
    T centerX() { return center_x_; }
    T centerY() { return center_y_; }
    virtual cv::Mat image() = 0;
    virtual cv::Mat imageL() = 0;
    virtual cv::Mat imageR() = 0;
    virtual cv::Mat imageDepth() = 0;
    cv::Mat distCoeffs() { return dist_coeffs_; }
    cv::Mat intrinsicMatrix() { return intrinsic_matrix_; }

private:
    /**
     * @brief setup Reads the camera parameters from ros parameter server and
     *     updates them in class
     */
    void setup();

    /**
     * @brief updateIntrinsicMatrix Updates the intrinsic matrix of the camera
     *     from current known parameters
     */
    void updateIntrinsicMatrix();

    /**
     * @brief undistortPoints Performs undistortion operation on array of key
     *     points
     * @param key_points: Input distorted key points
     * @param undist_key_points: Output undistorted key points
     * @param undist_intrinsic_matrix: The output undistorted intrinsic matrix
     */
    void undistortPoints(
        std::vector<cv::KeyPoint>& key_points,
        std::vector<cv::KeyPoint>& undist_key_points,
        cv::Mat& undist_intrinsic_matrix);

    uint16_t fps_ = {30}; //! Frames per second for the video
    uint32_t width_ = {0}; //! Image width
    uint32_t height_ = {0}; //! Image height
    T fov_x_ = {0}; //! Camera field of view X
    T fov_y_ = {0}; //! Camera field of view Y
    T focal_x_ = {0}; //! Camera focal length X
    T focal_y_ = {0};//! Camera focal length Y
    T focal_x_inv_ = {0}; //! Inverse of Camera focal length X
    T focal_y_inv_ = {0};//! Inverse of Camera focal length Y
    T center_x_ = {0}; //! Camera center offset X
    T center_y_ = {0}; //! Camera center offset Y
    cv::Mat image_; //! Camera image
    cv::Mat dist_coeffs_ = {cv::Mat_<T>(1, 5)}; //! Distortion coefficients
    //! Camera intrinsic matrix of the form
    //! [fx 0 cx]
    //! [0 fy cy]
    //! [0  0  1]
    cv::Mat intrinsic_matrix_ = {cv::Mat_<T>(3, 3)};

    //! ros node handle for reading parameters
    ros::NodeHandle nh_;
};

/**
 * @struct MonoCamera
 * @brief Holds information about a single monocular camera
 */
template <typename T = float>
class MonoCamera : public Camera<T>
{
public:
    /**
     * @brief Camera Constructor
     * @param nh ROS node handle
     */
    MonoCamera(const ros::NodeHandle& nh);

    /**
     * @brief ~Camera Destructor
     */
    ~MonoCamera();

    cv::Mat image() { return image_; }
    cv::Mat imageL() {
        throw std::runtime_error(
            "imageL() is undefined for monocular camera.");
    }
    cv::Mat imageR() {
        throw std::runtime_error(
            "imageR() is undefined for monocular camera.");
    }
    cv::Mat imageDepth() {
        throw std::runtime_error(
            "imageDepth() is undefined for monocular camera.");
    }

private:
    cv::Mat image_; //! Camera image
};

template <typename T = float>
using CameraPtr = std::shared_ptr<Camera<T>>;
template <typename T = float>
using MonoCameraPtr = std::shared_ptr<MonoCamera<T>>;

} // namespace geometry

} // namespace orb_slam