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
 * Camera type
 */
enum class CameraType {
    MONO
};

/**
 * @struct Camera
 * @brief Holds information about a single camera
 */
template <typename T = float>
class Camera
{
public:
    /**
     * @brief Camera Constructor
     */
    Camera();

    /**
     * @brief ~Camera Destructor
     */
    virtual ~Camera();

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

    /**
     * @brief computeImageBounds Computes the image bounds while taking into
     *     account the distortion in the image
     */
    void computeImageBounds();

    /**
     * Getters
     */
    const int& fps() { return fps_; }
    const int& width() { return width_; }
    const int& height() { return height_; }
    const int& undistWidth() { return undist_width_; }
    const int& undistHeight() { return undist_height_; }
    const T& minX() const { return min_x_; }
    const T& minY() const { return min_y_; }
    const T& maxX() const { return max_x_; }
    const T& maxY() const { return max_y_; }
    const T& fovX() const { return fov_x_; }
    const T& fovY() const { return fov_y_; }
    const T& focalX() const { return focal_x_; }
    const T& focalY() const { return focal_y_; }
    const T& invFocalX() const { return focal_x_inv_; }
    const T& invFocalY() const { return focal_y_inv_; }
    const T& centerX() const { return center_x_; }
    const T& centerY() const { return center_y_; }
    virtual const cv::Mat& image() = 0;
    virtual const CameraType type() = 0;
    virtual const cv::Mat& imageL() = 0;
    virtual const cv::Mat& imageR() = 0;
    virtual const cv::Mat& imageDepth() = 0;
    const cv::Mat& distCoeffs() { return dist_coeffs_; }
    const cv::Mat& intrinsicMatrix() { return intrinsic_matrix_; }

private:
    /**
     * @brief readParams Reads the camera para
     * @param nh: ROS node handle
     */
    void readParams(const ros::NodeHandle& nh);

    /**
     * @brief setup Sets up the camera variables.
     */
    void setup();

    /**
     * @brief updateIntrinsicMatrix Updates the intrinsic matrix of the camera
     *     from current known parameters
     */
    void updateIntrinsicMatrix();

    CameraType type_; //! Camera type
    int fps_ = {30}; //! Frames per second for the video
    int width_ = {0}; //! Image width
    int height_ = {0}; //! Image height
    int undist_width_ = {0}; //! Width after distortion is removed
    int undist_height_ = {0}; //! Height after distortion is removed
    cv::Mat undist_bounds_ = {cv::Mat_<T>(4, 2, CV_32F)};
    T min_x_; //! Min x after removed distortion
    T max_x_; //! Max x after removed distortion
    T min_y_; //! Min y after removed distortion
    T max_y_; //! Max y after removed distortion
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
     */
    MonoCamera();

    /**
     * @brief ~Camera Destructor
     */
    ~MonoCamera();

    const cv::Mat& image() { return image_; }
    const CameraType type() { return CameraType::MONO; }
    const cv::Mat& imageL() {
        throw std::runtime_error(
            "imageL() is undefined for monocular camera.");
    }
    const cv::Mat& imageR() {
        throw std::runtime_error(
            "imageR() is undefined for monocular camera.");
    }
    const cv::Mat& imageDepth() {
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