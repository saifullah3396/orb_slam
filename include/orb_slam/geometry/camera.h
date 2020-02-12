/**
 * This file declares the Camera class.
 *
 * @author <A href="mailto:saifullah3396@gmail.com">Saifullah</A>
 */

#pragma once

#include <iostream>
#include <opencv2/core/core.hpp>
#include <string>
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

#define ROS_CAMERA_STREAM 1

namespace orb_slam
{

class Tracker;
using TrackerPtr = std::shared_ptr<Tracker>;

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
     * @param nh: ROS node handle
     */
    Camera(const ros::NodeHandle& nh);

    /**
     * @brief ~Camera Destructor
     */
    virtual ~Camera();

    /**
     * @brief readParams Reads the camera parameters
     */
    void readParams();

    /**
     * @brief setup Sets up the camera variables.
     */
    void setup();

    /**
     * @brief setup Sets up the camera image streaming
     */
    virtual void setupCameraStream() {}

    /**
     * Calls this function after image is received and allocated to camera
     */
    void onImageReceived() {}

    /**
     * @brief undistortPoints Performs undistortion operation on array of key
     *     points
     * @param key_points: Input distorted key points
     * @param undist_key_points: Output undistorted key points
     */
    void undistortPoints(
        std::vector<cv::KeyPoint>& key_points,
        std::vector<cv::KeyPoint>& undist_key_points);

    /**
     * @brief computeImageBounds Computes the image bounds while taking into
     *     account the distortion in the image
     */
    void computeImageBounds();

    /**
     * Setters
     */
    void setTracker(const TrackerPtr& tracker) {
        tracker_ = tracker;
    }

    /**
     * Getters
     */
    #ifdef ROS_CAMERA_STREAM
    virtual const bool subscribed() = 0;
    #endif
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
    virtual const CameraType type() = 0;
    virtual cv::Mat image() = 0;
    virtual const cv::Mat& imageL() = 0;
    virtual const cv::Mat& imageR() = 0;
    virtual const cv::Mat& imageDepth() = 0;
    const ros::Time& last_timestamp() = 0;
    const cv::Mat_<T>& distCoeffs() { return dist_coeffs_; }
    const cv::Mat_<T>& intrinsicMatrix() { return intrinsic_matrix_; }

protected:
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
    std::vector<cv::Point2f> undist_bounds_;
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
    cv::Mat_<T> dist_coeffs_ = {cv::Mat_<T>(1, 5)}; //! Distortion coefficients
    //! Camera intrinsic matrix of the form
    //! [fx 0 cx]
    //! [0 fy cy]
    //! [0  0  1]
    cv::Mat_<T> intrinsic_matrix_ = {cv::Mat_<T>(3, 3)};

    // ROS node handle for image streaming
    ros::NodeHandle nh_;

    TrackerPtr tracker_; // Pointer to the tracker class
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
     * @param nh: ROS node handle
     */
    MonoCamera(const ros::NodeHandle& nh);

    /**
     * @brief ~Camera Destructor
     */
    ~MonoCamera();

    /**
     * Updates the image in the camera
     */
    void onImageReceived();

    /**
     * @brief setup Sets up the camera image streaming
     */
    virtual void setupCameraStream();

    /**
     * Getters
     */
    cv_bridge::CvImageConstPtr image() { return cv_image_; }
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
    const ros::Time& last_timestamp() {
        return cv_image_->header.stamp;
    }

private:
    #ifdef ROS_CAMERA_STREAM
    virtual const bool subscribed() {
        return rgb_image_subscriber_.getNumPublishers() > 0;
    }
    void imageCb(
        const sensor_msgs::ImageConstPtr& image_msg,
        const sensor_msgs::CameraInfoConstPtr& camera_info_msg);

    sensor_msgs::CameraInfoConstPtr rgb_image_info_;
    cv_bridge::CvImageConstPtr cv_image_;
    image_transport::CameraSubscriber rgb_image_subscriber_;
    std::shared_ptr<image_transport::ImageTransport> image_transport;
    #endif
};

template <typename T = float>
using CameraPtr = std::shared_ptr<Camera<T>>;
template <typename T = float>
using MonoCameraPtr = std::shared_ptr<MonoCamera<T>>;

} // namespace geometry

} // namespace orb_slam