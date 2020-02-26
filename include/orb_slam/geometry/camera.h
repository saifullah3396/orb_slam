/**
 * This file declares the Camera class.
 *
 * @author <A href="mailto:saifullah3396@gmail.com">Saifullah</A>
 */

#pragma once

#include <iostream>
#include <queue>
#include <opencv2/core/core.hpp>
#include <string>
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <cv_bridge/cv_bridge.h>

#define ROS_CAMERA_STREAM 1
using namespace sensor_msgs;

namespace orb_slam
{

namespace geometry
{

/**
 * Camera type
 */
enum class CameraType {
    MONO,
    RGBD
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
    virtual void readParams();

    /**
     * @brief setup Sets up the camera variables.
     */
    void setup();

    /**
     * @brief setup Sets up the camera image streaming
     */
    virtual void setupCameraStream() {}

    /**
     * @brief undistortPoints Performs undistortion operation on array of key
     *     points
     * @param key_points: Input distorted key points
     * @param undist_key_points: Output undistorted key points
     */
    void undistortPoints(
        std::vector<cv::KeyPoint>& key_points,
        std::vector<cv::KeyPoint>& undist_key_points) const;

    /**
     * @brief computeImageBounds Computes the image bounds while taking into
     *     account the distortion in the image
     */
    void computeImageBounds();

    /**
     * Getters
     */
    #ifdef ROS_CAMERA_STREAM
    virtual const bool subscribed() = 0;
    #endif
    const int& fps() const { return fps_; }
    const int& width() const { return width_; }
    const int& height() const { return height_; }
    const int& undistWidth() const { return undist_width_; }
    const int& undistHeight() const { return undist_height_; }
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
    virtual cv_bridge::CvImageConstPtr image() = 0;
    virtual const cv::Mat& imageL() = 0;
    virtual const cv::Mat& imageR() = 0;
    virtual cv_bridge::CvImageConstPtr imageDepth() = 0;
    const cv::Mat_<T>& distCoeffs() const { return dist_coeffs_; }
    const cv::Mat_<T>& intrinsicMatrix() const { return intrinsic_matrix_; }

    /**
     * Setters
     */
    void setPreprocess(const bool& preprocess) { preprocess_ = preprocess; }

protected:
    /**
     * @brief updateIntrinsicMatrix Updates the intrinsic matrix of the camera
     *     from current known parameters
     */
    virtual void updateIntrinsicMatrix();

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

    // Preprocessing toggle
    bool preprocess_ = {true};

    // ROS node handle for image streaming
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
     * @param nh: ROS node handle
     */
    MonoCamera(const ros::NodeHandle& nh);

    /**
     * @brief ~Camera Destructor
     */
    ~MonoCamera();

    /**
     * @brief setup Sets up the camera image streaming
     */
    virtual void setupCameraStream();

    /**
     * @brief Pops the front element of the queue and returns it. Returns null if queue
     * is empty
     */
    cv_bridge::CvImageConstPtr image() {
        if (!cv_image_queue_.empty()) {
            auto image = cv_image_queue_.front();
            cv_image_queue_.pop();
            return image;
        }
        return nullptr;
    }
    const CameraType type() { return CameraType::MONO; }
    const cv::Mat& imageL() {
        throw std::runtime_error(
            "imageL() is undefined for monocular camera.");
    }
    const cv::Mat& imageR() {
        throw std::runtime_error(
            "imageR() is undefined for monocular camera.");
    }
    cv_bridge::CvImageConstPtr imageDepth() {
        throw std::runtime_error(
            "imageDepth() is undefined for monocular camera.");
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
    std::queue<cv_bridge::CvImageConstPtr> cv_image_queue_;
    image_transport::CameraSubscriber rgb_image_subscriber_;
    std::shared_ptr<image_transport::ImageTransport> image_transport;
    #endif
};

/**
 * @struct RGBDCamera
 * @brief Holds information about a single rgbd camera
 */
template <typename T = float>
class RGBDCamera : public Camera<T>
{
public:
    /**
     * @brief Camera Constructor
     * @param nh: ROS node handle
     */
    RGBDCamera(const ros::NodeHandle& nh);

    /**
     * @brief ~Camera Destructor
     */
    ~RGBDCamera();

    /**
     * @brief readParams Reads the camera parameters
     */
    virtual void readParams();

    /**
     * @brief setup Sets up the camera image streaming
     */
    virtual void setupCameraStream();

    /**
     * @brief Pops the front element of the queue and returns it. Returns null if queue
     * is empty
     */
    cv_bridge::CvImageConstPtr image() {
        if (!cv_image_queue_.empty()) {
            auto image = cv_image_queue_.front();
            cv_image_queue_.pop();
            return image;
        }
        return nullptr;
    }
    const CameraType type() { return CameraType::RGBD; }
    const cv::Mat& imageL() {
        throw std::runtime_error(
            "imageL() is undefined for monocular camera.");
    }
    const cv::Mat& imageR() {
        throw std::runtime_error(
            "imageR() is undefined for monocular camera.");
    }
    cv_bridge::CvImageConstPtr imageDepth() {
        if (!depth_image_queue_.empty()) {
            auto image = depth_image_queue_.front();
            depth_image_queue_.pop();
            return image;
        }
        return nullptr;
    }
    const cv::Mat_<T>& distCoeffsDepth() const { return dist_coeffs_depth_; }
    const cv::Mat_<T>& intrinsicMatrixDepth() const { return intrinsic_matrix_depth_; }

private:
    /**
     * @brief updateIntrinsicMatrix Updates the intrinsic matrix of the camera
     *     from current known parameters
     */
    void updateIntrinsicMatrix();

    #ifdef ROS_CAMERA_STREAM
    virtual const bool subscribed() {
        return subscribed_;
    }
    void imageCb(
        const sensor_msgs::ImageConstPtr& image_msg,
        const sensor_msgs::CameraInfoConstPtr& image_info_msg,
        const sensor_msgs::ImageConstPtr& depth_msg,
        const sensor_msgs::CameraInfoConstPtr& depth_info_msg);

    sensor_msgs::CameraInfoConstPtr rgb_image_info_;
    sensor_msgs::CameraInfoConstPtr depth_image_info_;
    std::queue<cv_bridge::CvImageConstPtr> cv_image_queue_;
    std::queue<cv_bridge::CvImageConstPtr> depth_image_queue_;

    std::shared_ptr<message_filters::Subscriber<Image>> rgb_image_subscriber_;
    std::shared_ptr<message_filters::Subscriber<Image>> depth_image_subscriber_;
    std::shared_ptr<message_filters::Subscriber<CameraInfo>> rgb_info_subscriber_;
    std::shared_ptr<message_filters::Subscriber<CameraInfo>> depth_info_subscriber_;

    // camera stream synchronizer
    typedef message_filters::sync_policies::ApproximateTime<
        sensor_msgs::Image,
        sensor_msgs::CameraInfo,
        sensor_msgs::Image,
        sensor_msgs::CameraInfo> SyncPolicy;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> synchronizer_;
    #endif

    bool subscribed_ = {false};
    float depth_scale_ = {1.f}; // sometimes datasets have scaled depths

    int fps_ = {30}; //! Frames per second for the depth image
    int width_depth_;
    int height_depth_;
    T focal_x_depth_ = {0}; //! Camera focal length X
    T focal_y_depth_ = {0};//! Camera focal length Y
    T focal_x_inv_depth_ = {0}; //! Inverse of Camera focal length X
    T focal_y_inv_depth_ = {0};//! Inverse of Camera focal length Y
    T center_x_depth_ = {0}; //! Camera center offset X
    T center_y_depth_ = {0}; //! Camera center offset Y
    //! Depth distortion coefficients
    cv::Mat_<T> dist_coeffs_depth_ = {cv::Mat_<T>(1, 5)};
    cv::Mat_<T> intrinsic_matrix_depth_ = {cv::Mat_<T>(3, 3)};
    //! Depth camera to image camera static transform
    cv::Mat_<T> image_T_depth_;
};

template <typename T = float>
using CameraPtr = std::shared_ptr<Camera<T>>;
template <typename T = float>
using CameraConstPtr = std::shared_ptr<const Camera<T>>;

template <typename T = float>
using MonoCameraPtr = std::shared_ptr<MonoCamera<T>>;
template <typename T = float>
using MonoCameraConstPtr = std::shared_ptr<const MonoCamera<T>>;

template <typename T = float>
using RGBDCameraPtr = std::shared_ptr<RGBDCamera<T>>;
template <typename T = float>
using RGBDCameraConstPtr = std::shared_ptr<const RGBDCamera<T>>;

} // namespace geometry

} // namespace orb_slam