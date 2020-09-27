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
#include <orb_slam/geometry/camera.h>

using namespace sensor_msgs;

namespace orb_slam
{

namespace geometry
{

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
    virtual ~RGBDCamera();

    /**
     * @brief makeCamera Constructs a RGBDCamera based on given input type
     */
    static std::shared_ptr<RGBDCamera<T>> makeCamera(
        const ros::NodeHandle& nh, const std::string& input_type);
    /**
     * @brief readParams Reads the camera parameters
     */
    virtual void readParams();

    /**
     * @brief registerDepth Registers depth image to rgb image.
     */
    void registerDepth(
        const cv::Mat& depth_in, cv::Mat& depth_out);

    const CameraType type() const { return CameraType::RGBD; }
    const cv::Mat& imageL() {
        throw std::runtime_error(
            "imageL() is undefined for monocular camera.");
    }
    const cv::Mat& imageR() {
        throw std::runtime_error(
            "imageR() is undefined for monocular camera.");
    }
    const cv::Mat_<T>& distCoeffsDepth() const { return dist_coeffs_depth_; }
    const cv::Mat_<T>& intrinsicMatrixDepth() const { return intrinsic_matrix_depth_; }

protected:
    /**
     * @brief updateIntrinsicMatrix Updates the intrinsic matrix of the camera
     *     from current known parameters
     */
    void updateIntrinsicMatrix();

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

/**
 * @struct ROSRGBDCamera
 * @brief RGBDCamera with ROS based input
 */
template <typename T = float>
class ROSRGBDCamera : public RGBDCamera<T>
{
public:
    /**
     * @brief Camera Constructor
     * @param nh: ROS node handle
     */
    ROSRGBDCamera(const ros::NodeHandle& nh);

    /**
     * @brief ~Camera Destructor
     */
    ~ROSRGBDCamera();

    /**
     * @brief setup Sets up the camera image streaming
     */
    virtual void setupCameraStream();

    /**
     * @brief Pops the front element of the queue and returns it. Returns null if queue
     * is empty
     */
    virtual cv_bridge::CvImageConstPtr image() {
        if (!cv_image_queue_.empty()) {
            auto image = cv_image_queue_.front();
            cv_image_queue_.pop();
            return image;
        }
        return nullptr;
    }
    const CameraType type() const { return CameraType::RGBD; }
    const cv::Mat& imageL() {
        throw std::runtime_error(
            "imageL() is undefined for monocular camera.");
    }
    const cv::Mat& imageR() {
        throw std::runtime_error(
            "imageR() is undefined for monocular camera.");
    }
    virtual cv_bridge::CvImageConstPtr imageDepth() {
        if (!depth_image_queue_.empty()) {
            auto image = depth_image_queue_.front();
            depth_image_queue_.pop();
            return image;
        }
        return nullptr;
    }
private:
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

    bool subscribed_ = {false};
};

/**
 * @struct TUMRGBDCamera
 * @brief RGBDCamera with TUM dataset based input
 */
template <typename T = float>
class TUMRGBDCamera : public RGBDCamera<T>
{
public:
    /**
     * @brief Camera Constructor
     * @param nh: ROS node handle
     */
    TUMRGBDCamera(const ros::NodeHandle& nh);

    /**
     * @brief ~Camera Destructor
     */
    ~TUMRGBDCamera();

    virtual cv_bridge::CvImageConstPtr image() {
        if (img_count_ < rgb_files_.size()) {
            cv_bridge::CvImagePtr cv_image =
                cv_bridge::CvImagePtr(new cv_bridge::CvImage());
            cv_image->header.seq = img_count_++;
            cv_image->header.stamp = ros::Time(time_stamps_[img_count_]);
            cv_image->image =
                cv::imread(
                    dataset_dir_ + rgb_files_[img_count_],
                    CV_LOAD_IMAGE_UNCHANGED);
            return
                boost::static_pointer_cast<const cv_bridge::CvImage>(cv_image);
        } else {
            return nullptr;
        }
    }
    virtual cv_bridge::CvImageConstPtr imageDepth() {
        if (depth_count_ < depth_files_.size()) {
            cv_bridge::CvImagePtr cv_image =
                cv_bridge::CvImagePtr(new cv_bridge::CvImage());
            cv_image->header.seq = depth_count_++;
            cv_image->header.stamp = ros::Time(time_stamps_[depth_count_]);
            cv_image->image =
                cv::imread(dataset_dir_ + depth_files_[depth_count_]);
            cv_image->image.convertTo(
                cv_image->image, CV_32F, this->depth_scale_);
            return
                boost::static_pointer_cast<const cv_bridge::CvImage>(cv_image);
        } else {
            return nullptr;
        }
    }

    /**
     * @brief setup Sets up the camera image streaming
     */
    virtual void setupCameraStream();
private:
    virtual const bool subscribed() {
        return subscribed_;
    }
    bool subscribed_ = {false};

    std::vector<std::string> rgb_files_;
    std::vector<std::string> depth_files_;
    std::vector<double> time_stamps_;
    int img_count_ = 0;
    int depth_count_ = 0;
    std::string dataset_dir_;
};

template <typename T = float>
using RGBDCameraPtr = std::shared_ptr<RGBDCamera<T>>;
template <typename T = float>
using RGBDCameraConstPtr = std::shared_ptr<const RGBDCamera<T>>;

} // namespace geometry

} // namespace orb_slam