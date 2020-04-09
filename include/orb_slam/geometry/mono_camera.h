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
    virtual ~MonoCamera();

    /**
     * @brief makeCamera Constructs a MonoCamera based on given input type
     */
    static std::shared_ptr<MonoCamera<T>> makeCamera(
        const ros::NodeHandle& nh, const std::string& input_type);

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
    const CameraType type() const { return CameraType::MONO; }
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
    std::queue<cv_bridge::CvImageConstPtr> cv_image_queue_;
};

/**
 * @struct ROSMonoCamera
 * @brief Defines the MonoCamera with ROS based video input
 */
template <typename T = float>
class ROSMonoCamera : public MonoCamera<T>
{
    /**
     * @brief Camera Constructor
     * @param nh: ROS node handle
     */
    ROSMonoCamera(const ros::NodeHandle& nh);

    /**
     * @brief ~Camera Destructor
     */
    ~ROSMonoCamera();

    /**
     * @brief setup Sets up the camera image streaming
     */
    virtual void setupCameraStream();
private:
    const bool subscribed() {
        return rgb_image_subscriber_.getNumPublishers() > 0;
    }

    void imageCb(
        const sensor_msgs::ImageConstPtr& image_msg,
        const sensor_msgs::CameraInfoConstPtr& camera_info_msg);

    sensor_msgs::CameraInfoConstPtr rgb_image_info_;
    image_transport::CameraSubscriber rgb_image_subscriber_;
    std::shared_ptr<image_transport::ImageTransport> image_transport;
};

template <typename T = float>
using MonoCameraPtr = std::shared_ptr<MonoCamera<T>>;
template <typename T = float>
using MonoCameraConstPtr = std::shared_ptr<const MonoCamera<T>>;

} // namespace geometry

} // namespace orb_slam