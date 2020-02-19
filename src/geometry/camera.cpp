/**
 *
 * This file implements the Camera class.
 *
 * @author <A href="mailto:saifullah3396@gmail.com">Saifullah</A>
 */

#include <opencv2/imgproc/imgproc.hpp>
#include <orb_slam/geometry/camera.h>

namespace orb_slam
{

namespace geometry
{

template <typename T>
Camera<T>::Camera(const ros::NodeHandle& nh): nh_(nh)
{
}

template <typename T>
Camera<T>::~Camera()
{
}

template <typename T>
void Camera<T>::readParams()
{
    std::string prefix = "/orb_slam/camera/";

    // read all the camera parameters
    nh_.param<int>("/orb_slam/camera/fps", fps_, 30);
    nh_.param<int>(prefix + "width", width_, 0);
    nh_.param<int>(prefix + "height", height_, 0);
    nh_.param<T>(prefix + "fov_x", fov_x_, 0.0);
    nh_.param<T>(prefix + "fov_y", fov_y_, 0.0);
    nh_.param<T>(prefix + "focal_x", focal_x_, 0.0);
    nh_.param<T>(prefix + "focal_y", focal_y_, 0.0);
    focal_x_inv_ = 1.0 / focal_x_;
    focal_y_inv_ = 1.0 / focal_y_;
    nh_.param<T>(prefix + "center_x", center_x_, 0.0);
    nh_.param<T>(prefix + "center_y", center_y_, 0.0);

    // read dist coefficients list
    std::vector<T> dist_coeffs;
    nh_.param<std::vector<T>>(
        prefix + "dist_coeffs", dist_coeffs, dist_coeffs);
    dist_coeffs_(0, 0) = dist_coeffs[0];
    dist_coeffs_(0, 1) = dist_coeffs[1];
    dist_coeffs_(0, 2) = dist_coeffs[2];
    dist_coeffs_(0, 3) = dist_coeffs[3];
    dist_coeffs_(0, 4) = dist_coeffs[4];
}

template <typename T>
void Camera<T>::setup()
{
    // update the intrinsic matrix
    updateIntrinsicMatrix();

    // compute the image bounds for given distortion
    computeImageBounds();
}

template <typename T>
void Camera<T>::updateIntrinsicMatrix() {
    intrinsic_matrix_ =
        (
            cv::Mat_<T>(3, 3) <<
                focal_x_, 0, center_x_,
                0, focal_y_, center_y_,
                0,        0,         1
        );
}

template <typename T>
void Camera<T>::undistortPoints(
    std::vector<cv::KeyPoint>& key_points,
    std::vector<cv::KeyPoint>& undist_key_points) const
{
    std::vector<cv::Point2f> points;
    for(auto it = key_points.begin(); it != key_points.end(); it++) {
        points.push_back(it->pt);
    }
    // perform undistortion
    cv::undistortPoints(
        points,
        points,
        intrinsic_matrix_,
        dist_coeffs_,
        cv::Mat(),
        intrinsic_matrix_);

    // Fill undistorted keypoint vector
    auto size = key_points.size();
    undist_key_points.resize(size);
    for(int i = 0; i < size; i++) {
        auto kp = key_points[i]; // copy the point
        kp.pt = points[i];
        undist_key_points[i] = kp;
    }
}

template <typename T>
void Camera<T>::computeImageBounds()
{
    // actually distorted right now
    undist_bounds_.push_back(cv::Point2f(0.0, 0.0));
    undist_bounds_.push_back(cv::Point2f(width_, 0.0));
    undist_bounds_.push_back(cv::Point2f(0.0, height_));
    undist_bounds_.push_back(cv::Point2f(width_, height_));

    if(dist_coeffs_(0, 0) != 0.0)
    {
        // perform undistortion
        cv::undistortPoints(
            undist_bounds_,
            undist_bounds_,
            intrinsic_matrix_,
            dist_coeffs_,
            cv::Mat(),
            intrinsic_matrix_);
    }

    min_x_ = // clip at zero
        std::max(std::min(undist_bounds_[0].x, undist_bounds_[2].x), (float)0);
    max_x_ = // clip at width_
        std::min(
            std::max(undist_bounds_[1].x, undist_bounds_[3].x), (float)width_);
    min_y_ = // clip at zero
        std::max(std::min(undist_bounds_[0].y, undist_bounds_[1].y), (float)0);
    max_y_ = // clip at height_
        std::min(
            std::max(undist_bounds_[2].y, undist_bounds_[3].y), (float)height_);

    undist_width_ = max_x_ - min_x_;
    undist_height_ = max_y_ - min_y_;
}

template class Camera<float>;
template class Camera<double>;

template <typename T>
MonoCamera<T>::MonoCamera(const ros::NodeHandle& nh) : Camera<T>(nh)
{
}

template <typename T>
MonoCamera<T>::~MonoCamera()
{
}

#ifdef ROS_CAMERA_STREAM
template <typename T>
void MonoCamera<T>::imageCb(
    const sensor_msgs::ImageConstPtr& image_msg,
    const sensor_msgs::CameraInfoConstPtr& camera_info_msg)
{
    cv_image_queue_.push(cv_bridge::toCvShare(image_msg));
    rgb_image_info_ = camera_info_msg;
}
#endif

template <typename T>
void MonoCamera<T>::setupCameraStream()
{
    #ifdef ROS_CAMERA_STREAM
    std::string prefix = "/orb_slam/camera/", rgb_topic;

    // read all the camera parameters
    this->nh_.getParam(
        prefix + "rgb_topic", rgb_topic);
    image_transport =
        std::shared_ptr<image_transport::ImageTransport>(
            new image_transport::ImageTransport(this->nh_));
    // queue size less than 10 will mostly drop images, we do not want missing
    // frames
    rgb_image_subscriber_ =
        image_transport->subscribeCamera(
            rgb_topic, 10, &MonoCamera<T>::imageCb, this);
    #endif
}

template class MonoCamera<float>;
template class MonoCamera<double>;

template <typename T>
RGBDCamera<T>::RGBDCamera(const ros::NodeHandle& nh) : Camera<T>(nh)
{
    this->nh_.getParam(
        "/orb_slam/depth_camera/depth_scale",
        depth_scale_);
}

template <typename T>
RGBDCamera<T>::~RGBDCamera()
{
}
} // namespace geometry

} // namespace orb_slam