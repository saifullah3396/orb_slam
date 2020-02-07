/**
 *
 * This file implements the Camera class.
 *
 * @author <A href="mailto:saifullah3396@gmail.com">Saifullah</A>
 */

#include <opencv2/imgproc/imgproc.hpp>
#include "orb_slam/geometry/camera.h"

namespace orb_slam
{

namespace geometry
{

template <typename T>
Camera<T>::Camera(const ros::NodeHandle& nh) : nh_(nh)
{
    setup();
}

template <typename T>
Camera<T>::~Camera()
{
}

template <typename T>
void Camera<T>::setup()
{
    std::string prefix = "orb_slam/camera/";

    // read all the camera parameters
    nh_.getParam(prefix + "fps", fps_);
    nh_.getParam(prefix + "width", width_);
    nh_.getParam(prefix + "height", height_);
    nh_.getParam(prefix + "fov_x", fov_x_);
    nh_.getParam(prefix + "fov_y", fov_y_);
    nh_.getParam(prefix + "focal_x", focal_x_);
    nh_.getParam(prefix + "focal_y", focal_y_);
    focal_x_inv_ = 1.0 / focal_x_;
    focal_y_inv_ = 1.0 / focal_y_;
    nh_.getParam(prefix + "center_x", center_x_);
    nh_.getParam(prefix + "center_y", center_y_);

    // read dist coefficients list
    std::vector<T> dist_coeffs;
    nh_.getParam(prefix + "dist_coeffs", dist_coeffs);
    dist_coeffs_.at<T>(0, 0) = dist_coeffs[0];
    dist_coeffs_.at<T>(0, 1) = dist_coeffs[1];
    dist_coeffs_.at<T>(0, 2) = dist_coeffs[2];
    dist_coeffs_.at<T>(0, 3) = dist_coeffs[3];
    dist_coeffs_.at<T>(0, 4) = dist_coeffs[4];

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
    std::vector<cv::KeyPoint>& undist_key_points,
    cv::Mat& undist_intrinsic_matrix)
{
    std::vector<cv::Point2f> points;
    for(auto it = key_points.begin(); it != key_points.end(); it++) {
        points.push_back(it->pt);
    }
    cv::Mat mat(points);
    cv::undistortPoints(
        mat, mat, intrinsic_matrix_, dist_coeffs_, undist_intrinsic_matrix);

    // Fill undistorted keypoint vector
    auto size = key_points.size();
    undist_key_points.resize(size);
    for(int i = 0; i < size; i++) {
        auto kp = key_points[i]; // copy the point
        kp.pt.x = mat.at<T>(i, 0);
        kp.pt.y = mat.at<T>(i, 1);
        undist_key_points[i] = kp;
    }
}

template <typename T>
void Camera<T>::computeImageBounds()
{
    // actually distorted right now
    undist_bounds_ =
        (
            cv::Mat_<T>(4, 2, CV_32F) <<
                0.0,    0.0,
                width_, 0.0,
                0.0,    height_,
                width_, height_
        );

    if(dist_coeffs_.at<T>(0) != 0.0)
    {
        // perform undistortion
        //mat=mat.reshape(2);
        cv::undistortPoints(
            undist_bounds_,
            undist_bounds_,
            intrinsic_matrix_,
            dist_coeffs_,
            cv::Mat(),
            intrinsic_matrix_);
        //mat=mat.reshape(1);
    }

    min_x_ =
        std::min(undist_bounds_.at<T>(0,0), undist_bounds_.at<T>(2,0));
    max_x_ =
        std::max(undist_bounds_.at<T>(1,0), undist_bounds_.at<T>(3,0));
    min_y_ =
        std::min(undist_bounds_.at<T>(0,1), undist_bounds_.at<T>(1,1));
    max_y_ =
        std::max(undist_bounds_.at<T>(2,1), undist_bounds_.at<T>(3,1));

    undist_width_ = max_x_ - min_x_;
    undist_height_ = max_y_ - min_y_;
}

template class Camera<float>;
template class Camera<double>;

template <typename T>
MonoCamera<T>::MonoCamera(const ros::NodeHandle& nh) :
    Camera<T>(nh)
{
}

template <typename T>
MonoCamera<T>::~MonoCamera()
{
}

template class MonoCamera<float>;
template class MonoCamera<double>;

} // namespace geometry

} // namespace orb_slam