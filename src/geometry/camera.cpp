/**
 *
 * This file implements the Camera class.
 *
 * @author <A href="mailto:saifullah3396@gmail.com">Saifullah</A>
 */

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
    inv_focal_x_ = 1.0 / focal_x_;
    inv_focal_y_ = 1.0 / focal_y_;
    nh_.getParam(prefix + "center_x", center_x_);
    nh_.getParam(prefix + "center_y", center_y_);
    nh_.getParam(prefix + "dist_coeffs", center_x_);

    // read dist coefficients list
    std::vector<T> dist_coeffs;
    nh_.getParam(prefix + "dist_coeffs", dist_coeffs)
    dist_coeffs_.at<T>(0, 0) = dist_coeffs[0];
    dist_coeffs_.at<T>(0, 1) = dist_coeffs[1];
    dist_coeffs_.at<T>(0, 2) = dist_coeffs[2];
    dist_coeffs_.at<T>(0, 3) = dist_coeffs[3];
    dist_coeffs_.at<T>(0, 4) = dist_coeffs[4];

    // update the intrinsic matrix
    updateIntrinsicMatrix();
}

template <typename T>
void Camera<T>::updateIntrinsicMatrix() {
    intrinsic_matrix =
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
        undist_key_points[i] =
            cv::KeyPoint(mat.at<float>(i, 0), mat.at<float>(i, 1))
    }
}

template struct Camera<float>;
template struct Camera<double>;

template <typename T>
MonoCamera<T>::MonoCamera(const ros::NodeHandle& nh) :
    Camera<T>(nh)
{
}

template <typename T>
MonoCamera<T>::~MonoCamera()
{
}

} // namespace geometry

} // namespace orb_slam