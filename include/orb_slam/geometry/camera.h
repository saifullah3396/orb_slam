/**
 * This file declares the Camera class.
 *
 * @author <A href="mailto:saifullah3396@gmail.com">Saifullah</A>
 */

#pragma once

#include <opencv2/core/core.hpp>
#include <string>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>

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
    RGBD,
    STEREO
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
     * @brief makeCamera Constructs a camera of given type
     */
    static std::shared_ptr<Camera<T>> makeCamera(
        const ros::NodeHandle& nh, const geometry::CameraType& type);

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
     * @brief Checks whether a given point is within the image bounds
     * @param p: Point to check
     * @returns true if it is
     */
    bool pointWithinBounds(const cv::Point2f& p) const;

    /**
     * Getters
     */
    virtual const bool subscribed() = 0;
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
    virtual const CameraType type() const = 0;
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

template <typename T = float>
using CameraPtr = std::shared_ptr<Camera<T>>;
template <typename T = float>
using CameraConstPtr = std::shared_ptr<const Camera<T>>;

} // namespace geometry

} // namespace orb_slam