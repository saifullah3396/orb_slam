/**
 *
 * This file implements the Camera class.
 *
 * @author <A href="mailto:saifullah3396@gmail.com">Saifullah</A>
 */

#include <opencv2/rgbd.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <orb_slam/geometry/rgbd_camera.h>

namespace orb_slam
{

namespace geometry
{

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

template <typename T>
std::shared_ptr<RGBDCamera<T>> RGBDCamera<T>::makeCamera(
    const ros::NodeHandle& nh, const std::string& input_type) {

    if (input_type == "ros") {
        return geometry::RGBDCameraPtr<T>(
            new geometry::ROSRGBDCamera<T>(nh));
    } else if (input_type == "tum_dataset") {
        return geometry::RGBDCameraPtr<T>(
            new geometry::TUMRGBDCamera<T>(nh));
    }
}

template <typename T>
void RGBDCamera<T>::readParams()
{
    // base class update
    Camera<T>::readParams();

    std::string prefix = "/orb_slam/depth_camera/";

    // read all the camera parameters
    this->nh_.getParam(prefix + "fps", fps_);
    this->nh_.getParam(prefix + "width_depth", width_depth_);
    this->nh_.getParam(prefix + "height_depth", height_depth_);
    this->nh_.getParam(prefix + "focal_x_depth", focal_x_depth_);
    this->nh_.getParam(prefix + "focal_y_depth", focal_y_depth_);
    focal_x_inv_depth_ = 1.0 / focal_x_depth_;
    focal_y_inv_depth_ = 1.0 / focal_y_depth_;
    this->nh_.getParam(prefix + "center_x_depth", center_x_depth_);
    this->nh_.getParam(prefix + "center_y_depth", center_y_depth_);

    // read dist coefficients list
    std::vector<T> dist_coeffs;
    this->nh_.getParam(
        prefix + "dist_coeffs_depth", dist_coeffs);
    dist_coeffs_depth_(0, 0) = dist_coeffs[0];
    dist_coeffs_depth_(0, 1) = dist_coeffs[1];
    dist_coeffs_depth_(0, 2) = dist_coeffs[2];
    dist_coeffs_depth_(0, 3) = dist_coeffs[3];
    dist_coeffs_depth_(0, 4) = dist_coeffs[4];
}

template <typename T>
void RGBDCamera<T>::updateIntrinsicMatrix() {
    Camera<T>::updateIntrinsicMatrix();
    intrinsic_matrix_depth_ =
        (
            cv::Mat_<T>(3, 3) <<
                focal_x_depth_, 0, center_x_depth_,
                0, focal_y_depth_, center_y_depth_,
                0,        0,         1
        );
}

template <typename T>
void RGBDCamera<T>::registerDepth(
    const cv::Mat& depth_in, cv::Mat& depth_out)
{
    cv::rgbd::registerDepth(
        intrinsic_matrix_depth_,
        this->intrinsic_matrix_,
        this->dist_coeffs_,
        image_T_depth_,
        depth_in,
        cv::Size(this->width_, this->height_),
        depth_out);
}

template <typename T>
ROSRGBDCamera<T>::ROSRGBDCamera(const ros::NodeHandle& nh) : RGBDCamera<T>(nh)
{
}

template <typename T>
ROSRGBDCamera<T>::~ROSRGBDCamera()
{
}

template <typename T>
void ROSRGBDCamera<T>::imageCb(
    const sensor_msgs::ImageConstPtr& image_msg,
    const sensor_msgs::CameraInfoConstPtr& image_info_msg,
    const sensor_msgs::ImageConstPtr& depth_msg,
    const sensor_msgs::CameraInfoConstPtr& depth_info_msg)
{
    // set info
    rgb_image_info_ = image_info_msg;
    depth_image_info_ = depth_info_msg;

    // set images
    cv_image_queue_.push(cv_bridge::toCvShare(image_msg));
    cv_bridge::CvImageConstPtr depth_image;
    if (this->preprocess_) {
        auto depth_image_temp =
            cv_bridge::toCvCopy(depth_msg, image_encodings::TYPE_32FC1);
        if ((fabs(this->depth_scale_ - 1.f) > 1e-5)) {
            depth_image_temp->image.convertTo(
                depth_image_temp->image, CV_16UC1, this->depth_scale_);
        }
        cv::Mat undist;
        cv::undistort(
            depth_image_temp->image,
            undist,
            this->intrinsic_matrix_depth_,
            this->dist_coeffs_depth_,
            this->intrinsic_matrix_depth_);
        //cv::Mat registered;
        //registerDepth(depth_image_temp->image, registered);
        //depth_image_temp->image = registered;
        depth_image_queue_.push(
            boost::const_pointer_cast<const cv_bridge::CvImage>(
                depth_image_temp));
    } else {
        cv_bridge::CvImageConstPtr depth_image =
            cv_bridge::toCvShare(depth_msg, image_encodings::TYPE_32FC1);
        depth_image_queue_.push(depth_image);
    }
    subscribed_ = true;
}

template <typename T>
void ROSRGBDCamera<T>::setupCameraStream()
{
    std::string prefix = "/orb_slam/depth_camera/", rgb_topic, rgb_info_topic;
    std::string depth_topic, depth_info_topic;

    // read all the camera parameters
    this->nh_.getParam(
        prefix + "rgb_topic", rgb_topic);
    this->nh_.getParam(
        prefix + "rgb_info_topic", rgb_info_topic);
    this->nh_.getParam(
        prefix + "depth_topic", depth_topic);
    this->nh_.getParam(
        prefix + "depth_info_topic", depth_info_topic);

    // queue size less than 10 will mostly drop images, we do not want missing
    // frames
    using namespace sensor_msgs;
    rgb_image_subscriber_ =
        std::shared_ptr<message_filters::Subscriber<Image>>(
            new message_filters::Subscriber<Image>(
                this->nh_, rgb_topic, 5));
    rgb_info_subscriber_ =
        std::shared_ptr<message_filters::Subscriber<CameraInfo>>(
            new message_filters::Subscriber<CameraInfo>(
                this->nh_, rgb_info_topic, 5));
    depth_image_subscriber_ =
        std::shared_ptr<message_filters::Subscriber<Image>>(
            new message_filters::Subscriber<Image>(
                this->nh_, depth_topic, 5));
    depth_info_subscriber_ =
        std::shared_ptr<message_filters::Subscriber<CameraInfo>>(
            new message_filters::Subscriber<CameraInfo>(
                this->nh_, depth_info_topic, 5));

    synchronizer_ =
        std::shared_ptr<message_filters::Synchronizer<SyncPolicy>>(
            new message_filters::Synchronizer<SyncPolicy>(
                SyncPolicy(10),
                *rgb_image_subscriber_,
                *rgb_info_subscriber_,
                *depth_image_subscriber_,
                *depth_info_subscriber_));
    synchronizer_->registerCallback(
        boost::bind(
            &RGBDCamera<T>::imageCb, this, _1, _2, _3, _4));
}

template class RGBDCamera<float>;
template class RGBDCamera<double>;
} // namespace geometry

} // namespace orb_slam