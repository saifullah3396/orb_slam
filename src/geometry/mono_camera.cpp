/**
 *
 * This file implements the Camera class.
 *
 * @author <A href="mailto:saifullah3396@gmail.com">Saifullah</A>
 */

#include <opencv2/rgbd.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <orb_slam/geometry/mono_camera.h>

namespace orb_slam
{

namespace geometry
{

template <typename T>
MonoCamera<T>::MonoCamera(const ros::NodeHandle& nh) : Camera<T>(nh)
{
}

template <typename T>
MonoCamera<T>::~MonoCamera()
{
}

template <typename T>
ROSMonoCamera<T>::ROSMonoCamera(const ros::NodeHandle& nh) : MonoCamera<T>(nh)
{
}

template <typename T>
ROSMonoCamera<T>::~ROSMonoCamera()
{
}

template <typename T>
std::shared_ptr<MonoCamera<T>> MonoCamera<T>::makeCamera(
    const ros::NodeHandle& nh, const std::string& input_type) {

    if (input_type == "ros") {
        return geometry::MonoCameraPtr<T>(
            new geometry::ROSMonoCamera<T>(nh));
    }
}

template <typename T>
void ROSMonoCamera<T>::imageCb(
    const sensor_msgs::ImageConstPtr& image_msg,
    const sensor_msgs::CameraInfoConstPtr& camera_info_msg)
{
    cv_image_queue_.push(cv_bridge::toCvShare(image_msg));
    rgb_image_info_ = camera_info_msg;
}

template <typename T>
void ROSMonoCamera<T>::setupCameraStream()
{
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
            rgb_topic, 10, &ROSMonoCamera<T>::imageCb, this);
}

template class MonoCamera<float>;
template class MonoCamera<double>;

} // namespace geometry

} // namespace orb_slam