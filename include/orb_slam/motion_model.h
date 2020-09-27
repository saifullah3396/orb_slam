/**
 * Declares the MotionModel class.
 */

#include <memory>
#include <ros/ros.h>
#include <sophus/se3.hpp>
#include <opencv2/core/eigen.hpp>

namespace orb_slam
{

template <typename T = float>
class MotionModel {
public:
    MotionModel() {
        velocity_.setZero();
    }
    ~MotionModel() {}

    /**
     * Predicts the world pose in new camera pose based on previous world pose
     * and previous velocity.
     * @param predicted_pose: Predicted camera to world pose
     * @param time: Time at which prediction is done
     */
    bool predict(Sophus::SE3<T>& predicted_pose, const ros::Time& time) {
        if (!initialized_) {
            //ROS_DEBUG_STREAM_NAMED(name_tag_, "Motion model uninitilized. Cannot predict.");
            return false;
        }
        auto time_diff = time - last_time_;
        predicted_pose =
            Sophus::SE3<T>::exp(velocity_ * time_diff.toSec()) * last_pose_;
        return true;
    }

    /**
     * Wrapper for predict for cv::Mat
     * @param predicted_pose: Predicted camera to world pose
     * @param time: Time at which prediction is done
     */
    bool predict(cv::Mat& predicted_pose, const ros::Time& time) {
        Sophus::SE3<T> predicted_pose_SE3;
        if (!predict(predicted_pose_SE3, time))
            return false;
        cv::eigen2cv(predicted_pose_SE3.matrix(), predicted_pose);
        return true;
    }

    /**
     * Updates the motion model using current pose and current time stamp.
     * @param current_pose: Current camera to world pose
     * @param time: Timestamp of the current pose
     */
    void updateModel(const Sophus::SE3<T>& current_pose, const ros::Time& time) {
        if (!last_time_.isZero()) {
            // this makes c_T_w * last_T_w
            time_diff_ = time - last_time_;
            velocity_ =
                (current_pose * last_pose_.inverse()).log() /
                time_diff_.toSec();

            initialized_ = true;
        }
        last_pose_ = current_pose;
        last_time_ = time;
    }

    /**
     * Wrapper for updateModel for cv::Mat
     * @param current_pose: Current camera to world pose
     * @param time: Timestamp of the current pose
     */
    void updateModel(const cv::Mat& current_pose, const ros::Time& time) {
        Eigen::Matrix<T, 4, 4> current_pose_eigen;
        cv::cv2eigen(current_pose, current_pose_eigen);
        auto current_pose_SE3 = Sophus::SE3<T>(current_pose_eigen);
        updateModel(current_pose_SE3, time);
    }

    /**
     * Getters
     */
    const bool& initialized() const { return initialized_; }

private:
    ros::Time last_time_ = ros::Time(0); // time stamp of last frame
    // time stamp difference of current frame and last frame
    ros::Duration time_diff_;
    // pose velocity from latest frame to previous frame
    Eigen::Matrix<T, 6, 1> velocity_;
    Sophus::SE3<T> last_pose_; // last pose of world in camera
    bool initialized_ = {false};

    std::string name_tag_ = {"MotionModel"};
};

template <typename T = float>
using MotionModelPtr = std::shared_ptr<MotionModel<T>>;
template <typename T = float>
using MotionModelConstPtr = std::shared_ptr<const MotionModel<T>>;

} // namespace orb_slam