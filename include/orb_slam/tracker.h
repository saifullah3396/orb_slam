/**
 * Declares the Tracker class.
 */

#include <cv_bridge/cv_bridge.h>
#include <opencv2/core/core.hpp>
#include <memory.h>

namespace orb_slam
{

namespace geometry {
    template <typename T>
    class Camera;
    template <typename T>
    using CameraPtr = std::shared_ptr<Camera<T>>;

    class ORBExtractor;
    using ORBExtractorPtr = std::shared_ptr<ORBExtractor>;
    class ORBMatcher;
    using ORBMatcherPtr = std::shared_ptr<ORBMatcher>;
}

class Initializer;
using InitializerPtr = std::unique_ptr<Initializer>;

class Frame;
using FramePtr = std::shared_ptr<Frame>;

#define MIN_REQ_MATCHES 100

class Tracker {
public:
    /**
     * Constructor
     * @param nh: ROS node handle
     */
    Tracker(const ros::NodeHandle& nh);

    /**
     * Destructor
     */
    ~Tracker();

    /**
     * Updates the visual odometry tracking
     */
    void update();

private:
    void trackFrame();
    void monocularInitialization();

    // last image
    cv_bridge::CvImageConstPtr last_image_;

    // latest frame
    FramePtr current_frame_;
    std::vector<cv::Mat> camera_pose_history_; // vector of poses

    // monocular initialization
    FramePtr ref_frame_;
    FramePtr last_frame_;
    InitializerPtr initializer_;
    double initializer_sigma_;
    int initializer_iterations_;

    // ros node handle
    ros::NodeHandle nh_;

    // pointer to camera
    geometry::CameraPtr<float> camera_;
    geometry::ORBExtractorPtr orb_extractor_;
    geometry::ORBMatcherPtr orb_matcher_;

    // Tracking states same as in original orb slam package
    enum TrackingState {
        SYSTEM_NOT_READY = -1,
        NO_IMAGES_YET = 0,
        NOT_INITIALIZED = 1,
        OK = 2,
        LOST = 3
    };
    TrackingState state_;
    TrackingState last_proc_state_;
};
using TrackerPtr = std::shared_ptr<Tracker>;

} // namespace orb_slam