/**
 * Declares the Tracker class.
 */

namespace orb_slam
{

namespace geometry {
    template <typename T = float>
    class Camera;
    template <typename T = float>
    using CameraPtr = std::shared_ptr<Camera<T>>;

    class ORBExtractor;
    using ORBExtractorPtr = std::shared_ptr<ORBExtractor>;
}

class Tracker
{
public:
    Tracker(const ros::NodeHandle& nh);
    ~Tracker();

    void run();

private:
    cv::Mat getLatestImage();

    // ros node handle
    ros::NodeHandle nh_;

    // pointer to camera
    geometry::CameraPtr<float> camera_;
    geometry::ORBExtractorPtr orb_extractor_;
};

} // namespace orb_slam