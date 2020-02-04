/**
 * Defines the Frame class.
 */

#include <shared_ptr.hpp>
#include <orb_slam/geometry/camera.h>
#include <orb_slam/geometry/orb_extractor.h>

namespace orb_slam {

class Frame {
public:
    /**
     * Constructor
     *
     * @param time_stamp: Frame time stamp on creation
     */
    Frame(const ros::Time& time_stamp);
    /**
     * Destructor
     */
    virtual ~Frame();

    /**
     * Sets up the uniform key points extractor
     */
    static void setupUniformKeyPointsExtractor(const ros::NodeHandle& nh);

    /**
     * Extracts orb features from the frame image
     */
    virtual void extractFeatures() = 0;

    /**
     * Getters
     */
    cv::Mat getWorldToCamT() { return c_T_w_; }

    /**
     * Setters
     */
    static void setCamera(const geometry::CameraPtr<float>& camera)
        { camera_ = camera; }
    static void setORBExtractor(const geometry::ORBExtractorPtr& orb_extractor)
        { orb_extractor_ = orb_extractor; }

protected:
    /**
     * Extracts key points uniformly over the image in a grid
     *
     * @param key_points: Input key points that are updated in place according
     *     to uniform extraction parameters
     */
    static void extractUniformKeyPointsInGrid(
        std::vector<cv::KeyPoint>& key_points);

    int id_; // frame id
    ros::Time time_stamp_; // frame time stamp

    cv::Mat c_T_w_; // world to camera transformation matrix

    static geometry::CameraPtr<float> camera_; // camera info
    static geometry::ORBExtractorPtr orb_extractor_; // orb features extractor

    // uniform points extraction parameters
    static int extract_uniform_key_points_;
    static int max_key_points_;
    static int uniform_key_points_grid_size_;
    static int max_key_points_per_grid_;
    static int grid_rows_;
    static int grid_cols_;

    static int id_global_; // global ids accumulator
};

class MonoFrame : public Frame {
public:
    /**
     * Constructor
     *
     * @param time_stamp: Frame time stamp on creation
     */
    MonoFrame(const ros::Time& time_stamp);

    /**
     * Destructor
     */
    ~MonoFrame();

    /**
     * Extracts orb features from the frame image
     */
    virtual void extractFeatures();

private:
    /**
     * Returns the derived camera class
     */
    geometry::MonoCameraPtr<float> camera();
};

//! pointer alias
using FramePtr = std::shared_ptr<Frame>;

} // namespace orb_slam