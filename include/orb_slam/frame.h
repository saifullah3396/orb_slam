/**
 * Defines the Frame class.
 */

#include <array>
#include <memory>
#include <orb_slam/geometry/camera.h>
#include <orb_slam/geometry/orb_extractor.h>
#include <orb_slam/geometry/orb_matcher.h>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>

namespace orb_slam {

template<class T>
using Grid = std::vector<std::vector<T>>;

class Frame : public std::enable_shared_from_this<Frame> {
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
    static void setupGrid(const ros::NodeHandle& nh);

    /**
     * Extracts orb features from the frame image
     */
    virtual void extractFeatures() = 0;

    /**
     * Sets the frame to be the first and act as reference
     */
    virtual void setupFirstFrame();

    /**
     * Matches the frame with a reference frame using the orb feature matcher.
     *
     * @param ref_frame: Frame to matche with
     */
    void match(const std::shared_ptr<Frame>& ref_frame);

    /**
     * Draws the extracted features on the given image
     * @param image: Drawing image
     */
    virtual void drawFeatures(cv::Mat& image) {}

    /**
     * Draws the extracted features on a copy image and shows it
     * @param name: Image output name while showing
     */
    virtual void showImageWithFeatures(const std::string& name) {}

    /**
     * Shows the matches between this frame and reference frame.
     * @param name: Image output name while showing
     */
    virtual void showMatchesWithRef(const std::string& name) {}

    /**
     * Getters
     */
    const cv::Mat& getWorldToCamT() const { return c_T_w_; }
    const int nFeatures() const { return key_points_.size(); }
    const std::vector<cv::KeyPoint>& features() const
        { return key_points_; }
    const int nFeaturesUndist() const { return undist_key_points_.size(); }
    const std::vector<cv::KeyPoint>& featuresUndist() const
        { return undist_key_points_; }
    const cv::Mat& descriptorsUndist() const { return undist_descriptors_; }
    const int nDescriptorsUnDist() const { return undist_descriptors_.rows; }
    const std::vector<cv::DMatch> matches() const { return matches_; }
    const int nMatches() const { return matches_.size(); }
    virtual const cv_bridge::CvImageConstPtr& image() = 0;

    /**
     * Setters
     */
    static void setCamera(const geometry::CameraPtr<float>& camera)
        { camera_ = camera; }
    static void setORBExtractor(const geometry::ORBExtractorPtr& orb_extractor)
        { orb_extractor_ = orb_extractor; }
    static void setORBMatcher(const geometry::ORBMatcherPtr& orb_matcher)
        { orb_matcher_ = orb_matcher; }

protected:
    /**
     * Extracts key points uniformly over the image in a grid
     *
     * @param key_points_: Input key points that are updated in place according
     *     to uniform extraction parameters
     * @param grid: Input grid to be updated
     */
    static void assignFeaturesToGrid(
        std::vector<cv::KeyPoint>& key_points_,
        Grid<std::vector<size_t>>& grid);

    /**
     * Finds the location of a key point in grid
     * @param key_point: Input key point
     * @param int: x location
     * @param int: y location
     *
     * @return True if the point lies within the grid, false otherwise
     */
    static bool pointInGrid(
        const cv::KeyPoint& key_point, int& pos_x, int& pos_y);

    // frame info
    int id_; // frame id
    ros::Time time_stamp_; // frame time stamp
    cv::Mat c_T_w_; // world to camera transformation matrix
    std::vector<cv::KeyPoint> key_points_;
    std::vector<cv::KeyPoint> undist_key_points_;
    cv::Mat undist_descriptors_;
    cv::Mat undist_intrinsic_matrix;

    // frame matching info
    std::shared_ptr<Frame> ref_frame_;
    std::vector<cv::DMatch> matches_;

    // static class pointers
    static geometry::CameraPtr<float> camera_; // camera info
    static geometry::ORBExtractorPtr orb_extractor_; // orb features extractor
    static geometry::ORBMatcherPtr orb_matcher_; // orb features extractor

    // uniform points extraction parameters
    Grid<std::vector<size_t>> grid_;
    static int grid_size_x_;
    static int grid_size_y_;
    static int grid_rows_;
    static int grid_cols_;

    static int id_global_; // global ids accumulator
};

class MonoFrame : public Frame {
public:
    /**
     * Constructor
     *
     * @param image: Image assigned to this frame
     * @param time_stamp: Frame time stamp on creation
     */
    MonoFrame(
        const cv_bridge::CvImageConstPtr& image, const ros::Time& time_stamp);

    /**
     * Destructor
     */
    ~MonoFrame();

    /**
     * Extracts orb features from the frame image
     */
    virtual void extractFeatures();

    void drawFeatures(cv::Mat& image);
    void showImageWithFeatures(const std::string& name);
    void showMatchesWithRef(const std::string& name);

    /**
     * Getters
     */
    const cv_bridge::CvImageConstPtr& image() { return image_; }

private:
    /**
     * Returns the derived camera class
     */
    geometry::MonoCameraPtr<float> camera();

    cv_bridge::CvImageConstPtr image_; // Frame image
};

//! pointer alias
using FramePtr = std::shared_ptr<Frame>;

} // namespace orb_slam