/**
 * Defines the Frame class.
 */

#pragma once

#include <array>
#include <memory>
#include <orb_slam/geometry/camera.h>
#include <orb_slam/geometry/orb_extractor.h>
#include <orb_slam/geometry/orb_matcher.h>
#include <orb_slam/geometry/utils.h>
#include <orb_slam/orb_vocabulary.h>
#include <orb_slam/map_point.h>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>

namespace orb_slam {

template<class T>
using Grid = std::vector<std::vector<T>>;

class MapPoint;
using MapPointPtr = std::shared_ptr<MapPoint>;

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
     * Computes the bag of words from orb vocabulary and frame features
     */
    void computeBow();

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
     * @param type: Matcher type to use
     */
    void match(
        const std::shared_ptr<Frame>& ref_frame,
        const geometry::OrbMatcherTypes type);

    /**
     * Matches the frame with a reference frame using bow feature matcher
     *
     * @param ref_frame: Frame to match with
     * @param check_orientation: Whether to check feature orientation match
     * @param nn_ratio: First to second best match ratio
     * @param filter_matches: Whether to filter matches afterwards
     */
    void matchByBowFeatures(
        const std::shared_ptr<Frame>& ref_frame,
        const bool check_orientation = true,
        const float nn_ratio = 0.6,
        const bool filter_matches = false);

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
    virtual void showMatchesWithRef(const std::string& name, const size_t n = 0) {}

    /**
     * @brief Checks whether a given point is within the image bounds
     * @param p: Point to check
     * @returns true if it is
     */
    bool pointWithinBounds(const cv::Point2f& p);

    /**
     * Computes the box indices around a point in the grid.
     * @param p: Point
     * @param box_radius: Radius of the box
     * @param left: Left index
     * @param right: Right index
     * @param up: Up index
     * @param down: Down index
     * @returns false if it lies outside the grid or image
     */
    bool getBoxAroundPoint(
        const cv::Point2f& p,
        const float& box_radius,
        int& left,
        int& right,
        int& up,
        int& down);

    /**
     * Finds the key points around a given point which lie within the radius.
     * @param p: Point
     * @param radius: Radius in which the other points are to be found
     * @param matches: The points found
     * @param returns false if failed
     */
    bool getFeaturesAroundPoint(
        const cv::Point2f& p,
        const float& radius,
        std::vector<size_t>& matches
    );

    /**
     * Finds the key points around a given point which lie within the radius
     * and above the given scale level.
     * @param p: Point
     * @param radius: Radius in which the other points are to be found
     * @param min_level: Minimum scale level
     * @param matches: The points found
     * @param returns false if failed
     */
    bool getFeaturesAroundPoint(
        const cv::Point2f& p,
        const float& radius,
        const int& min_level,
        std::vector<size_t>& matches
    );

    /**
     * Finds the key points around a given point which lie within the radius
     * and between the given scale levels.
     * @param p: Point
     * @param radius: Radius in which the other points are to be found
     * @param min_level: Minimum scale level
     * @param max_level: Maximum scale level
     * @param matches: The points found
     * @param returns false if failed
     */
    bool  getFeaturesAroundPoint(
        const cv::Point2f& p,
        const float& radius,
        const int& min_level,
        const int& max_level,
        std::vector<size_t>& matches
    );

    /**
     * Converts a point from frame camera coordinates to pixel coordinates
     *
     * @param p: Point in camera coordinates
     */
    template <typename U, typename V>
    cv::Point_<U> cameraToFrame(const cv::Point3_<V>& p) {
        return cv::Point_<U>(
                camera_->focalX() * p.x / p.z + camera_->centerX(),
                camera_->focalY() * p.y / p.z + camera_->centerY()
        );
    }

    /**
     * Converts a point from pixel oordinates to frame camera coordinates
     */
    template <typename U, typename V>
    cv::Point3_<U> frameToCamera(const cv::Point_<V>& p, const float& depth) {
        return cv::Point3_<U>(
                (p.x - camera_->centerX()) * depth * camera_->invFocalX(),
                (p.y - camera_->centerY()) * depth * camera_->invFocalY(),
                depth
        );
    }

    /**
     * Converts a point from camera coordinates to world coordinates
     */
    template <typename T>
    cv::Point3_<T> cameraToWorld(const cv::Point3_<T>& p) {
        return cv::Point3_<T>(cv::Mat(w_R_c_ * cv::Mat(p) + w_t_c_));
    }

    /**
     * Converts a point from world coordinates to camera coordinates
     */
    template <typename T>
    cv::Point3_<T> worldToCamera(const cv::Point3_<T>& p) {
        return cv::Point3_<T>(cv::Mat(c_R_w_ * cv::Mat(p) + c_t_w_));
    }

    /**
     * Converts a point from pixel coordinates to world coordinates
     */
    template <typename U, typename V>
    cv::Point3_<U> frameToWorld(const cv::Point_<V>& p, const float& depth) {
        return cameraToWorld<U>(frameToCamera<U, V>(p, depth));
    }

    /**
     * Converts a point from pixel coordinates to world coordinates
     */
    template <typename U, typename V>
    cv::Point3_<U> worldToFrame(const cv::Point3_<V>& p) {
        return cameraToFrame<U, V>(worldToCamera<V>(p));
    }

    /**
     * Getters
     */
    std::vector<MapPointPtr> obsMapPoints() const;
    const cv::Mat& getCamInWorldT() const { return w_T_c_; }
    const cv::Mat& getCamInWorldR() const { return w_R_c_; }
    const cv::Mat& getCamInWorldt() const { return w_t_c_; }
    const cv::Mat& getWorldInCamT() const { return c_T_w_; }
    const cv::Mat& getWorldInCamR() const { return c_R_w_; }
    const cv::Mat& getWorldInCamt() const { return c_t_w_; }
    const vector<bool>& outliers() const { return outliers_; }
    const int nFeatures() const { return key_points_.size(); }
    const std::vector<cv::KeyPoint>& features() const
        { return key_points_; }
    const int nFeaturesUndist() const { return undist_key_points_.size(); }
    const std::vector<cv::KeyPoint>& featuresUndist() const
        { return undist_key_points_; }
    virtual const std::vector<float>& featureDepthsUndist() const {
        throw std::runtime_error("Not implemented for this frame type.");
    }
    const cv::Mat& descriptorsUndist() const { return undist_descriptors_; }
    const int nDescriptorsUnDist() const { return undist_descriptors_.rows; }
    const std::vector<cv::DMatch> matches() const { return matches_; }
    const int nMatches() const { return matches_.size(); }
    const DBoW2::BowVector& bow() const { return bow_vec_; }
    const DBoW2::FeatureVector& bowFeatures() const { return feature_vec_; }
    virtual const cv_bridge::CvImageConstPtr& image() = 0;

    static const geometry::CameraConstPtr<float>& camera()
        { return camera_; }
    static const geometry::ORBExtractorConstPtr& orbExtractor()
        { return orb_extractor_; }
    static const geometry::ORBMatcherConstPtr& orbMatcher()
        { return orb_matcher_; }
    static const ORBVocabularyConstPtr& orbVocabulary()
        { return orb_vocabulary_;}

    /**
     * Setters
     */
    void setRefFrame(const FramePtr& ref_frame)
        { ref_frame_ = ref_frame; }
    void setPose(const cv::Mat& w_T_c) {
        w_T_c_ = w_T_c.clone(); // camera in world or world to camera
        w_R_c_ = w_T_c_.rowRange(0, 3).colRange(0, 3);
        w_t_c_ = w_T_c_.rowRange(0, 3).col(3);

        c_R_w_ = w_R_c_.t(); // transposed = inverse
        c_t_w_ = -c_R_w_ * w_t_c_; // -R.t() * t = translation inverse

         // world in camera or camera to world
        c_T_w_ = cv::Mat::eye(4, 4, CV_32F);
        c_R_w_.copyTo(c_T_w_.rowRange(0, 3).colRange(0, 3));
        c_t_w_.copyTo(c_T_w_.rowRange(0, 3).col(3));
    }
    static void setCamera(const geometry::CameraConstPtr<float>& camera)
        { camera_ = camera; }
    static void setORBExtractor(const geometry::ORBExtractorConstPtr& orb_extractor)
        { orb_extractor_ = orb_extractor; }
    static void setORBMatcher(const geometry::ORBMatcherConstPtr& orb_matcher)
        { orb_matcher_ = orb_matcher; }
    static void setORBVocabulary(
        const ORBVocabularyConstPtr& orb_vocabulary)
        { orb_vocabulary_ = orb_vocabulary; }

    void resizeMap(const size_t& n);
    void addMapPoint(const MapPointPtr& mp, const size_t& idx);
    void removeMapPointAt(const unsigned long& idx);

    void setOutlier(const size_t& idx, const bool is_outlier) {
        outliers_[idx] = is_outlier;
    }
    void setOutliers(const std::vector<bool>& outliers) {
        outliers_ = outliers;
    }

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
    cv::Mat w_T_c_; // camera in world transformation matrix
    cv::Mat w_R_c_; // camera in world rotation
    cv::Mat w_t_c_; // camera in world translation
    cv::Mat c_T_w_; // world in camera transformation matrix
    cv::Mat c_R_w_; // world in camera rotation
    cv::Mat c_t_w_; // world in camera translation
    std::vector<cv::KeyPoint> key_points_;
    std::vector<cv::KeyPoint> undist_key_points_;
    std::vector<bool> outliers_;
    cv::Mat undist_descriptors_;
    cv::Mat undist_intrinsic_matrix;

    // Bag of words vectors
    DBoW2::BowVector bow_vec_;
    DBoW2::FeatureVector feature_vec_;

    // frame matching info
    std::shared_ptr<Frame> ref_frame_;
    std::vector<cv::DMatch> matches_;

    // map points associated with frame
    std::vector<MapPointPtr> obs_map_points_;

    // static class pointers
    static geometry::CameraConstPtr<float> camera_; // camera info
    static geometry::ORBExtractorConstPtr orb_extractor_; // orb features extractor
    static geometry::ORBMatcherConstPtr orb_matcher_; // orb features extractor
    static ORBVocabularyConstPtr orb_vocabulary_; // orb vocabulary

    // uniform points extraction parameters
    Grid<std::vector<size_t>> grid_;
    static int grid_size_x_;
    static int grid_size_y_;
    static int grid_rows_;
    static int grid_cols_;

    static int id_global_; // global ids accumulator

    // map points access mutex
    std::mutex mutex_map_points_;

    // define key frame as friend for access
    friend class KeyFrame;
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
    void showMatchesWithRef(const std::string& name, const size_t n = 0);

    /**
     * Getters
     */
    const cv_bridge::CvImageConstPtr& image() { return image_; }

private:
    /**
     * Returns the derived camera class
     */
    geometry::MonoCameraConstPtr<float> camera();

    cv_bridge::CvImageConstPtr image_; // Frame image
};

class RGBDFrame : public Frame {
public:
    /**
     * Constructor
     *
     * @param image: Image assigned to this frame
     * @param depth: Image depth assigned to this frame
     * @param time_stamp: Frame time stamp on creation
     */
    RGBDFrame(
        const cv_bridge::CvImageConstPtr& image,
        const cv_bridge::CvImageConstPtr& depth,
        const ros::Time& time_stamp);

    /**
     * Destructor
     */
    ~RGBDFrame();

    /**
     * Extracts orb features from the frame image
     */
    virtual void extractFeatures();

    void drawFeatures(cv::Mat& image);
    void showImageWithFeatures(const std::string& name);
    void showMatchesWithRef(const std::string& name, const size_t n = 0);

    /**
     * Getters
     */
    const cv_bridge::CvImageConstPtr& image() { return image_; }
    const cv_bridge::CvImageConstPtr& depth() { return depth_; }
    const std::vector<float>& featureDepthsUndist() const {
        return undist_key_point_depths_;
    }

private:
    /**
     * Returns the derived camera class
     */
    geometry::RGBDCameraConstPtr<float> camera();

    cv_bridge::CvImageConstPtr image_; // Frame image
    cv_bridge::CvImageConstPtr depth_; // Frame image depth
    cv::Mat gray_image_; // gray_scale image
    bool rgb_ = {false};
    std::vector<float> undist_key_point_depths_;
};

//! pointer alias
using FramePtr = std::shared_ptr<Frame>;
using FrameConstPtr = std::shared_ptr<const Frame>;

} // namespace orb_slam