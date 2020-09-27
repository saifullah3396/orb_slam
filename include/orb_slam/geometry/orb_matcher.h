/**
 * This file declares the ORBMatcher class.
 *
 * @author <A href="mailto:saifullah3396@gmail.com">Saifullah</A>
 */

#pragma once

#include <iostream>
#include <memory>
#include <ros/ros.h>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d.hpp>
#include <orb_slam/geometry/utils.h>

namespace orb_slam
{

class Frame;
using FramePtr = std::shared_ptr<Frame>;

class KeyFrame;
using KeyFramePtr = std::shared_ptr<KeyFrame>;

class MapPoint;
using MapPointPtr = std::shared_ptr<MapPoint>;

namespace geometry
{

//! Types of matchers
enum OrbMatcherTypes {
    BF_WITH_RADIUS,
    BF_WITH_PROJ,
    BOW_ORB,
    EPIPOLAR_CONSTRAINT,
    CV_ORB,
    MATCHER_TYPES
};

/**
 * Base class for all matcher types
 */
struct MatcherBase {
    /**
     * Returns radius to use based on cosine of angle between vector from frame
     * to the 3d point and global view vector of point.
     */
    float radiusByViewCosine(const float& view_cosine) {
        if (view_cosine > 0.998)
            return 2.5;
        else
            return 4.0;
    }

    void applyRotationConstraint(
        std::vector<int>* rot_hist,
        std::vector<int>& matched,
        const int& hist_length_)
    {
        // apply rotation consistency
        int ind1 = -1;
        int ind2 = -1;
        int ind3 = -1;

        // take the 3 top most histograms
        geometry::computeThreeMaxima(rot_hist, hist_length_, ind1, ind2, ind3);

        // remove all the points that have rotation difference other than the
        // top 3 histogram bins.
        for (int i = 0; i < hist_length_; i++) {
            if (i != ind1 && i != ind2 && i != ind3) {
                for (size_t j = 0, jend = rot_hist[i].size(); j < jend; j++) {
                    matched[rot_hist[i][j]] = false;
                }
            }
        }
    }

    void createMatches(
        const std::vector<int>& matched,
        const std::vector<int>& feature_dists,
        std::vector<cv::DMatch>& matches) {
        // create matches
        for (size_t i = 0; i < matched.size(); ++i) {
            if (matched[i] > 0) { // matches from frame to reference frame
                matches.push_back(
                    cv::DMatch(
                        i, matched[i], static_cast<float>(feature_dists[i])));
            }
        }
    }

    std::string name_tag_ = {"Matcher"};
};

/**
 * Defines a brute force matcher for matching features between two frames
 */
struct BruteForceWithRadiusMatcher : public MatcherBase {
    /**
     * @brief Constructor
     * @param nh: ROS node handle for reading parameters
     */
    BruteForceWithRadiusMatcher(const ros::NodeHandle& nh);

    /**
     * @brief Finds the closest feature in ref_frame for a feature in frame by
     *     iterating over all the pixels that lie within a tolerance of feature
     *     in ref_frame and matching their descriptors
     * @param key_points: Input key points to match
     * @param ref_key_points: Reference key points to match with
     * @param descriptors: Input key points descriptors
     * @param ref_descriptors: Reference key points descriptors
     * @param matches: Output features that are matched
     */
    void match(
        const std::vector<cv::KeyPoint>& key_points,
        const std::vector<cv::KeyPoint>& ref_key_points,
        const cv::Mat& descriptors,
        const cv::Mat& ref_descriptors,
        std::vector<cv::DMatch>& matches);

    double max_matching_pixel_dist_;
    double max_matching_pixel_dist_sqrd_;
};

/**
 * Defines a brute force matcher for matching features between two frames after
 * 3d to 2d projection from one frame to another
 */
struct BruteForceWithProjectionMatcher : public MatcherBase {
    /**
     * @brief Constructor
     * @param nh: ROS node handle for reading parameters
     */
    BruteForceWithProjectionMatcher(const ros::NodeHandle& nh) {}

    /**
     * @brief Finds the closest feature in ref_frame for a feature in frame by
     *     iterating over all the pixels that lie within a tolerance of feature
     *     in ref_frame and matching their descriptors
     * @param frame: Input frame to match
     * @param ref_frane: Reference frame to match with
     * @param matches: Output features that are matched
     */
    void match(
        const FramePtr& frame,
        const FramePtr& ref_frame,
        std::vector<cv::DMatch>& matches);

    /**
     * @brief Finds the closest feature in ref_key_frame for a feature in frame by
     *     iterating over all the pixels that lie within a tolerance of feature
     *     in ref_key_frame and matching their descriptors
     * @param frame: Input frame to match
     * @param ref_frane: Reference frame to match with
     * @param matches: Output features that are matched
     */
    void match(
        const FramePtr& frame,
        const KeyFramePtr& ref_frame,
        std::vector<cv::DMatch>& matches);

    /**
     * @brief Finds the closest feature in ref_frame for a feature in frame by
     *     iterating over all the pixels that lie within a tolerance of feature
     *     in ref_key_frame and matching their descriptors
     * @param frame: Input frame to match
     * @param ref_frane: Reference frame to match with
     * @param ref_T_f: Transformation from frame to reference frame
     * @param ref_map_points: Map points observed in reference frame
     * @param matches: Output features that are matched
     */
    void match(
        const FramePtr& frame,
        const FramePtr& ref_frame,
        const cv::Mat& ref_T_f,
        const std::vector<MapPointPtr>& ref_map_points,
        std::vector<cv::DMatch>& matches);

    /**
     * @brief Finds the closest point in points_3d for a feature in frame by
     *     iterating over all the pixels that lie within a tolerance of projected
     *     point in frame and matching their descriptors
     * @param frame: Input frame to match
     * @param points_3d: Reference frame to match with
     * @param matches: Output features that are matched
     */
    void match(
        const FramePtr& frame,
        const std::vector<MapPointPtr>& points_3d,
        std::vector<cv::DMatch>& matches);

    float radius_ = 1.0;
    float nn_ratio_ = 0.6;
    bool compute_track_info_ = true;
    bool check_orientation_ = false;
    const int hist_length_ = 30;
    const int low_threshold_ = 50;
    const int high_threshold_ = 100;
};

/**
 * Defines a orb feature matcher for matching features between two frames
 * corresponding to the same bag of words
 */
struct BowOrbMatcher : public MatcherBase {
    /**
     * @brief Constructor
     * @param nh: ROS node handle for reading parameters
     */
    BowOrbMatcher(const ros::NodeHandle& nh) {}

    /**
     * @brief Finds the closest feature in ref_key_frame for a feature in frame by
     *     iterating over all the pixels that lie within a tolerance of feature
     *     in ref_key_frame and matching their descriptors
     * @param frame: Input frame to match
     * @param ref_frane: Reference frame to match with
     * @param matches: Output features that are matched
     */
    void match(
        const FramePtr& frame,
        const KeyFramePtr& ref_key_frame,
        std::vector<cv::DMatch>& matches);

    /**
     * @brief Finds the closest feature in ref_key_frame for a feature in key_frame by
     *     iterating over all the pixels that lie within a tolerance of feature
     *     in ref_key_frame and matching their descriptors
     * @param frame: Input frame to match
     * @param ref_frane: Reference frame to match with
     * @param matches: Output features that are matched
     */
    void match(
        const KeyFramePtr& key_frame,
        const KeyFramePtr& ref_key_frame,
        std::vector<cv::DMatch>& matches);

    /**
     * @brief Finds the closest feature in ref_frame for a feature in frame by
     *     iterating over all the pixels that lie within a tolerance of feature
     *     in ref_frame and matching their descriptors
     * @param frame: Input frame to match
     * @param ref_frane: Reference frame to match with
     * @param map_points: Map points associated with frame
     * @param ref_map_points: Map points associated with ref_frame
     * @param matches: Output features that are matched
     */
    void match(
        const FramePtr& frame,
        const FramePtr& ref_frame,
        const std::vector<MapPointPtr>& map_points,
        const std::vector<MapPointPtr>& ref_map_points,
        std::vector<cv::DMatch>& matches);

    bool check_orientation_ = false;
    float nn_ratio_ = 0.6;
    const int hist_length_ = 30;
    const int low_threshold_ = 50;
    const int high_threshold_ = 100;
};

/**
 * Defines a matcher for matching features between two frames based on epipolar
 * constraint and speed-up based on bow features.
 */
struct EpipolarConstraintWithBowMatcher : public MatcherBase {
    /**
     * @brief Constructor
     * @param nh: ROS node handle for reading parameters
     */
    EpipolarConstraintWithBowMatcher(const ros::NodeHandle& nh) {}

    /**
     * @brief Finds the closest feature in ref_frame for a feature in frame by
     *     iterating over all features within the same orb vocabulary word and
     *     with minimum distance to epipolar lines.
     * @param frame: Input frame to match
     * @param ref_frane: Reference frame to match with
     * @param matches: Output features that are matched
     */
    void match(
        const KeyFramePtr& key_frame,
        const KeyFramePtr& ref_key_frame,
        std::vector<cv::DMatch>& matches);

    void match(
        const FramePtr& frame,
        const FramePtr& ref_frame,
        const cv::Mat& fundamental_mat,
        const std::vector<MapPointPtr>& map_points,
        const std::vector<MapPointPtr>& ref_map_points,
        std::vector<cv::DMatch>& matches);

    bool check_epipolar_dist(
        const cv::KeyPoint& kp1,
        const cv::KeyPoint& kp2,
        const cv::Mat& f_mat,
        const FramePtr& ref_frame) const;

    bool check_orientation_ = false;
    const int hist_length_ = 30;
    const int low_threshold_ = 50;
    const int high_threshold_ = 100;
};

/**
 * Defines the opencv orb feature matcher
 */
struct CVORBMatcher : public MatcherBase {
    /**
     * @brief Constructor
     * @param nh: ROS node handle for reading parameters
     */
    CVORBMatcher(const ros::NodeHandle& nh);

    /**
     * @brief Matches the descriptors of two images using desired opencv-based
     *     matcher.
     * @param descriptors: Input key points descriptors
     * @param ref_descriptors: Reference key points descriptors
     * @param matches: Output features that are matched
     */
    void match(
        const cv::Mat& descriptors,
        const cv::Mat& ref_descriptors,
        std::vector<cv::DMatch>& matches);

    cv::Ptr<cv::DescriptorMatcher> matcher_;
};

/**
 * @struct ORBMatcher
 * @brief The class that is used to match orb features between two image frames
 */
class ORBMatcher
{
public:
    ORBMatcher(const ros::NodeHandle& nh);
    ~ORBMatcher();

    /**
     * Matches the input key points
     *
     * @param key_points: Input key points to match
     * @param ref_key_points: Reference key points to match with
     * @param descriptors: Input key points descriptors
     * @param ref_descriptors: Reference key points descriptors
     * @param matches: Output features that are matched
     * @param filter_matches: Matches are filtered if true
     */
    void match(
        const std::vector<cv::KeyPoint>& key_points,
        const std::vector<cv::KeyPoint>& ref_key_points,
        const cv::Mat& descriptors,
        const cv::Mat& ref_descriptors,
        std::vector<cv::DMatch>& matches,
        bool filter_matches = true) const;

    /**
     * Matches the input key points
     *
     * @param descriptors: Input key points descriptors
     * @param ref_descriptors: Reference key points descriptors
     * @param matches: Output features that are matched
     * @param filter_matches: Matches are filtered if true
     */
    void match(
        const cv::Mat& descriptors,
        const cv::Mat& ref_descriptors,
        std::vector<cv::DMatch>& matches,
        bool filter_matches = true) const;

    /**
     * Matches the input key points with output points using bow features
     *
     * @param frame: Input frame to match
     * @param ref_frame: Reference frame to match with
     * @param matches: Output features that are matched
     * @param check_orientation: Also checks orb feature orientations while
     *   matching if true
     * @param nn_ratio: Best to second best match ratio threshold. Best match
     *   distance should be at least smaller than nn_ratio.
     */
    void matchByBowFeatures(
        const FramePtr& frame,
        const KeyFramePtr& ref_frame,
        std::vector<cv::DMatch>& matches,
        const bool check_orientation = true,
        const float nn_ratio = 0.6) const;

    /**
     * Matches the input key points with output points using 3d to 2d
     * projection from reference frame over frame
     *
     * @param frame: Input frame to match
     * @param ref_frame: Reference frame to match with
     * @param matches: Output features that are matched
     * @param check_orientation: Also checks orb feature orientations while
     *   matching if true
     * @param radius: Window size multiplier for search
     */
    void matchByProjection(
        const FramePtr& frame,
        const FramePtr& ref_frame,
        std::vector<cv::DMatch>& matches,
        const bool check_orientation = true,
        const float radius = 1.0) const;

    /**
     * Matches the input key points with output points using 3d to 2d
     * projection from reference frame over frame
     *
     * @param frame: Input frame to match
     * @param ponits_3d: Points to project and match
     * @param matches: Output features that are matched
     * @param compute_track_info: Whether to compute map points track info for
     *     processing or should consider it to be already computed
     * @param nn_ratio: Best to second best match ratio threshold. Best match
     *   distance should be at least smaller than nn_ratio.
     * @param radius: Window size multiplier for search
     */
    void matchByProjection(
        const FramePtr& frame,
        const std::vector<MapPointPtr>& points_3d,
        std::vector<cv::DMatch>& matches,
        const bool compute_track_info,
        const float nn_ratio = 0.6,
        const float radius = 1.0) const;

    /**
     * Matches the features of one frame by another using the epipolar
     * constraint.
     * @param frame: Input frame to match
     * @param ref_frame: Reference frame to match with
     * @param matches: Output features that are matched
     * @param check_orientation: Also checks orb feature orientations while
     *   matching if true
     * @param filter_matches: Matches are filtered if true
     */
    void matchByEpipolarConstraint(
        const KeyFramePtr& frame,
        const KeyFramePtr& ref_frame,
        std::vector<cv::DMatch>& matches,
        const bool check_orientation = true
    ) const;

    /**
     * Projects the points in 3d on to the frame and removes duplicates.
     * @param key_frame: Frame to project on
     * @param points_3d: Points in 3d to project
     * @param radius_multiplier: Threshold radius for matching
     */
    int fuse(
        const KeyFramePtr key_frame,
        const std::vector<MapPointPtr>& points_3d,
        const float radius_multiplier = 3.0
    ) const;

    /**
     * Filters out the matched points based on min/max distance threshold
     * @param descriptors: Input descriptors
     * @param matches: Found matches
     */
    void filterMatches(
        const cv::Mat& descriptors,
        std::vector<cv::DMatch>& matches) const;

private:
    //! ros node handle for reading parameters
    ros::NodeHandle nh_;

    //! orb matcher parameters
    std::string method_;

    //! opencv orb extractors
    std::vector<std::shared_ptr<MatcherBase> > matcher_;

    const int hist_length_ = 30;
    const int low_threshold_ = 50;
    const int high_threshold_ = 100;
    std::string name_tag_ = {"ORBMatcher"};
};

using ORBMatcherPtr = std::shared_ptr<ORBMatcher>;
using ORBMatcherConstPtr = std::shared_ptr<const ORBMatcher>;

} // namespace geometry

} // namespace orb_slam