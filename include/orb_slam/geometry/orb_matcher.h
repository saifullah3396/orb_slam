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

namespace geometry
{

//! Types of matchers
enum OrbMatcherTypes {
    BF_WITH_RADIUS,
    BF_WITH_PROJ,
    BOW_ORB,
    CV_ORB,
    MATCHER_TYPES
};

/**
 * Base class for all matcher types
 */
struct MatcherBase {
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

    float radius_ = 1.0;
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

    bool check_orientation_ = false;
    float nn_ratio_ = 0.6;
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
     * Matches the key points in two frames
     *
     * @param frame: Input frame to match
     * @param ref_frame: Reference frame to match with
     * @param matches: Output features that are matched
     * @param type: Type of orb matcher to use
     * @param filter_matches: Matches are filtered if true
     */
    void match(
        const FramePtr& frame,
        const FramePtr& ref_frame,
        std::vector<cv::DMatch>& matches,
        const OrbMatcherTypes type,
        bool filter_matches = true) const;

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
     *   distance should be at least smaller than nn_ratio * second best match.
     * @param filter_matches: Matches are filtered if true
     */
    void matchByBowFeatures(
        const FramePtr& frame,
        const FramePtr& ref_frame,
        std::vector<cv::DMatch>& matches,
        const bool check_orientation = true,
        const float nn_ratio = 0.6,
        const bool filter_matches = false) const;

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
     * @param filter_matches: Matches are filtered if true
     */
    void matchByProjection(
        const FramePtr& frame,
        const FramePtr& ref_frame,
        std::vector<cv::DMatch>& matches,
        const bool check_orientation = true,
        const float radius = 1.0,
        const bool filter_matches = false) const;

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
};

using ORBMatcherPtr = std::shared_ptr<ORBMatcher>;
using ORBMatcherConstPtr = std::shared_ptr<const ORBMatcher>;

} // namespace geometry

} // namespace orb_slam