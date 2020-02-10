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

namespace orb_slam
{

class Frame;
using FramePtr = std::shared_ptr<Frame>;

namespace geometry
{

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
     * @param filter_matches: Matches are filtered if true
     */
    void match(
        const FramePtr& frame,
        const FramePtr& ref_frame,
        std::vector<cv::DMatch>& matches,
        bool filter_matches = true);

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
        bool filter_matches = true);

    /**
     * Filters out the matched points based on min/max distance threshold
     * @param descriptors: Input descriptors
     * @param matches: Found matches
     */
    void filterMatches(
        const cv::Mat& descriptors,
        std::vector<cv::DMatch>& matches);

private:
    //! ros node handle for reading parameters
    ros::NodeHandle nh_;

    //! orb matcher parameters
    std::string method_;
    std::function<void(
        const std::vector<cv::KeyPoint>&,
        const std::vector<cv::KeyPoint>&,
        const cv::Mat&,
        const cv::Mat&,
        std::vector<cv::DMatch>&)> match_;


    //! opencv orb extractors
    cv::Ptr<cv::ORB> cv_orb_detector_;
    cv::Ptr<cv::ORB> cv_orb_descriptor_;
    std::shared_ptr<MatcherBase> matcher_;
};

using ORBMatcherPtr = std::shared_ptr<ORBMatcher>;

} // namespace geometry

} // namespace orb_slam