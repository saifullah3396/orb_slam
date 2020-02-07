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
 * Defines a brute force matcher for matching features between two frames
 */
struct BruteForceWithRadiusMatcher {
    /**
     * @brief Constructor
     * @param nh: ROS node handle for reading parameters
     */
    BruteForceWithRadiusMatcher(const ros::NodeHandle& nh);

    /**
     * @brief Finds the closest feature in ref_frame for a feature in frame by
     *     iterating over all the pixels that lie within a tolerance of feature
     *     in ref_frame and matching their descriptors
     * @param frame: Matched frame
     * @param ref_frame: Reference frame for match
     */
    void match(
        const FramePtr& frame,
        const FramePtr& ref_frame,
        std::vector<cv::DMatch>& matches);

    double max_matching_pixel_dist_;
    double max_matching_pixel_dist_sqrd_;
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
     */
    void match(
        const FramePtr& frame,
        const FramePtr& ref_frame,
        std::vector<cv::DMatch>& matches);

private:
    //! ros node handle for reading parameters
    ros::NodeHandle nh_;

    //! orb matcher parameters
    std::string method_;
    std::function<void(FramePtr, FramePtr, std::vector<cv::DMatch>&)> match_;


    //! opencv orb extractors
    cv::Ptr<cv::ORB> cv_orb_detector_;
    cv::Ptr<cv::ORB> cv_orb_descriptor_;
};

using ORBMatcherPtr = std::shared_ptr<ORBMatcher>;

} // namespace geometry

} // namespace orb_slam