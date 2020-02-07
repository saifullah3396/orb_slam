/**
 * This file implements the ORBMatcher class.
 *
 * @author <A href="mailto:saifullah3396@gmail.com">Saifullah</A>
 */

#pragma once

#include <orb_slam/frame.h>

namespace orb_slam
{

namespace geometry
{

BruteForceWithRadiusMatcher::BruteForceWithRadiusMatcher(const ros::NodeHandle& nh) {
    std::string prefix = "orb_slam/brute_force_width_radius/";
    nh.getParam(
        prefix + "max_matching_pixel_dist", max_matching_pixel_dist_);
    max_matching_pixel_dist_sqrd_ =
        max_matching_pixel_dist_ * max_matching_pixel_dist_;
}

void BruteForceWithRadiusMatcher::match(
    const FramePtr& frame,
    const FramePtr& ref_frame,
    std::vector<cv::DMatch>& matches)
{
    // get undistorted key points
    const auto& n = frame->nFeaturesUndist();
    const auto& n_ref = ref_frame->nFeaturesUndist();

    // undistorted key points size == undistored descriptors size
    assert(
        n == frame->nDescriptorsUnDist() &&
        n_ref == ref_frame->nDescriptorsUnDist());

    const auto& key_points = frame->featuresUndist();
    const auto& ref_key_points = ref_frame->featuresUndist();
    const auto& descriptors = frame->descriptorsUndist();
    const auto& ref_descriptors = ref_frame->descriptorsUndist();
    for (int i = 0; i < n; i++) {
        const auto& kp = key_points[i];
        auto matched = false;
        double min_feature_dist = 99999999.0;
        int matched_id = 0;
        // finds closest of all the features in ref_frame that lies within
        // pixel radius of feature i
        for (int j = 0; j < n_ref; j++) {
            const auto& ref_kp = ref_key_points[j];
            auto dx = ref_kp.pt.x - kp.pt.x;
            auto dy = ref_kp.pt.y - kp.pt.y;
            auto sqrd_dist = dx * dx + dy * dy;
            if (sqrd_dist <= max_matching_pixel_dist_sqrd_) {
                // if points are within range, find descriptor distances
                cv::Mat diff;
                cv::absdiff(
                    descriptors.row(i), ref_descriptors.row(j), diff);
                double feature_dist = cv::sum(diff)[0] / descriptors.cols;
                if (feature_dist < min_feature_dist) {
                    // the two features i and j match
                    min_feature_dist = feature_dist;
                    matched_id = j;
                    matched = true;
                }
            }
        }
        if (matched)
            matches.push_back(
                cv::DMatch(
                    i, matched_id, static_cast<float>(min_feature_dist)));
    }
}

ORBMatcher::ORBMatcher(const ros::NodeHandle& nh): nh_(nh) {
    std::string prefix = "orb_slam/orb_matcher/";
    nh_.getParam(prefix + "method", method_);
    if (method_ == "bruteForce") {
        using namespace std::placeholders;
        auto matcher = BruteForceWithRadiusMatcher(nh_);
        match_ =
            std::bind(
                &BruteForceWithRadiusMatcher::match,
                &matcher,
                std::placeholders::_1,
                std::placeholders::_2,
                std::placeholders::_3);
    }
}

ORBMatcher::~ORBMatcher() {

}

void ORBMatcher::match(
    const FramePtr& frame,
    const FramePtr& ref_frame,
    std::vector<cv::DMatch>& matches)
{
    match_(frame, ref_frame, matches);
}

} // namespace geometry

} // namespace orb_slam