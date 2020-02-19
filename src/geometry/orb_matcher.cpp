/**
 * This file implements the ORBMatcher class.
 *
 * @author <A href="mailto:saifullah3396@gmail.com">Saifullah</A>
 */

#include <orb_slam/frame.h>

namespace orb_slam
{

namespace geometry
{

BruteForceWithRadiusMatcher::BruteForceWithRadiusMatcher(const ros::NodeHandle& nh) {
    std::string prefix = "orb_slam/brute_force_with_radius/";
    nh.getParam(
        prefix + "max_matching_pixel_dist", max_matching_pixel_dist_);
    max_matching_pixel_dist_sqrd_ =
        max_matching_pixel_dist_ * max_matching_pixel_dist_;
}

void BruteForceWithRadiusMatcher::match(
    const std::vector<cv::KeyPoint>& key_points,
    const std::vector<cv::KeyPoint>& ref_key_points,
    const cv::Mat& descriptors,
    const cv::Mat& ref_descriptors,
    std::vector<cv::DMatch>& matches)
{
    // get undistorted key points
    const auto& n = key_points.size(); // frame->nFeaturesUndist();
    const auto& n_ref = ref_key_points.size(); // ref_frame->nFeaturesUndist();

    // undistorted key points size == undistored descriptors size
    assert(
        n == descriptors.rows &&
        n_ref == ref_descriptors.rows);

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

        if (matched) {
            matches.push_back(
                cv::DMatch(
                    i, matched_id, static_cast<float>(min_feature_dist)));
        }
    }
}

CVORBMatcher::CVORBMatcher(const ros::NodeHandle& nh) {
    std::string prefix = "orb_slam/cv_orb_matcher/";
    std::string matcher_type;
    nh.getParam(
        prefix + "type", matcher_type);
    matcher_ = cv::DescriptorMatcher::create(matcher_type);
}

void CVORBMatcher::match(
    const cv::Mat& descriptors,
    const cv::Mat& ref_descriptors,
    std::vector<cv::DMatch>& matches)
{
    matcher_->match(descriptors, ref_descriptors, matches);
}

ORBMatcher::ORBMatcher(const ros::NodeHandle& nh): nh_(nh) {
    std::string prefix = "orb_slam/orb_matcher/";
    nh_.getParam(prefix + "method", method_);
    if (method_ == "cv_orb_matcher") {
        using namespace std::placeholders;
        matcher_ = std::shared_ptr<MatcherBase>(new CVORBMatcher(nh_));
        matcher_2 =
            std::bind(
                &CVORBMatcher::match,
                std::static_pointer_cast<CVORBMatcher>(matcher_),
                std::placeholders::_1,
                std::placeholders::_2,
                std::placeholders::_3);
    } else if (method_ == "brute_force_with_radius") {
        using namespace std::placeholders;
        matcher_ =
            std::shared_ptr<MatcherBase>(new BruteForceWithRadiusMatcher(nh_));
        matcher_1 =
            std::bind(
                &BruteForceWithRadiusMatcher::match,
                std::static_pointer_cast<BruteForceWithRadiusMatcher>(matcher_),
                std::placeholders::_1,
                std::placeholders::_2,
                std::placeholders::_3,
                std::placeholders::_4,
                std::placeholders::_5);
    }

    if (matcher_1 == nullptr && matcher_2 == nullptr) {
        throw std::runtime_error(
            "Please set an orb matcher in cfg/orb_matcher.yaml.");
    }
}

ORBMatcher::~ORBMatcher() {

}

void ORBMatcher::match(
    const FramePtr& frame,
    const FramePtr& ref_frame,
    std::vector<cv::DMatch>& matches,
    bool filter_matches) const
{
    if (matcher_1 != nullptr) {
    match(
        frame->featuresUndist(),
        ref_frame->featuresUndist(),
        frame->descriptorsUndist(),
        ref_frame->descriptorsUndist(),
        matches,
        filter_matches);
    } else if (matcher_2 != nullptr) {
        match(
            frame->descriptorsUndist(),
            ref_frame->descriptorsUndist(),
            matches,
            filter_matches);
}
}

void ORBMatcher::match(
    const std::vector<cv::KeyPoint>& key_points,
    const std::vector<cv::KeyPoint>& ref_key_points,
    const cv::Mat& descriptors,
    const cv::Mat& ref_descriptors,
    std::vector<cv::DMatch>& matches,
    bool filter_matches) const
{
    if (matcher_1 != nullptr) {
        matcher_1(key_points, ref_key_points, descriptors, ref_descriptors, matches);
    }
    if (filter_matches) {
        filterMatches(descriptors, matches);
    }
}

void ORBMatcher::match(
    const cv::Mat& descriptors,
    const cv::Mat& ref_descriptors,
    std::vector<cv::DMatch>& matches,
    bool filter_matches) const
{
    if (matcher_2 != nullptr) {
        matcher_2(descriptors, ref_descriptors, matches);
    }
    if (filter_matches) {
        filterMatches(descriptors, matches);
    }
}

void ORBMatcher::filterMatches(
    const cv::Mat& descriptors,
    std::vector<cv::DMatch>& matches) const
{
    // filter out based on distance
    auto min_max =
        minmax_element(
            matches.begin(), matches.end(),
            [](const cv::DMatch &m1, const cv::DMatch &m2)
            { return m1.distance < m2.distance; });
    double min_dist = min_max.first->distance;
    double max_dist = min_max.second->distance;

    std::vector<cv::DMatch> good_matches;
    for (int i = 0; i < descriptors.rows; i++) {
        if (matches[i].distance <= std::max(2 * min_dist, 30.0)) {
            good_matches.push_back(matches[i]);
        }
    }

    sort(good_matches.begin(), good_matches.end(),
        [](const cv::DMatch &m1, const cv::DMatch &m2) {
            return m1.trainIdx < m2.trainIdx;
        });

    std::vector<cv::DMatch> removed_duplicates;
    if (!good_matches.empty())
        removed_duplicates.push_back(matches[0]);
    for (int i = 1; i < good_matches.size(); i++) {
        if (good_matches[i].trainIdx != good_matches[i - 1].trainIdx) {
            removed_duplicates.push_back(good_matches[i]);
        }
    }
    matches = removed_duplicates;
}

} // namespace geometry

} // namespace orb_slam