/**
 * This file implements the ORBMatcher class.
 *
 * @author <A href="mailto:saifullah3396@gmail.com">Saifullah</A>
 */

#include <memory>
#include <orb_slam/frame.h>
#include <orb_slam/key_frame.h>

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

void BruteForceWithProjectionMatcher::match(
    const FramePtr& frame,
    const FramePtr& ref_frame,
    std::vector<cv::DMatch>& matches)
{
    // find frame in reference frame...
    const cv::Mat ref_T_f =
        ref_frame->worldInCameraT() *
        frame->cameraInWorldT();
    // get the transform from last frame to this frame, don't copy
    const auto& ref_map_points = ref_frame->obsMapPoints();
    match(
        frame,
        ref_frame,
        ref_T_f,
        ref_map_points,
        matches);
}

void BruteForceWithProjectionMatcher::match(
    const FramePtr& frame,
    const FramePtr& ref_frame,
    const cv::Mat& ref_T_f,
    const std::vector<MapPointPtr>& ref_map_points,
    std::vector<cv::DMatch>& matches)
{
    const auto& ref_key_points = ref_frame->featuresUndist();
    const auto& key_points = frame->featuresUndist();
    const auto& descriptors = frame->descriptorsUndist();
    std::vector<int> matched(key_points.size(), -1);
    std::vector<int> feature_dists(key_points.size(), -1);

    std::vector<int> rot_hist[hist_length_];
    for (size_t i = 0; i < hist_length_; i++)
        rot_hist[i].reserve(500);
    const auto factor = 1.0f / hist_length_;

    // camera z is pointing to the front
    //    z
    //   /
    //  /
    // ----- x
    // |
    // | y
    //
    const auto z = ref_T_f.at<float>(2, 3);
    float min_base_line = 0.1;
    const bool frame_in_front = z > min_base_line; // frame is in front of reference frame
    const bool frame_behind = -z > min_base_line; // frame is behind reference frame

    for (
        size_t ref_idx = 0; ref_idx < ref_frame->nFeaturesUndist(); ++ref_idx)
    {
        const auto& mp = ref_map_points[ref_idx];
        if (mp) { // every feature may not have 3d correspondence
            cv::Mat point_3d_world = mp->worldPos();
            // convert from world to current camera frame
            auto point_3d_cam = frame->worldToCamera<float>(point_3d_world);
            if (point_3d_cam.z < 0) // if depth is negative
                continue;
            auto point_2d_frame =
                frame->cameraToFrame<float, float>(point_3d_cam);
            // ignore if point is outside image bounds
            if (!frame->pointWithinBounds(point_2d_frame))
                continue;

            // scale level at which point is found in reference frame
            auto feature_level_ref = ref_key_points[ref_idx].octave;

            // now create a window around the point in the level this feature
            // was found
            auto radius_scaled =
                radius_ *
                frame->orbExtractor()->scaleFactors()[feature_level_ref];
            std::vector<size_t> close_points;
            if (frame_in_front) {
                // the frame if in front of the reference frame, then the image
                // will be bigger in size. That means that we will need to
                // downscale it at least more than the scale (feature_level_ref)
                // it was found in previous frame. For example... If a point is
                // found in scale [1] -> 1.2 in first frame, then in the next
                // frame, the same point would be found bigger (we're closer to
                // it). So, we'd need to downsample the new bigger point by at
                // least a factor of 1.2 to get the same point in the next
                // frame.
            if (!frame->getFeaturesAroundPoint(
                point_2d_frame,
                radius_scaled,
                        feature_level_ref,
                        close_points))
                { // if no points are found in region
                    continue;
                }
            } else if (frame_behind) {
                if (!frame->getFeaturesAroundPoint(
                        point_2d_frame,
                        radius_scaled,
                        0,
                        feature_level_ref,
                        close_points))
                { // if no points are found in region
                    continue;
                }
            } else {
                if (!frame->getFeaturesAroundPoint(
                    point_2d_frame,
                    radius_scaled,
                feature_level_ref - 1,
                feature_level_ref + 1,
                    close_points))
                { // if no points are found in region
                    continue;
                };
            }
            const auto& desc = mp->bestDescriptor();
            // finds closest of all the features in ref_frame that lies within
            // pixel radius of feature i
            int min_feature_dist = 256;
            int matched_id = -1;
            for (const auto& idx: close_points) {
                if (matched[idx] > 0) // matched map point already assigned
                    continue;
                // find distance between descriptors of this point and all the
                // close points
                const auto dist =
                    geometry::descriptorDistance(desc, descriptors.row(idx));
                if (dist <= min_feature_dist) {
                    min_feature_dist = dist;
                    matched_id = idx;
                }
            }

            if (min_feature_dist <= high_threshold_) {
                matched[matched_id] = ref_idx;
                feature_dists[matched_id] = min_feature_dist;
                if(check_orientation_) {
                    float rot_diff =
                        ref_key_points[ref_idx].angle -
                        key_points[matched_id].angle;
                    if (rot_diff < 0.0)
                        rot_diff += 360.0f;
                    // add rot_diff to bin
                    int bin = round(rot_diff * factor);
                    if (bin == hist_length_)
                        bin = 0;
                    assert(bin >= 0 && bin < hist_length_);
                    // add rotation difference to histogram
                    rot_hist[bin].push_back(matched_id);
                }
            }
        }
    }

    // apply rotation constraint
    if(check_orientation_)
        applyRotationConstraint(rot_hist, matched, hist_length_);

    // create matches
    createMatches(matched, feature_dists, matches);
}
    {
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

    // create matches
    for (size_t i = 0; i < matched.size(); ++i) {
        if (matched[i]) // matches from frame to reference frame
            matches.push_back(
                cv::DMatch(
                    i, matched[i], static_cast<float>(feature_dists[i])));
    }
}

void BowOrbMatcher::match(
    const FramePtr& frame,
    const FramePtr& ref_frame,
    std::vector<cv::DMatch>& matches)
{
    std::vector<int> rot_hist[hist_length_];
    for (size_t i = 0; i < hist_length_; i++)
        rot_hist[i].reserve(500);
    const auto factor = 1.0f / hist_length_;

    const auto& ref_key_points = ref_frame->featuresUndist();
    const auto& key_points = frame->featuresUndist();
    std::vector<int> matched(key_points.size(), -1);
    std::vector<int> feature_dists(key_points.size(), -1);

    const auto& ref_descs = ref_frame->descriptorsUndist();
    const auto& descs = frame->descriptorsUndist();
    const auto& ref_features = ref_frame->bowFeatures();
    const auto& features = frame->bowFeatures();

    auto ref_f = ref_features.begin();
    auto f = features.begin();
    while (ref_f != ref_features.end() && f != features.end()) {
        // if both features are in the same node level, meaning if they
        // are in the same histogram bin of bag of words
        if (ref_f->first == f->first) {
            // get all the features from both frames and match
            const auto& idxs = f->second;
            const auto& ref_idxs = ref_f->second;
            for (const auto& ref_idx: ref_idxs) {
                // if the feature at this index is bad for any reason
                //if (ref_frame->isBadFeature(ref_idx)) continue;
                const auto& ref_desc = ref_descs.row(ref_idx);

                // finds closest of all the features in ref_frame that lies within
                // pixel radius of feature i
                int min_feature_dist = 256;
                int min_feature_dist_2 = 256; // second minimum
                int matched_id = -1;
                for (const auto& idx: idxs) {
                    if (matched[idx] > 0) // match already assigned
                        continue;
                    // find distance between descriptors of this point and all the
                    // close points
                    const auto& desc = descs.row(idx);
                    const auto dist =
                        geometry::descriptorDistance(ref_desc, desc);
                    if (dist <= min_feature_dist) {
                        min_feature_dist = dist;
                        matched_id = idx;
                    } else if (dist < min_feature_dist_2) {
                        min_feature_dist_2 = dist;
                    }
                }

                if (min_feature_dist <= low_threshold_ &&
                    // not sure what nn ratio is
                    static_cast<float>(min_feature_dist) <
                    nn_ratio_ * static_cast<float>(min_feature_dist_2))
                {
                    matched[matched_id] = ref_idx;
                    feature_dists[matched_id] = min_feature_dist;
                    if(check_orientation_) {
                        float rot_diff =
                            ref_key_points[ref_idx].angle -
                            key_points[matched_id].angle;
                        if (rot_diff < 0.0)
                            rot_diff += 360.0f;
                        // add rot_diff to bin
                        int bin = round(rot_diff * factor);
                        if (bin == hist_length_)
                            bin = 0;
                        assert(bin >= 0 && bin < hist_length_);
                        // add rotation difference to histogram
                        rot_hist[bin].push_back(matched_id);
                    }
                }
            }

            f++; // move forward
            ref_f++; // move forward
        } else if(ref_f->first < f->first) { // not sure whats happening here
            ref_f = ref_features.lower_bound(f->first);
        } else {
            f = features.lower_bound(ref_f->first);
        }
    }

    // apply rotation consistency
    if(check_orientation_)
    {
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

    // create matches
    for (size_t i = 0; i < matched.size(); ++i) {
        if (matched[i] > 0) {// matches from frame to reference frame
            matches.push_back(
                cv::DMatch(
                    i, matched[i], static_cast<float>(feature_dists[i])));
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
    matcher_.resize(MATCHER_TYPES);
    matcher_[BF_WITH_RADIUS] =
        std::shared_ptr<MatcherBase>(new BruteForceWithRadiusMatcher(nh_));
    matcher_[BF_WITH_PROJ] =
        std::shared_ptr<MatcherBase>(new BruteForceWithProjectionMatcher(nh_));
    matcher_[BOW_ORB] =
        std::shared_ptr<MatcherBase>(new BowOrbMatcher(nh_));
    matcher_[CV_ORB] =
        std::shared_ptr<MatcherBase>(new CVORBMatcher(nh_));
}

ORBMatcher::~ORBMatcher() {

}

void ORBMatcher::match(
    const FramePtr& frame,
    const FramePtr& ref_frame,
    std::vector<cv::DMatch>& matches,
    const OrbMatcherTypes type,
    const bool filter_matches) const
{
    if (type == BF_WITH_RADIUS) {
        static_pointer_cast<BruteForceWithRadiusMatcher>(matcher_[type])->
            match(
                frame->featuresUndist(),
                ref_frame->featuresUndist(),
                frame->descriptorsUndist(),
                ref_frame->descriptorsUndist(),
                matches);
    } else if (type == BF_WITH_PROJ) {
        static_pointer_cast<BruteForceWithProjectionMatcher>(matcher_[type])->
            match(frame, ref_frame, matches);
    } else if (type == BOW_ORB) {
        static_pointer_cast<BowOrbMatcher>(matcher_[type])->
            match(frame, ref_frame, matches);
    } else if (type == CV_ORB) {
        static_pointer_cast<CVORBMatcher>(matcher_[type])->
            match(
                frame->descriptorsUndist(),
                ref_frame->descriptorsUndist(),
                matches);
    }
    if (filter_matches)
        filterMatches(frame->descriptorsUndist(), matches);
}

void ORBMatcher::match(
    const std::vector<cv::KeyPoint>& key_points,
    const std::vector<cv::KeyPoint>& ref_key_points,
    const cv::Mat& descriptors,
    const cv::Mat& ref_descriptors,
    std::vector<cv::DMatch>& matches,
    bool filter_matches) const
{
    static_pointer_cast<BruteForceWithRadiusMatcher>(matcher_[BF_WITH_RADIUS])->
        match(
            key_points,
            ref_key_points,
            descriptors,
            ref_descriptors,
            matches);
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
    static_pointer_cast<CVORBMatcher>(matcher_[CV_ORB])->
        match(
            descriptors,
            ref_descriptors,
            matches);
    if (filter_matches) {
        filterMatches(descriptors, matches);
    }
}

void ORBMatcher::matchByBowFeatures(
    const FramePtr& frame,
    const FramePtr& ref_frame,
    std::vector<cv::DMatch>& matches,
    const bool check_orientation,
    const float nn_ratio,
    const bool filter_matches) const
{
    const auto& matcher = static_pointer_cast<BowOrbMatcher>(matcher_[BOW_ORB]);
    matcher->check_orientation_ = check_orientation;
    matcher->nn_ratio_ = nn_ratio;
    matcher->match(frame, ref_frame, matches);
    if (filter_matches)
        filterMatches(frame->descriptorsUndist(), matches);
}

void ORBMatcher::matchByProjection(
    const FramePtr& frame,
    const FramePtr& ref_frame,
    std::vector<cv::DMatch>& matches,
    const bool check_orientation,
    const float radius,
    const bool filter_matches) const
{
    const auto& matcher =
        static_pointer_cast<BruteForceWithProjectionMatcher>(
            matcher_[BF_WITH_PROJ]);
    matcher->check_orientation_ = check_orientation;
    matcher->radius_ = radius;
    matcher->match(frame, ref_frame, matches);
    if (filter_matches)
        filterMatches(frame->descriptorsUndist(), matches);
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