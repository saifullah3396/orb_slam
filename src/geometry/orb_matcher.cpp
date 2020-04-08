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
    const KeyFramePtr& ref_key_frame,
    std::vector<cv::DMatch>& matches)
{
    // find frame in reference frame...
    const cv::Mat ref_T_f =
        ref_key_frame->worldInCameraT() *
        frame->cameraInWorldT();

    // get the transform from last frame to this frame, copy if thread safe
    const auto ref_map_points = ref_key_frame->obsMapPoints();
    match(
        frame,
        ref_key_frame->frame(),
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

void BruteForceWithProjectionMatcher::match(
    const FramePtr& frame,
    const std::vector<MapPointPtr>& points_3d,
    std::vector<cv::DMatch>& matches)
    {
    const auto& key_points = frame->featuresUndist();
    const auto& descriptors = frame->descriptorsUndist();
    std::vector<int> matched(key_points.size(), -1);
    std::vector<int> feature_dists(key_points.size(), -1);

    // get the transform from last frame to this frame
    for (
        size_t ref_idx = 0; ref_idx < points_3d.size(); ++ref_idx)
    {
        const auto& mp = points_3d[ref_idx];
        if (mp) { // may be not every feature has a 3d correspondence
            if (compute_track_info_) {
                TrackProperties track_results;
                mp->resetTrackProperties();
                if (frame->isInCameraView(mp, track_results, 0.5)) {
                    mp->setTrackProperties(track_results);
                }
            }

            const auto& props = mp->trackProperties();
            if (!props.in_view_ || mp->isBad()) { // not in view or bad
                continue;
                }

            // scale level at which point is found in reference frame
            auto feature_level_ref = props.pred_scale_level_;

            // now create a window around the point in the level this feature
            // was found
            auto radius_scaled =
                radius_ * radiusByViewCosine(props.view_cosine_) *
                frame->orbExtractor()->scaleFactors()[feature_level_ref];
            std::vector<size_t> close_points;
            if (!frame->getFeaturesAroundPoint(
                props.proj_xy_,
                radius_scaled,
                feature_level_ref - 1,
                feature_level_ref,
                close_points))
            { // if no points are found
                continue;
            }

            const auto& desc = mp->bestDescriptor();
            // finds closest of all the features in ref_frame that lies within
            // pixel radius of feature i
            int min_feature_dist = 256;
            int matched_id = -1;
            int min_2_feature_dist = 256;
            int matched_id_2 = -1;
            for (const auto& idx: close_points) {
                if (matched[idx] > 0) // matched map point already assigned
                    continue;
                // find distance between descriptors of this point and all the
                // close points
                const auto dist =
                    geometry::descriptorDistance(desc, descriptors.row(idx));
                if (dist < min_feature_dist) {
                    // assign second best
                    min_2_feature_dist = min_feature_dist;
                    matched_id_2 = matched_id;

                    // assign best
                    min_feature_dist = dist;
                    matched_id = idx;
                } else if (dist < min_2_feature_dist) {
                    // assign second best
                    min_2_feature_dist = dist;
                    matched_id_2 = idx;
        }

                // apply ratio to second match
                // (only if best and second are in the same scale level)
                if(min_feature_dist <= high_threshold_) {
                    if(
                        key_points[matched_id].octave ==
                        key_points[matched_id_2].octave &&
                        min_feature_dist > nn_ratio_ * min_2_feature_dist)
                    {
                        continue;
    }
                    matched[matched_id] = ref_idx;
                    feature_dists[matched_id] = min_feature_dist;
                }
            }
        }
    }

    // create matches
    createMatches(matched, feature_dists, matches);
    }

void BowOrbMatcher::match(
    const FramePtr& frame,
    const KeyFramePtr& ref_key_frame,
    std::vector<cv::DMatch>& matches)
{
    // get the transform from last frame to this frame, don't copy
    const auto& map_points = frame->obsMapPoints();
    const auto ref_map_points = ref_key_frame->obsMapPoints();
    match(
        frame,
        ref_key_frame->frame(),
        map_points,
        ref_map_points,
        matches);
}

void BowOrbMatcher::match(
    const KeyFramePtr& key_frame,
    const KeyFramePtr& ref_key_frame,
    std::vector<cv::DMatch>& matches)
{
    // get the transform from last frame to this frame, copy if thread safe
    const auto map_points = key_frame->obsMapPoints();
    const auto ref_map_points = ref_key_frame->obsMapPoints();
    match(
        key_frame->frame(),
        ref_key_frame->frame(),
        map_points,
        ref_map_points,
        matches);
}

void BowOrbMatcher::match(
    const FramePtr& frame,
    const FramePtr& ref_frame,
    const std::vector<MapPointPtr>& map_points,
    const std::vector<MapPointPtr>& ref_map_points,
    std::vector<cv::DMatch>& matches)
{
    std::vector<int> rot_hist[hist_length_];
    for (size_t i = 0; i < hist_length_; i++)
        rot_hist[i].reserve(500);
    const auto factor = 1.0f / hist_length_;

    const auto& ref_key_points = ref_frame->featuresUndist();
    const auto& key_points = frame->featuresUndist();
    // matched is the map from key points -> reference key points
    // matched[key_point_i] = ref_key_point_k
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
                // first see if this feature has a map point associated with it
                // or not. If it is, see if it is not a bad point. Since at the
                // end we need to attach matched map points from reference frame
                // to frame
                if (
                        !ref_map_points[ref_idx] ||
                        ref_map_points[ref_idx]->isBad())
                    continue;

                // get reference key point descriptor
                const auto& ref_desc = ref_descs.row(ref_idx);

                // finds closest of all the features in ref_frame that lies within
                // pixel radius of feature i
                int min_feature_dist = 256;
                int min_feature_dist_2 = 256; // second minimum
                int matched_idx = -1;
                for (const auto& idx: idxs) {
                    if (matched[idx] > 0) // match already assigned
                        continue;
                    // find distance between descriptors of this point and all the
                    // close points
                    const auto& desc = descs.row(idx);

                    // find descriptor distance
                    const auto dist =
                        geometry::descriptorDistance(ref_desc, desc);
                    if (dist <= min_feature_dist) {
                        min_feature_dist = dist;
                        // matched id is the match found in idxs for ref_idx
                        matched_idx = idx;
                    } else if (dist < min_feature_dist_2) {
                        min_feature_dist_2 = dist;
                    }
                }

                // if minimum dist is within lower threshold annd nn ratio
                // is satisfied
                if (min_feature_dist <= low_threshold_ &&
                    // not sure what nn ratio is
                    static_cast<float>(min_feature_dist) <
                    nn_ratio_ * static_cast<float>(min_feature_dist_2))
                {
                    // set matching from frame point at idx to reference point at ref_idx
                    matched[matched_idx] = ref_idx;
                    feature_dists[matched_idx] = min_feature_dist;
                    if(check_orientation_) {
                        float rot_diff =
                            ref_key_points[ref_idx].angle -
                            key_points[matched_idx].angle;
                        if (rot_diff < 0.0)
                            rot_diff += 360.0f;
                        // add rot_diff to bin
                        int bin = round(rot_diff * factor);
                        if (bin == hist_length_)
                            bin = 0;
                        assert(bin >= 0 && bin < hist_length_);
                        // add rotation difference to histogram
                        rot_hist[bin].push_back(matched_idx);
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

    if(check_orientation_)
        applyRotationConstraint(rot_hist, matched, hist_length_);

    // create matches
    createMatches(matched, feature_dists, matches);
}

void EpipolarConstraintWithBowMatcher::match(
    const KeyFramePtr& key_frame,
    const KeyFramePtr& ref_key_frame,
    std::vector<cv::DMatch>& matches)
    {
    const auto map_points = key_frame->obsMapPoints();
    const auto ref_map_points = ref_key_frame->obsMapPoints();

    // compute the fundamental matrix from frame to ref_frame
    cv::Mat fundamental_mat;
    key_frame->computeFundamentalMat(
        fundamental_mat, ref_key_frame);
    match(
        key_frame->frame(),
        ref_key_frame->frame(),
        fundamental_mat,
        map_points,
        ref_map_points,
        matches);
}

void EpipolarConstraintWithBowMatcher::match(
    const FramePtr& frame,
    const FramePtr& ref_frame,
    const cv::Mat& fundamental_mat,
    const std::vector<MapPointPtr>& map_points,
    const std::vector<MapPointPtr>& ref_map_points,
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
            for (const auto& idx: idxs) {
                // if a map point is already assigned
                if (map_points[idx]) {
                    continue;
                }

                const auto& desc = descs.row(idx);

                // finds closest of all the features in ref_frame that lies within
                // pixel radius of feature i
                int min_feature_dist = low_threshold_;
                int matched_id = -1;
                for (const auto& ref_idx: ref_idxs) {
                    if (matched[ref_idx] > 0 || ref_map_points[ref_idx]) {
                        // match already assigned or a map point is already
                        // found for this index in reference
                        continue;
                    }

                    // find distance between descriptors of this point and all the
                    // close points
                    const auto& ref_desc = ref_descs.row(idx);
                    const auto dist =
                        geometry::descriptorDistance(ref_desc, desc);
                    if (dist > low_threshold_ || dist > min_feature_dist) {
                        continue;
                    }

                    if (check_epipolar_dist(
                        key_points[idx],
                        ref_key_points[ref_idx],
                        fundamental_mat,
                        ref_frame))
                    {
                        matched_id = ref_idx;
                        min_feature_dist = dist;
                    }
                }

                if (matched_id >= 0) {
                    matched[idx] = matched_id;
                    feature_dists[idx] = min_feature_dist;
                    if(check_orientation_) {
                        float rot_diff =
                            key_points[idx].angle -
                            ref_key_points[matched_id].angle;
                        if (rot_diff < 0.0)
                            rot_diff += 360.0f;
                        // add rot_diff to bin
                        int bin = round(rot_diff * factor);
                        if (bin == hist_length_)
                            bin = 0;
                        assert(bin >= 0 && bin < hist_length_);
                        // add rotation difference to histogram
                        rot_hist[bin].push_back(idx);
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

    // apply rotation constraint
    if(check_orientation_)
        applyRotationConstraint(rot_hist, matched, hist_length_);

    // create matches
    createMatches(matched, feature_dists, matches);
    }

bool EpipolarConstraintWithBowMatcher::check_epipolar_dist(
        const cv::KeyPoint& kp1,
        const cv::KeyPoint& kp2,
        const cv::Mat& f_mat,
        const FramePtr& ref_frame) const
{
    const auto& p1 = kp1.pt;
    const auto& p2 = kp2.pt;
    // epipolar line in second image l = x1'f_mat = [a b c]
    const float a =
        p1.x * f_mat.at<float>(0, 0) +
        p1.y * f_mat.at<float>(1, 0) +
        f_mat.at<float>(2, 0);
    const float b =
        p1.x * f_mat.at<float>(0, 1) +
        p1.y * f_mat.at<float>(1, 1) +
        f_mat.at<float>(2, 1);
    const float c =
        p1.x * f_mat.at<float>(0, 2) +
        p1.y * f_mat.at<float>(1, 2) +
        f_mat.at<float>(2, 2);

    const float num = a * p2.x + b * p2.y + c;
    const float den = a * a + b * b;

    if(den==0)
        return false;

    // perpendicular distance point to line
    const float dsqr = num*num/den;

    return // see if distance is within threshold
        dsqr < 3.84 * ref_frame->orbExtractor()->scaleSigmas()[kp2.octave];
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
    matcher_[EPIPOLAR_CONSTRAINT] =
        std::shared_ptr<MatcherBase>(new EpipolarConstraintWithBowMatcher(nh_));
    matcher_[CV_ORB] =
        std::shared_ptr<MatcherBase>(new CVORBMatcher(nh_));
}

ORBMatcher::~ORBMatcher() {

}

void ORBMatcher::match(
    const std::vector<cv::KeyPoint>& key_points,
    const std::vector<cv::KeyPoint>& ref_key_points,
    const cv::Mat& descriptors,
    const cv::Mat& ref_descriptors,
    std::vector<cv::DMatch>& matches,
    bool filter_matches) const
{
    std::static_pointer_cast<BruteForceWithRadiusMatcher>(matcher_[BF_WITH_RADIUS])->
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
    std::static_pointer_cast<CVORBMatcher>(matcher_[CV_ORB])->
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
    const KeyFramePtr& ref_key_frame,
    std::vector<cv::DMatch>& matches,
    const bool check_orientation,
    const float nn_ratio) const
{
    const auto& matcher = std::static_pointer_cast<BowOrbMatcher>(matcher_[BOW_ORB]);
    matcher->check_orientation_ = check_orientation;
    matcher->nn_ratio_ = nn_ratio;
    matcher->match(frame, ref_key_frame, matches);
}

void ORBMatcher::matchByProjection(
    const FramePtr& frame,
    const FramePtr& ref_frame,
    std::vector<cv::DMatch>& matches,
    const bool check_orientation,
    const float radius) const
{
    const auto& matcher =
        std::static_pointer_cast<BruteForceWithProjectionMatcher>(
            matcher_[BF_WITH_PROJ]);
    matcher->check_orientation_ = check_orientation;
    matcher->radius_ = radius;
    matcher->match(frame, ref_frame, matches);
}

void ORBMatcher::matchByProjection(
    const FramePtr& frame,
    const std::vector<MapPointPtr>& points_3d,
    std::vector<cv::DMatch>& matches,
    const bool compute_track_info,
    const float nn_ratio,
    const float radius) const
{
    const auto& matcher =
        std::static_pointer_cast<BruteForceWithProjectionMatcher>(
            matcher_[BF_WITH_PROJ]);
    matcher->nn_ratio_ = nn_ratio;
    matcher->radius_ = radius;
    matcher->compute_track_info_ = compute_track_info;
    matcher->match(frame, points_3d, matches);
}

void ORBMatcher::matchByEpipolarConstraint(
    const KeyFramePtr& key_frame,
    const KeyFramePtr& ref_key_frame,
    std::vector<cv::DMatch>& matches,
    const bool check_orientation) const
{
    const auto& matcher =
        std::static_pointer_cast<EpipolarConstraintWithBowMatcher>(
            matcher_[EPIPOLAR_CONSTRAINT]);
    matcher->check_orientation_ = check_orientation;
    matcher->match(key_frame, ref_key_frame, matches);
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