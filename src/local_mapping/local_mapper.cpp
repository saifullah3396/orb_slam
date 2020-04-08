/**
 * Implements the LocalMapper class.
 */

#include <orb_slam/map.h>
#include <orb_slam/frame.h>
#include <orb_slam/key_frame.h>
#include <orb_slam/local_mapping/local_mapper.h>
#include <orb_slam/local_mapping/local_bundle_adjuster.h>

namespace orb_slam
{

LocalMapper::LocalMapper(const MapPtr& map) : map_(map) {
    local_ba_ = LocalBundleAdjusterPtr(new LocalBundleAdjuster(map));
}

void LocalMapper::update()
{
    finished_ = false;

    while (true) {
        setBusy(true); // mapper gets busy at the start of loop

        // check if new key frames exist for processing
        if (newKeyFramesExist()) {
            // bow conversion and insertion in map
            processNewKeyFrame();

            // map points culling
            mapPointCulling();

            // triangulate new map points from covisible key frames
            createNewMapPoints();

            if (!newKeyFramesExist()) {
                // if no more key frames then also search neighbors
                searchInNeighbors();
            }

            abort_ba_ = false;

            if (!newKeyFramesExist() && !stopRequested()) {
                // perform local bundle adjustments if no more key frames
                // are available and if stop is not requested
                if (map_->nKeyFrames() > 2) // minimum 2 key frames for ba
                    local_ba_->solve(key_frame_in_process_, &abort_ba_);

                // check redundant local Keyframes
                keyFramesCulling();
            }
        } else if (stop()) { // stopped if required
            // wait if stopped and keep running idly
            while (stopped() && !finishRequested()) { // stopped but not finished
                usleep(3000); // idle loop
            }

            if (finishRequested()) // finish requested
                break;
        }

        resetIfRequested();

        setBusy(false); // mapper is no more busy

        if (finishRequested()) // finish requested
            break;
        usleep(3000);
    }

    setFinished();
}

bool LocalMapper::stop()
{
    LOCK_STOPPABLE;
    if(stop_requested_ && stoppable_) { // stop if possible and requested
        stopped_ = true;
        ROS_DEBUG_STREAM("Local mapper stopped.");
        return true;
    }

    return false;
}

void LocalMapper::processNewKeyFrame() {
    { // shared
        LOCK_NEW_KF_QUEUE;
        key_frame_in_process_ = new_key_frames_.front();
        new_key_frames_.pop();
    }

    key_frame_in_process_->frame()->computeBow();

    // associate MapPoints to the new keyframe and update normal and descriptor
    const auto map_points = key_frame_in_process_->obsMapPoints();
    for (int idx = 0; idx < map_points.size(); ++idx) {
        const auto& mp = map_points[idx];
        if (mp && !mp->isBad()) {
            // check if this map point is not already observed in the key frame
            // usually this should be true
            if (!mp->isObservedInKeyFrame(key_frame_in_process_)) {
                // add the key frame to map point in which it is observed
                // i corresponds to index of the key point in frame class
                mp->addObservation(key_frame_in_process_, idx);

                // compute map point best descriptor out of all observing key frames
                mp->computeBestDescriptor();

                // compute normal vector, and scale distance for map point
                mp->updateNormalAndScale();
            }
        }
    }
    // add all unmatched map points as invalid points which must be filtered out
    map_points_to_cull_.insert(
        map_points_to_cull_.begin(),
        key_frame_in_process_->frame()->unmatchedMapPoints().begin(),
        key_frame_in_process_->frame()->unmatchedMapPoints().end());

    // update the key frame covisibility graph
    key_frame_in_process_->updateConnections();

    // insert key frame to map
    map_->addKeyFrame(key_frame_in_process_);
}

void LocalMapper::mapPointCulling()
{
    // check all invalid map points and see if they can be validated,
    // otherwise remove them
    const auto& current_kf_id = key_frame_in_process_->id();

    const int min_required_observations = 3;
    auto it = map_points_to_cull_.begin();
    while (it != map_points_to_cull_.end()) {
        const auto& mp = *it;
        auto first_to_curr_observation =
            ((int) current_kf_id - (int) mp->refKeyFrame()->id());
        if (mp->isBad()) {
            it = map_points_to_cull_.erase(it); // remove invalid point
        } else if (mp->foundRatio() < 0.25f) {
            // must be tracked in 25% of frames
            mp->removeFromMap(); // set point as bad
            it = map_points_to_cull_.erase(it); // remove invalid point
        } else if (
                first_to_curr_observation >= 2 &&
                mp->nObservations() <= min_required_observations) {
            // if more than one key frame exist in map, this point must be seen
            // at least min_required_observations times, otherwise it is bad
            mp->removeFromMap(); // set point as bad
            it = map_points_to_cull_.erase(it); // remove invalid point
        } else if(first_to_curr_observation >= 3) {
            // remove it from culling cuz its a good point and we don't have to
            // cull it again, I guess.
            it = map_points_to_cull_.erase(it);
        } else {
            // move forward
            it++;
        }
    }
}

void LocalMapper::keyFramesCulling()
{
    // check redundant keyframes (only local keyframes)
    // a keyframe is considered redundant if the 90% of the map points it sees,
    // are seen in at least other 3 keyframes (in the same or finer scale)
    // we only consider close stereo points
    std::vector<KeyFramePtr> local_key_frames =
        key_frame_in_process_->getCovisibles();

    for (auto& cov_kf: local_key_frames) {
        if(cov_kf->id() == 0)
            continue;

        // map points observed in this key frame
        const auto& key_points = cov_kf->frame()->featuresUndist();
        const auto& depths = cov_kf->frame()->featureDepthsUndist();
        const auto map_points = cov_kf->obsMapPoints();

        const int th_obs = 3;
        int n_redundant_obs = 0;
        int n_map_points = 0;
        for (int idx = 0; idx < map_points.size(); ++idx) {
            const auto& mp = map_points[idx];
            if(mp && !mp->isBad()) {
                // if not monocular...
                {
                    float close_points_threshold = 1;
                    if (depths[idx] < 0 || depths[idx] > close_points_threshold)
                        continue;
                }

                n_map_points++;
                if (mp->nObservations() > th_obs) {
                    const int& cov_scale_level = key_points[idx].octave;
                    const auto observations = mp->observations();
                    int n_obs = 0;
                    for (const auto& obs: observations) {
                        const auto& kf = obs.first;
                        if (kf == cov_kf)
                            continue;
                        const int& scale_level =
                            kf->frame()->featuresUndist()[obs.second].octave;

                        if (scale_level <= cov_scale_level + 1)
                        {
                            n_obs++;
                            if (n_obs >= th_obs)
                                break;
                        }
                    }

                    if (n_obs >= th_obs)
                    {
                        n_redundant_obs++;
                    }
                }
            }
        }

        if (n_redundant_obs > 0.9 * n_map_points)
            cov_kf->removeFromMap();
    }
}

void LocalMapper::createNewMapPoints()
{
    // get the covisible key frames
    int n_neighbors = 10;
    const auto cov_key_frames =
        key_frame_in_process_->getBestCovisibles(n_neighbors);

    // search for matches between key frame in process and its covisibles using
    // the epipolar constraint
    const auto& w_t_c = key_frame_in_process_->getWorldPos();

    // find frame projection matrices
    const auto& frame = key_frame_in_process_->frame();
    const auto c_T_w = frame->worldInCameraT();
    const auto c_R_w = frame->worldInCameraR();
    const auto c_t_w = frame->worldInCamerat();
    const auto w_R_c = frame->cameraInWorldR();
    cv::Mat K = frame->camera()->intrinsicMatrix();
    K.convertTo(K, CV_64F); // convert to double
    const auto proj = K * c_T_w;

    // frame features
    const auto& key_points = frame->featuresUndist();
    const auto& depths = frame->featureDepthsUndist();

    // orb scale sigmas
    const auto& scale_factors = frame->orbExtractor()->scaleFactors();
    const auto& scale_sigmas = frame->orbExtractor()->scaleSigmas();
    const float ratio_factor = 1.5f * frame->orbExtractor()->scaleFactor();

    // start matching
    for (const auto& cov_kf : cov_key_frames) {
        // check if baseline between two poses is not small since epipolar
        // constraint would result in bad estimates in that case
        const float baseline = cv::norm(cov_kf->getWorldPos() - w_t_c);
        float min_base_line = 0.1; // @todo: define this somewhere else
        if (baseline < min_base_line)
            continue;


        // find matches
        std::vector<cv::DMatch> matches;
        key_frame_in_process_->match(cov_kf, matches);

        if (matches.empty()) {
            continue;
        }

        const auto& cov_frame = cov_kf->frame();
        const auto& cov_key_points = cov_frame->featuresUndist();
        const auto cov_T_w = cov_frame->worldInCameraT();
        const auto cov_R_w = cov_frame->worldInCameraR();
        const auto cov_t_w = cov_frame->worldInCamerat();
        const auto w_R_cov = cov_frame->cameraInWorldR();
        const auto w_t_cov = cov_frame->cameraInWorldt();
        cv::Mat cov_K = cov_frame->camera()->intrinsicMatrix();
        cov_K.convertTo(cov_K, CV_64F); // convert to double
        const auto cov_proj = cov_K * cov_T_w;
        const auto& cov_depths = cov_frame->featureDepthsUndist();

        cv::Mat point_3d;
        for (const auto& m : matches) {
            const auto& p = key_points[m.queryIdx];
            const auto& cov_p = key_points[m.trainIdx];

            // Check parallax between rays
            auto pn =
                frame->frameToCamera<float, float>(p.pt, 1.0);
            auto cov_pn =
                cov_frame->frameToCamera<float, float>(cov_p.pt, 1.0);

            cv::Mat ray = frame->cameraInWorldR() * cv::Mat(pn);
            cv::Mat cov_ray = cov_frame->cameraInWorldR() * cv::Mat(cov_pn);
            const float cos_ray_parallex =
                ray.dot(cov_ray) / (cv::norm(ray) * cv::norm(cov_ray));
            const float min_parallex = std::min(
                cos(2 * atan2(min_base_line / 2, depths[m.queryIdx])),
                cos(2 * atan2(min_base_line / 2, cov_depths[m.trainIdx])));
            if (
                cos_ray_parallex < min_parallex && cos_ray_parallex > 0 && cos_ray_parallex < 0.9998)
            {
                // linearly triangulate point using svd
                cv::Mat A(4, 4, CV_32F);
                A.row(0) = pn.x * c_T_w.row(2) - c_T_w.row(0);
                A.row(1) = pn.y * c_T_w.row(2) - c_T_w.row(1);
                A.row(2) = cov_pn.x * cov_T_w.row(2) - cov_T_w.row(0);
                A.row(3) = cov_pn.y * cov_T_w.row(2) - cov_T_w.row(1);

                cv::Mat w, u, vt;
                cv::SVD::compute(
                    A, w, u, vt, cv::SVD::MODIFY_A| cv::SVD::FULL_UV);
                point_3d = vt.row(3).t();
                if(point_3d.at<float>(3) == 0)
                    continue;
                // re-scale
                point_3d = point_3d.rowRange(0,3) / point_3d.at<float>(3);
            } else if (depths[m.queryIdx] > 0) { // depth is positive
                // transform key point to 3d world space from frame depth info
                auto point_in_cam =
                    frame->frameToCamera<float, float>(p.pt, depths[m.queryIdx]);
                point_3d = w_R_c * cv::Mat(point_in_cam) + w_t_c;
            } else if (cov_depths[m.trainIdx] > 0) { // depth is positive
                // transform key point to 3d world space from frame depth info
                auto point_in_cam =
                    cov_frame->frameToCamera<float, float>(cov_p.pt, cov_depths[m.queryIdx]);
                point_3d = w_R_cov * cv::Mat(point_in_cam) + w_t_cov;
            } else {
                continue; // no point triangulated
            }

            // check triangulated point is in front of camera 1
            auto reproj_p = cv::Point3f(cv::Mat(c_R_w * point_3d + c_t_w));
            if (reproj_p.z <= 0) // negative depth
                continue;

            // check triangulated point is in front of camera 2
            auto cov_reproj_p = cv::Point3f(cv::Mat(cov_R_w * point_3d + cov_t_w));
            if (cov_reproj_p.z <= 0) // behind the camera
                continue;

            // check reprojection error in first keyframe
            const float& var = scale_sigmas[p.octave];
            const auto reproj_img =
                frame->cameraToFrame<float, float>(reproj_p);
            auto error = cv::norm(reproj_img - p.pt);
            if (error > 5.992 * var)
                continue;

            // check reprojection error in second keyframe
            const float& cov_var = scale_sigmas[cov_p.octave];
            const auto cov_reproj_img =
                cov_frame->cameraToFrame<float, float>(cov_reproj_p);
            auto cov_error = cv::norm(cov_reproj_img - cov_p.pt);
            if (cov_error > 5.992 * var)
                continue;

            // check scale consistency
            cv::Mat normal = point_3d - w_t_c;
            float dist = cv::norm(normal);

            cv::Mat cov_normal = point_3d - w_t_cov;
            float cov_dist = cv::norm(cov_normal);

            if (dist == 0 || cov_dist == 0)
                continue;

            const float dist_ratio = cov_dist / dist;
            const float scale_ratio =
                scale_factors[p.octave] / scale_factors[cov_p.octave];

            // distance ratio should be similar to scale ratio?
            if (
                dist_ratio * ratio_factor < scale_ratio ||
                dist_ratio > scale_ratio * ratio_factor)
                continue;

            // traingulated point is good
            // create a map point given 2d-3d correspondence
            auto mp =
                MapPointPtr(
                    new MapPoint(point_3d, key_frame_in_process_, map_));

            // add the map point in local observation map of key frame
            // i corresponds to index of the map point here
            key_frame_in_process_->setMapPointAt(mp, m.queryIdx);
            cov_kf->setMapPointAt(mp, m.trainIdx);

            // add the key frame to map point in which it is observed
            // i corresponds to index of the key point in frame class
            mp->addObservation(key_frame_in_process_, m.queryIdx);
            mp->addObservation(cov_kf, m.trainIdx);

            // compute map point best descriptor out of all observing key frames
            mp->computeBestDescriptor();

            // compute normal vector, and scale distance for map point
            mp->updateNormalAndScale();

            // add map points to map and to reference map
            map_->addMapPoint(mp);
            map_->addRefMapPoint(mp);
            map_points_to_cull_.push_back(mp);
        }

        if (newKeyFramesExist()) { // if more frames exist only add one neighbor
            return;
        }
    }
}

void LocalMapper::searchInNeighbors()
{
    // Retrieve neighbor keyframes
    // get the covisible key frames
    int n_neighbors = 10;
    int n_neighbors_2 = 5;
    auto cov_key_frames =
        key_frame_in_process_->getBestCovisibles(n_neighbors);

    // try to add neighbors of current frame

    std::vector<KeyFramePtr> extended_key_frames;
    for (auto& cov_kf: cov_key_frames) {
        // if bad or already searched
        if(cov_kf->isBad() || cov_kf->searchedInMapWith(key_frame_in_process_->id()))
            continue;
        extended_key_frames.push_back(cov_kf);
        cov_kf->setSearchedInMapWith(key_frame_in_process_->id());

        // also add neighbors of neigbhor
        auto cov_key_frames_2 = cov_kf->getBestCovisibles(n_neighbors_2);
        for (auto& cov_kf2: cov_key_frames_2) {
            // if bad or already searched or is the same frame as parent
            if(
                cov_kf2->isBad() ||
                cov_kf2->searchedInMapWith(key_frame_in_process_->id()) ||
                cov_kf2->id() == key_frame_in_process_->id())
                continue;
            extended_key_frames.push_back(cov_kf2);
            cov_kf2->setSearchedInMapWith(key_frame_in_process_->id());
        }
    }


    // search matches by projection from current key frame in to target key frames
    auto map_points = key_frame_in_process_->obsMapPoints();
    for (const auto& kf : extended_key_frames) {
        key_frame_in_process_->frame()->orbMatcher()->fuse(kf, map_points);
    }

    // search matches by projection from target KFs in current KF
    std::vector<MapPointPtr> fuse_candidates;
    fuse_candidates.reserve(extended_key_frames.size() * map_points.size());
    auto key_frame_in_process_id = key_frame_in_process_->id();
    for (const auto& kf : extended_key_frames)
    {
        const auto map_points_kf = kf->obsMapPoints();
        for (const auto& mp: map_points_kf) {
            if (!mp) continue;
            if (mp->isBad() || mp->isFuseCandidateOf(key_frame_in_process_id))
                continue;
            mp->setFuseCandidateOf(key_frame_in_process_id);
            fuse_candidates.push_back(mp);
        }
    }
    key_frame_in_process_->frame()->orbMatcher()->fuse(key_frame_in_process_, fuse_candidates);

    // update points
    auto map_points_updated = key_frame_in_process_->obsMapPoints();
    for (const auto& mp: map_points_updated) {
        if(mp && !mp->isBad()) {
            mp->computeBestDescriptor();
            mp->updateNormalAndScale();
        }
    }

    // update connections in covisibility graph
    key_frame_in_process_->updateConnections();
}

} // namespace orb_slam
