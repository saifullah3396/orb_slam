/**
 * Implements the MapPoint class.
 */

#include <orb_slam/frame.h>
#include <orb_slam/map.h>
#include <orb_slam/map_point.h>
#include <orb_slam/key_frame.h>

namespace orb_slam
{

std::atomic_uint64_t MapPoint::global_id_;
geometry::ORBExtractorConstPtr MapPoint::orb_extractor_;
geometry::ORBMatcherConstPtr MapPoint::orb_matcher_;

MapPoint::MapPoint(
    const cv::Mat& world_pos,
    const KeyFramePtr& ref_key_frame,
    const MapPtr& map) :
    world_pos_(world_pos.clone()),
    ref_key_frame_(ref_key_frame),
    map_(map)
{
    view_vector_ = cv::Mat::zeros(3, 1, CV_32F); // initialize normal vector
    id_ = global_id_++; // set new id
}

MapPoint::~MapPoint()
{
}


std::map<KeyFramePtr, size_t> MapPoint::observations() {
    LOCK_OBSERVATIONS;
    return observations_;
}

const size_t MapPoint::nObservations() {
    LOCK_OBSERVATIONS;
    return observations_.size();
}

void MapPoint::addObservation(const KeyFramePtr& key_frame, const size_t idx) {
    LOCK_OBSERVATIONS;
    if (observations_.count(key_frame)) // if this key frame already exists
        return;
    observations_[key_frame] = idx;
}

void MapPoint::removeObservation(const KeyFramePtr& key_frame) {
    auto bad_point = false;
    { // shared
        LOCK_OBSERVATIONS;
        // if this key frame already exists
        if (observations_.count(key_frame)) {
            // get id of this point in key frame
            auto idx = observations_[key_frame];
            observations_.erase(key_frame); // remove the frame from this point

            if (nObservations() <= MIN_REQ_OBSERVATIONS) {
                bad_point = true;
            }
        }
    }

    if (bad_point) {
        removeFromMap();
    }
}

bool MapPoint::isObservedInKeyFrame(const KeyFramePtr& key_frame)
{
    LOCK_OBSERVATIONS;
    return observations_.count(key_frame) > 0;
}

int MapPoint::getIndexInKeyFrame(const KeyFramePtr& key_frame)
{
    LOCK_OBSERVATIONS;
    if (observations_.count(key_frame) > 0) {
        return observations_[key_frame];
    } else {
        return -1;
    }
}

void MapPoint::removeFromMap() {
    std::map<KeyFramePtr, size_t> observations;
    { // shared
        LOCK_OBSERVATIONS;
        LOCK_POS;
        bad_point_ = true;
        observations = observations_;
    observations_.clear();
    }

    for (const auto& obs: observations) {
        auto key_frame = obs.first;

        // remove point from the key frame
        key_frame->removeMapPointAt(obs.second);
    }

    // remove this point from the map
    map_->removeMapPoint(shared_from_this());
}

void MapPoint::computeBestDescriptor()
{
    // Retrieve all observed descriptors
    std::vector<cv::Mat> descriptors;
    std::map<KeyFramePtr, size_t> observations;
    { // shared
        LOCK_OBSERVATIONS;
        if (bad_point_ || observations_.empty()) // return if bad point since it has to be deleted
            return;
        observations = observations_; // get a local copy of observations
    }

    descriptors.reserve(observations.size());
    // get descriptors of matching points
    for (auto it = observations.begin(); it != observations.end(); ++it) {
        auto key_frame = it->first;
        if (!key_frame->isBad()) { // if key frame is not to be deleted
            descriptors.push_back(key_frame->descriptor(it->second));
        }
    }

    if (descriptors.empty())
        return;

    // compute distances between the descriptors
    const size_t n = descriptors.size();

    float distances[n][n];
    for (size_t i = 0; i < n; i++) {
        distances[i][i] = 0;
        for (size_t j = i + 1; j < n; j++) {
            int dist =
                geometry::descriptorDistance(descriptors[i], descriptors[j]);
            distances[i][j] = dist;
            distances[j][i] = dist;
        }
    }

    // take the descriptor with least median distance to the rest
    int best_median = INT_MAX;
    int best_idx = 0;
    for (size_t i = 0; i < n; i++) {
        std::vector<int> dists(distances[i], distances[i] + n);
        sort(dists.begin(),dists.end());
        int median = dists[0.5 * (n - 1)]; // find median

        if(median < best_median)
        {
            best_median = median;
            best_idx = i;
        }
    }

    { // shared
        LOCK_OBSERVATIONS;
        best_descriptor_ = descriptors[best_idx].clone();
    }
}

void MapPoint::updateNormalAndScale()
{
    std::map<KeyFramePtr, size_t> observations;
    KeyFramePtr ref_key_frame;
    cv::Mat pos;
    { // shared
        LOCK_OBSERVATIONS;
        LOCK_POS;
        if(bad_point_)
            return;

        if(observations_.empty())
            return;

        // get point state
        observations = observations_;
        ref_key_frame = ref_key_frame_;
        pos = world_pos_.clone();
    }

    cv::Mat normal = cv::Mat::zeros(3, 1, CV_32F);
    int n = 0;
    for (auto it = observations.begin(); it != observations.end(); it++) {
        const auto& key_frame = it->first;
        const auto& frame_pos = key_frame->getWorldPos();
        cv::Mat diff = world_pos_ - frame_pos;
        normal = normal + diff / cv::norm(diff);
        ++n;
    }

    cv::Mat p = world_pos_ - ref_key_frame_->getWorldPos();
    const float dist = cv::norm(p);
    const auto& undist_key_points = ref_key_frame_->frame()->featuresUndist();
    const int level = undist_key_points[observations[ref_key_frame_]].octave;
    const float scale_factor = orb_extractor_->scaleFactors()[level];
    const int n_levels = orb_extractor_->levels();

    { // shared
        LOCK_POS;
        // the maximum distance from which this point can be found again. The
        // scale_factor is from the pyramid level this point is found in the
        // image. scale_factor increases with levels [1, 1.2, 1.44, 1.728, ...]
        // Greater the pyramid level, farther the max scale distance.
        max_scale_distance_ = dist * scale_factor;

        // the minimum distance from which this point can be found again.
        min_scale_distance_ =
            max_scale_distance_ / orb_extractor_->scaleFactors()[n_levels - 1];
        view_vector_ = normal / n;
    }
}


int MapPoint::predictScale(const float& dist)
{
    float ratio;
    { // shared
        // don't change point position
        LOCK_POS;
        ratio = max_scale_distance_ / dist;
    }

    // find scale factor the same way orb slam original code does
    int scale = ceil(log(ratio) / orb_extractor_->logScaleFactor());
    const auto& levels = orb_extractor_->levels();
    if(scale < 0)
        scale = 0;
    else if(scale >= levels)
        scale = levels - 1;

    return scale;
}

} // namespace orb_slam