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
    std::unique_lock<std::mutex> observations_lock(observations_mutex_);
    return observations_;
}

const size_t MapPoint::nObservations() {
    std::unique_lock<std::mutex> observations_lock(observations_mutex_);
    return observations_.size();
}

void MapPoint::addObservation(const KeyFramePtr& key_frame, const size_t idx) {
    std::unique_lock<std::mutex> observations_lock(observations_mutex_);
    if (observations_.count(key_frame)) // if this key frame already exists
        return;
    observations_[key_frame] = idx;
}

void MapPoint::removeObservation(const KeyFramePtr& key_frame) {
    { // shared
        std::unique_lock<std::mutex> observations_lock(observations_mutex_);
        // if this key frame already exists
        if (observations_.count(key_frame)) {
            // get id of this point in key frame
            auto idx = observations_[key_frame];
            observations_.erase(key_frame); // remove the frame from this point

            if (nObservations() <= MIN_REQ_OBSERVATIONS) {
                bad_point_ = true;
            }
        }
    }

    if (bad_point_) {
        removeFromMap();
    }
}

void MapPoint::removeFromMap() {
    std::unique_lock<std::mutex> observations_lock(observations_mutex_);
    std::unique_lock<std::mutex> pos_lock(pos_mutex_);
    auto observations = observations_;
    observations_.clear();

    for (auto it = observations.begin(), end = observations.end();
            it != end; it++)
    {
        auto key_frame = it->first;
        // remove point from the key frame
        key_frame->removeMapPointAt(it->second);
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
        std::unique_lock<std::mutex> observations_lock(observations_mutex_);
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
        std::unique_lock<std::mutex> observations_lock(observations_mutex_);
        best_descriptor_ = descriptors[best_idx].clone();
    }
}

} // namespace orb_slam