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

} // namespace orb_slam