/**
 * Implements the KeyFrame class.
 */

#include <memory>
#include <mutex>
#include <orb_slam/frame.h>
#include <orb_slam/key_frame.h>
#include <orb_slam/map_point.h>
#include <orb_slam/geometry/utils.h>

namespace orb_slam
{

long unsigned int KeyFrame::global_id_;

KeyFrame::KeyFrame(
    const FramePtr& frame,
    const MapConstPtr& map)
{
    id_ = global_id_++;
    frame_ = frame;
    map_ = map;
}

KeyFrame::~KeyFrame() {
}

const cv::Mat& KeyFrame::getWorldPos() const {
    return frame_->getCamInWorldt();
}

const cv::Mat& KeyFrame::descriptors() const {
    return frame_->descriptorsUndist();
}

cv::Mat KeyFrame::descriptor(const size_t& idx) const {
    return frame_->descriptorsUndist().row(idx);
}

void KeyFrame::resizeMap(const size_t& n)
{
    frame_->resizeMap(n);
}

void KeyFrame::addMapPoint(const MapPointPtr& mp, const size_t& idx)
{
    frame_->addMapPoint(mp, idx);
}

void KeyFrame::removeMapPointAt(const unsigned long& idx) {
    frame_->removeMapPointAt(idx);
}

void KeyFrame::addChild(const KeyFramePtr& kf) {
    std::unique_lock<std::mutex> connections_lock(connections_mutex_);
    childs_.insert(kf);
}

void KeyFrame::removeChild(const KeyFramePtr& kf) {
    std::unique_lock<std::mutex> connections_lock(connections_mutex_);
    childs_.erase(kf);
}

void KeyFrame::addConnection(const KeyFramePtr& kf, const int& weight)
{
    { // shared
        std::unique_lock<std::mutex> connections_lock(connections_mutex_);
        if(!conn_key_frame_weights_.count(kf))
            conn_key_frame_weights_[kf] = weight;
        else if(conn_key_frame_weights_[kf] != weight)
            conn_key_frame_weights_[kf] = weight;
        else
            return;
    }

    updateBestCovisibles();
}

void KeyFrame::updateBestCovisibles() {
    std::unique_lock<std::mutex> connections_lock(connections_mutex_);
    std::vector<std::pair<int, KeyFramePtr> > pairs;
    pairs.reserve(conn_key_frame_weights_.size());
    for(auto it = conn_key_frame_weights_.begin();
        it != conn_key_frame_weights_.end(); it++) {
        pairs.push_back(make_pair(it->second, it->first));
    }

    sort(pairs.begin(), pairs.end());
    std::list<KeyFramePtr> key_frames_list;
    std::list<int> weights_list;
    for(size_t i = 0; i < pairs.size(); i++) {
        key_frames_list.push_front(pairs[i].second);
        weights_list.push_front(pairs[i].first);
    }

    ordered_conn_key_frames_ =
        std::vector<KeyFramePtr>(
            key_frames_list.begin(), key_frames_list.end());
    conn_weights_ =
        std::vector<int>(weights_list.begin(), weights_list.end());
}

void KeyFrame::updateConnections() {
    std::map<KeyFramePtr, int> kf_counter;
    std::vector<MapPointPtr> map_points;
    { // shared
        std::unique_lock<std::mutex> (frame_->mutex_map_points_);
        map_points = frame_->obs_map_points_;
    }

    // for all map points in keyframe check in which other keyframes
    // are they seen. Increase the counter for those keyframes
    for (auto it = map_points.begin(); it != map_points.end(); it++) {
        const auto& mp = *it;
        if (!mp || mp->isBad())
            continue;
        const auto& observations = mp->observations();
        for (auto it = observations.begin(); it != observations.end(); it++) {
            if(it->first->id_ == id_)
                continue;
            kf_counter[it->first]++;
        }
    }

    // this should not happen
    if(kf_counter.empty())
        return;

    // if the counter is greater than threshold add connection
    // in case no keyframe counter is over threshold add the one with maximum counter
    int n_max = 0;
    KeyFramePtr key_frame_max = NULL;
    int th = 15;

    std::vector<std::pair<int, KeyFramePtr>> pairs;
    pairs.reserve(kf_counter.size());
    for (auto it = kf_counter.begin(); it != kf_counter.end(); it++) {
        if (it->second > n_max) {
            n_max = it->second;
            key_frame_max = it->first;
        }

        if (it->second >= th) {
            pairs.push_back(std::make_pair(it->second, it->first));
            (it->first)->addConnection(shared_from_this(), it->second);
        }
    }

    if(pairs.empty()) {
        pairs.push_back(std::make_pair(n_max, key_frame_max));
        key_frame_max->addConnection(shared_from_this(), n_max);
    }

    sort(pairs.begin(), pairs.end());
    std::list<KeyFramePtr> key_frames_list;
    std::list<int> weights_list;
    for (size_t i = 0; i < pairs.size(); i++) {
        key_frames_list.push_front(pairs[i].second);
        weights_list.push_front(pairs[i].first);
    }

    { // shared
        std::unique_lock<std::mutex> connections_lock(connections_mutex_);

        // mspConnectedKeyFrames = spConnectedKeyFrames;
        conn_key_frame_weights_ = kf_counter;
        ordered_conn_key_frames_ =
            std::vector<KeyFramePtr>(
                key_frames_list.begin(),key_frames_list.end());
        conn_weights_ =
            std::vector<int>(weights_list.begin(), weights_list.end());

        if(first_connection_ && id_ != 0) {
            parent_ = ordered_conn_key_frames_.front();
            parent_->addChild(shared_from_this());
            first_connection_ = false;
        }
    }
}

} // namespace orb_slam
