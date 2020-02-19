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
    world_pos_ = frame->getWorldToCamT().rowRange(0,3).col(3);
}

KeyFrame::~KeyFrame() {
}


void KeyFrame::resizeMap(const size_t& n)
{
    std::unique_lock<std::mutex> lock(mutex_map_points_);
    frame_->obs_map_points_.resize(n);
}

void KeyFrame::addMapPoint(const MapPointPtr& mp, const size_t& idx)
{
    std::unique_lock<std::mutex> lock(mutex_map_points_);
    frame_->obs_map_points_[idx] = mp;
}

void KeyFrame::removeMapPointAt(const unsigned long& idx) {
    std::unique_lock<std::mutex> lock(mutex_map_points_);
    frame_->obs_map_points_[idx].reset();
}
} // namespace orb_slam
