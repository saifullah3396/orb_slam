/**
 * Implements the Map class.
 */

#include <memory>
#include <mutex>
#include <set>
#include <orb_slam/key_frame.h>
#include <orb_slam/map.h>
#include <orb_slam/map_point.h>

namespace orb_slam
{

void Map::addKeyFrame(const KeyFramePtr& key_frame) {
    LOCK_MAP;
    key_frames_.insert(key_frame);
    if (key_frame->id() >= max_key_frame_id_)
        max_key_frame_id_ = key_frame->id();
}

void Map::addRefKeyFrame(const KeyFramePtr& ref_key_frame) {
    LOCK_MAP;
    ref_key_frames_.push_back(ref_key_frame);
}

void Map::removeKeyFrame(const KeyFramePtr& kf)
{
    LOCK_MAP;
    key_frames_.erase(kf);
}

void Map::addMapPoint(const MapPointPtr& mp) {
    LOCK_MAP;
    map_points_.insert(mp);
}

void Map::removeMapPoint(const MapPointPtr& mp) {
    LOCK_MAP;
    map_points_.erase(mp);
}

void Map::addRefMapPoint(const MapPointPtr& mp) {
    LOCK_MAP;
    ref_map_points_.push_back(mp);
}

} // namespace orb_slam