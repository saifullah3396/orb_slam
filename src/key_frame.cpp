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

} // namespace orb_slam
