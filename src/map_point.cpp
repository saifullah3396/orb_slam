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

} // namespace orb_slam