/**
 * Declares the KeyFrame class.
 */

#pragma once

#include <memory>
#include <mutex>
#include <orb_slam/geometry/utils.h>
#include <DBoW2/BowVector.h>
#include <DBoW2/FeatureVector.h>

namespace orb_slam
{

class Map;
using MapConstPtr = std::shared_ptr<const Map>;

class MapPoint;
using MapPointPtr = std::shared_ptr<MapPoint>;
using MapPointConstPtr = std::shared_ptr<const MapPoint>;

class Frame;
using FramePtr = std::shared_ptr<Frame>;

class KeyFrame;
using KeyFramePtr = std::shared_ptr<KeyFrame>;

class KeyFrame : public std::enable_shared_from_this<KeyFrame> {
public:
    /**
     * Initializes the key frame for a given frame input.
     * @param frame: Const pointer to the associated frame
     * @param map: Const pointer to the underlying map
     */
    KeyFrame(
        const FramePtr& frame,
        const MapConstPtr& map);

    ~KeyFrame();

    /**
     * Getters
     */
    const long unsigned int& id() const { return this->id_; }
    const FramePtr& frame() { return frame_; }
    const cv::Mat& getWorldPos() const { return world_pos_; }

private:
    FramePtr frame_; // Associated frame
    cv::Mat world_pos_; // World position of camera for this frame
    long unsigned int id_; // Key frame id
    static long unsigned int global_id_; // Id accumulator

    // const pointer to the underlying map
    MapConstPtr map_;

    std::mutex mutex_map_points_; // mutex for updating map points
    std::mutex connections_mutex_; // mutex for updating connections
};

} // namespace orb_slam
