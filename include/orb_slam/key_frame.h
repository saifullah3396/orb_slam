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

class TrackProperties;

#define LOCK_CONNECTIONS \
    std::unique_lock<std::mutex> connections_lock(connections_mutex_)
#define LOCK_FRAME_MAP \
    std::unique_lock<std::mutex> map_points_lock(map_points_mutex_)
#define LOCK_FRAME_POSE \
    std::unique_lock<std::mutex> frame_pose_lock(pose_mutex_)

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
    const cv::Mat& getWorldPos() const;
    const bool& isBad() { return bad_frame_; }
    const cv::Mat& descriptors() const;
    cv::Mat descriptor(const size_t& idx) const;

    /**
     * Setters
     */
    void setRefKeyFrame(const KeyFramePtr& ref_key_frame) {
        ref_key_frame_ = ref_key_frame;
    }

    /**
     * Resizes the map to given size
     * @param n: Map size
     */
    void resizeMap(const size_t& n);

    /**
     * Adds a map point to the list of points associated with the frame.
     * @param mp: The map point to be pushed
     * @param idx: Index where the point is to be added
     */
    void addMapPoint(const MapPointPtr& mp, const size_t& idx);

    /**
     * Removes a map point at the given index
     * @param idx: The map point index
     */
    void removeMapPointAt(const unsigned long& idx);
    void addChild(const KeyFramePtr& kf);
    void removeChild(const KeyFramePtr& kf);
    void addConnection(const KeyFramePtr& kf, const int& weight);
    void updateBestCovisibles();
    void updateConnections();

private:
    FramePtr frame_; // Associated frame
    KeyFramePtr ref_key_frame_;
    bool bad_frame_ = {false}; // Whether this frame is bad
    long unsigned int id_; // Key frame id
    static long unsigned int global_id_; // Id accumulator

    // key frame connection parameters
    std::map<KeyFramePtr, int> conn_key_frame_weights_;
    std::vector<KeyFramePtr> ordered_conn_key_frames_;
    std::vector<int> conn_weights_;
    bool first_connection_ = {true};
    KeyFramePtr parent_;
    std::set<KeyFramePtr> childs_;
    std::set<KeyFramePtr> loop_edges_;

    // const pointer to the underlying map
    MapConstPtr map_;
    std::mutex connections_mutex_; // mutex for updating connections
};

} // namespace orb_slam
