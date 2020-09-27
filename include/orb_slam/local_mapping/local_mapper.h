/**
 * Declares the LocalMapper class.
 */

#pragma once

#include <atomic>
#include <map>
#include <mutex>
#include <memory>
#include <queue>
#include <thread>
#include <opencv2/core/core.hpp>

#define LOCK_BUSY std::unique_lock<std::mutex> lock_busy(busy_mutex_)
#define LOCK_RESET std::unique_lock<std::mutex> lock_reset(reset_mutex_)
#define LOCK_STOPPABLE std::unique_lock<std::mutex> lock_stoppable(stoppable_mutex_)
#define LOCK_FINISH std::unique_lock<std::mutex> lock_finish(finish_mutex_)
#define LOCK_NEW_KF_QUEUE \
    std::unique_lock<std::mutex> lock_new_key_frames(new_key_frames_mutex_)

namespace orb_slam
{

class Map;
using MapPtr = std::shared_ptr<Map>;

class LocalBundleAdjuster;
using LocalBundleAdjusterPtr = std::shared_ptr<LocalBundleAdjuster>;

class KeyFrame;
using KeyFramePtr = std::shared_ptr<KeyFrame>;

class LocalMapper : public std::enable_shared_from_this<LocalMapper>
{
public:
    LocalMapper(const MapPtr& map);
    ~LocalMapper() {
        local_mapper_thread_.join();
    }

    /**
     * Starts the thread for main local mapping operation
     */
    void startThread() {
        local_mapper_thread_ =
            std::thread(std::bind(&LocalMapper::threadCall, this));
    }

    /**
     * The main loop of the local mapper
     */
    void threadCall();

    /**
     * Processes a new key frame found in new_key_frames_ queue.
     */
    void processNewKeyFrame();

    /**
     * Removes bad key frames from the map
     */
    void keyFramesCulling();

    /**
     * Removes bad map points from the map
     */
    void mapPointCulling();

    /**
     * Generates new map points by triangulating between two covisible key
     * frames using the epipolar constraint.
     */
    void createNewMapPoints();

    /**
     * Searches more
     */
    void searchInNeighbors();

    void requestStop() {
        LOCK_STOPPABLE;
        stop_requested_ = true;
        LOCK_NEW_KF_QUEUE;
        abort_ba_ = true;
    }

    /**
     * Returns the number available key frames in queue
     * @return size_t
     */
    const size_t nKeyFramesInQueue() const {
        LOCK_NEW_KF_QUEUE;
        return new_key_frames_.size();
    }

    /**
     * Returns true if the new key frames exist in the queue.
     * @returns boolean
     */
    bool newKeyFramesExist() const {
        LOCK_NEW_KF_QUEUE;
        return !new_key_frames_.empty();
    }

    /**
     * Adds a key frame to queue for local mapping loop
     * @param key_frame: Key frame to add
     */
    void addKeyFrameToQueue(const KeyFramePtr& key_frame) {
        LOCK_NEW_KF_QUEUE;
        new_key_frames_.push(key_frame);
    }

    /**
     * Stops the local mapper completely
     */
    bool stop();

    /**
     * Resets the local mapper if requested
     */
    void resetIfRequested() {
        LOCK_RESET;
        if (reset_requested_) {
            new_key_frames_ = std::queue<KeyFramePtr>();
            map_points_to_cull_.clear();
            reset_requested_ = false;
        }
    }

    /**
     * Getters
     */
    const bool isBusy() const {
        LOCK_BUSY;
        return busy_;
    }

    const bool stopped() const {
        LOCK_STOPPABLE;
        return stopped_;
    }

    const bool stopRequested() const {
        LOCK_STOPPABLE;
        return stop_requested_;
    }

    const bool finishRequested() const {
        LOCK_FINISH;
        return finish_requested_;
    }

    /**
     * Setters
     */
    void setBusy(const bool busy) {
        LOCK_BUSY;
        busy_ = busy;
    }

    void setStoppable(const bool& stoppable) {
        LOCK_STOPPABLE;
        stoppable_ = stoppable;
    }

    void setFinished() {
        LOCK_FINISH;
        finished_ = true;
        LOCK_STOPPABLE;
        stopped_ = true;
    }

    void abortBA(const bool& abort_ba = true) {
        abort_ba_ = abort_ba;
    }
private:
    MapPtr map_; // map pointer
    LocalBundleAdjusterPtr local_ba_;

    KeyFramePtr key_frame_in_process_; // the key frame currently in process

    // all the newly added points that are created based on triangulation
    // between covisible key frames
    std::vector<MapPointPtr> triang_map_points_;
    std::vector<MapPointPtr> map_points_to_cull_;

    std::queue<KeyFramePtr> new_key_frames_; // queue of new incoming key frames
     // mutex for acccesing new_key_frames_ variable
    mutable std::mutex new_key_frames_mutex_;

    // whether the mapper is currently busy processing new frames
    bool busy_ = {false};
    mutable std::mutex busy_mutex_; // mutex of accessing busy_ variable

    bool stoppable_ = {true}; // whether the mapper can be stopped or not
    bool stopped_ = {false}; // whether the mapper is currently stopped
    bool stop_requested_ = {false}; // whether stop is requested
    mutable std::mutex stoppable_mutex_; // mutex of accessing stop variables

    bool finished_ = {false}; // whether the local mapper has finished
    bool finish_requested_ = {false}; // whether finish is requested
    mutable std::mutex finish_mutex_; // mutex of accessing finish variables

    bool reset_requested_ = {false}; // whether reset is requested
    mutable std::mutex reset_mutex_; // mutex of accessing reset variables

    bool abort_ba_ = {false}; // toggle for bundle adjustment
    std::thread local_mapper_thread_;

    std::string name_tag_ = {"LocalMapper"};
};

} // namespace orb_slam