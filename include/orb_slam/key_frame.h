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
using MapPtr = std::shared_ptr<Map>;

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
        const MapPtr& map);

    ~KeyFrame();

    void match(const KeyFramePtr& key_frame, std::vector<cv::DMatch>& matches);
    int fuse(const KeyFramePtr& key_frame);

    /**
     * Getters
     */
    const long unsigned int& id() const;
    const bool isInLocalMapOf(const long unsigned int& id) const
        { return this->is_in_local_map_of_ == id; }
    const bool queriedInFrame(const long unsigned int& id) const
        { return this->queried_in_frame_ == id; }
    const bool searchedInMapWith(const long unsigned int& id) const
        { return this->searched_in_map_with_ == id; }
    const bool inLocalAdjustmentOf(const long unsigned int& id) const
        { return this->in_local_adjustment_of_ == id; }
    const bool inLocalFixedAdjustmentOf(const long unsigned int& id) const
        { return this->in_local_fixed_adjustment_of_ == id; }
    const int& nMatchingWords() const { return n_matching_words_; }
    const float& matchingScore() const { return matching_score_; }
    const FramePtr& frame() const { return frame_; }
    const cv::Mat getWorldPos() const;
    const bool& isBad() { return bad_frame_; }
    const cv::Mat& descriptors() const;
    cv::Mat descriptor(const size_t& idx) const;
    // returns a copy in contrast with Frame::obsMapPoints() since map points
    // can be changed once a key frame is formed from frame
    std::vector<MapPointPtr> obsMapPoints() const;
    std::vector<KeyFramePtr> getCovisibles() const;
    std::vector<KeyFramePtr> getBestCovisibles(const int n) const;
    const int nTrackedPoints(const int n_min_obs = -1) const;

    /**
     * Setters
     */
    void setNMatchingWords(const int n) { n_matching_words_ = n; }
    void increaseNMatchingWords() { n_matching_words_++; }
    void setMatchingScore(const float& score) { matching_score_ = score; }
    void setRefKeyFrame(const KeyFramePtr& ref_key_frame);
    void setIsInLocalMapOf(const long unsigned int& id)
        { this->is_in_local_map_of_ = id; }
    void setSearchedInMapWith(const long unsigned int& id)
        { this->searched_in_map_with_ = id; }
    void setQueriedInFrame(const long unsigned int& id)
        { this->queried_in_frame_ = id; }
    void setInLocalAdjustmentOf(const long unsigned int& id)
        { this->in_local_adjustment_of_ = id; }
    void setInLocalFixedAdjustmentOf(const long unsigned int& id)
        { this->in_local_fixed_adjustment_of_ = id; }
    void setErasable(const bool erasable)
    {
        if (erasable) {
            LOCK_CONNECTIONS;
            if (loop_edges_.empty())
                erasable_ = erasable;
            if (to_be_erased_)
                removeFromMap();
        } else {
            LOCK_CONNECTIONS;
            erasable_ = false;
        }
    }

    /**
     * Resets the map to to null
     */
    void resetMap();

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
    void setMapPointAt(const MapPointPtr& mp, const size_t& idx);

    /**
     * Removes a given map point from key frame
     * @param mp: The map point to remove
     */
    void removeMapPoint(const MapPointPtr& mp);

    /**
     * Removes a map point at the given index
     * @param idx: The map point index
     */
    void removeMapPointAt(const unsigned long& idx);

    /**
     * Removes this key frame from the map
     */
    void removeFromMap();

    const KeyFramePtr& parent() const { return parent_; }
    const std::set<KeyFramePtr>& children() const { return children_; }
    void addChild(const KeyFramePtr& kf);
    void removeChild(const KeyFramePtr& kf);
    void changeParent(const KeyFramePtr& key_frame);
    void addConnection(const KeyFramePtr& kf, const int& weight);
    void removeConnection(const KeyFramePtr& key_frame);
    void updateBestCovisibles();
    void updateConnections();
    int getWeight(const KeyFramePtr& key_frame);

    // matching with another frame

    // frame related methods for thread safe access
    const cv::Mat cameraInWorldT() const;
    const cv::Mat cameraInWorldR() const;
    const cv::Mat cameraInWorldt() const;
    const cv::Mat worldInCameraT() const;
    const cv::Mat worldInCameraR() const;
    const cv::Mat worldInCamerat() const;
    const cv::Mat& cameraInWorldTLocal() const;
    const cv::Mat& cameraInWorldRLocal() const;
    const cv::Mat& cameraInWorldtLocal() const;
    const cv::Mat& worldInCameraTLocal() const;
    const cv::Mat& worldInCameraRLocal() const;
    const cv::Mat& worldInCameratLocal() const;
    template <typename T>
    cv::Point3_<T> cameraToWorld(const cv::Point3_<T>& p);
    template <typename T>
    cv::Point3_<T> cameraToWorld(const cv::Mat_<T>& p);
    template <typename T>
    cv::Point3_<T> worldToCamera(const cv::Point3_<T>& p);
    template <typename T>
    cv::Point3_<T> worldToCamera(const cv::Mat_<T>& p);
    template <typename U, typename V>
    cv::Point3_<U> frameToWorld(const cv::Point_<V>& p, const float& depth);
    template <typename U, typename V>
    cv::Point3_<U> worldToFrame(const cv::Point3_<V>& p);
    template <typename T>
    cv::Point3_<T> cameraToWorldLocal(const cv::Point3_<T>& p);
    template <typename T>
    cv::Point3_<T> cameraToWorldLocal(const cv::Mat_<T>& p);
    template <typename T>
    cv::Point3_<T> worldToCameraLocal(const cv::Point3_<T>& p);
    template <typename T>
    cv::Point3_<T> worldToCameraLocal(const cv::Mat_<T>& p);
    template <typename U, typename V>
    cv::Point3_<U> frameToWorldLocal(const cv::Point_<V>& p, const float& depth);
    template <typename U, typename V>
    cv::Point3_<U> worldToFrameLocal(const cv::Point3_<V>& p);

    /**
     * Setters
     */
    void setCamInWorld(const cv::Mat& w_T_c);
    void setWorldInCam(const cv::Mat& c_T_w);

    /**
     * Stores current pose information from camera to world in _local pose
     * variables.
     */
    void updateWorldInCamLocal();

    /**
     * Stores current pose information from world to camera in _local pose
     * variables.
     */
    void updateCamInWorldLocal();

    /**
     * Returns true if the map point in 3d space lies in the view or frustum of
     * current frame.
     * @param mp: Map point to check
     * @param view_cos_limit: Angle limit
     * @param track_res: Tracking resulting info
     * @returns boolean
     */
    bool isInCameraView(
        const MapPointPtr& mp,
        TrackProperties& track_res,
        const float view_cos_limit = 0.5);

    bool isInCameraViewLocal(
        const MapPointPtr& mp,
        TrackProperties& track_res,
        const float view_cos_limit = 0.5);

    void computeFundamentalMat(
        cv::Mat& f_mat,
        const FramePtr& frame);

    void computeFundamentalMat(
        cv::Mat& f_mat,
        const KeyFramePtr& key_frame);

private:
    FramePtr frame_; // Associated frame
    bool bad_frame_ = {false}; // Whether this frame is bad

    // id of frame this frame is in local bundle adjusment of
    int in_local_adjustment_of_;
    int in_local_fixed_adjustment_of_;

    // id of frame whose local map this key frame is in
    int is_in_local_map_of_;

    // id of frame this key frame is searched in map of
    int searched_in_map_with_;

    // key frame connection parameters
    std::map<KeyFramePtr, int> conn_key_frame_weights_;
    std::vector<KeyFramePtr> ordered_conn_key_frames_;
    std::vector<int> conn_weights_;
    bool first_connection_ = {true};
    KeyFramePtr parent_;
    std::set<KeyFramePtr> children_;
    std::set<KeyFramePtr> loop_edges_;
    cv::Mat c_T_p_; // parent in child transformation

    // data base search parameters
    // Id of the frame this key frame was queried in
    long unsigned int queried_in_frame_;
    // Number of matching words found between this and querying frame
    int n_matching_words_;
    // score of bag of words matching between this and querying frame
    float matching_score_;

    // const pointer to the underlying map
    MapPtr map_;
    mutable std::mutex connections_mutex_; // mutex for updating connections

    // frame related variables
    // pose information matrices that are used for computation and are not
    // updated through any thread
    cv::Mat w_T_c_local_;
    cv::Mat w_R_c_local_;
    cv::Mat w_t_c_local_;
    cv::Mat c_T_w_local_;
    cv::Mat c_R_w_local_;
    cv::Mat c_t_w_local_;

    // map points access mutex
    mutable std::mutex map_points_mutex_;
    mutable std::mutex pose_mutex_;

    // flags for removal of this frame
    bool erasable_ = {false};
    bool to_be_erased_ = {false};
};

} // namespace orb_slam
