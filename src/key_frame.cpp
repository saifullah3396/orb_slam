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

KeyFrame::KeyFrame(
    const FramePtr& frame,
    const MapConstPtr& map)
{
    frame_ = frame;
    map_ = map;
}

KeyFrame::~KeyFrame() {
}

void KeyFrame::match(
    const KeyFramePtr& key_frame, std::vector<cv::DMatch>& matches)
{
    frame_->orb_matcher_->matchByEpipolarConstraint(
        frame_, key_frame->frame(), matches);
}

const long unsigned int& KeyFrame::id() const {
    return frame_->id();
}

const cv::Mat KeyFrame::getWorldPos() const {
    return frame_->cameraInWorldt();
}

const cv::Mat& KeyFrame::descriptors() const {
    return frame_->descriptorsUndist();
}

cv::Mat KeyFrame::descriptor(const size_t& idx) const {
    return frame_->descriptorsUndist().row(idx);
}

void KeyFrame::setRefKeyFrame(const KeyFramePtr& ref_key_frame) {
    frame_->setRefKeyFrame(ref_key_frame);
}

void KeyFrame::resizeMap(const size_t& n)
{
    frame_->resizeMap(n);
}

void KeyFrame::setMapPointAt(const MapPointPtr& mp, const size_t& idx)
{
    frame_->setMapPointAt(mp, idx);
}

void KeyFrame::removeMapPointAt(const unsigned long& idx) {
    frame_->removeMapPointAt(idx);
}

void KeyFrame::addChild(const KeyFramePtr& kf) {
    LOCK_CONNECTIONS;
    childs_.insert(kf);
}

void KeyFrame::removeChild(const KeyFramePtr& kf) {
    LOCK_CONNECTIONS;
    childs_.erase(kf);
}

void KeyFrame::addConnection(const KeyFramePtr& kf, const int& weight)
{
    { // shared
        LOCK_CONNECTIONS;
        if(!conn_key_frame_weights_.count(kf))
            conn_key_frame_weights_[kf] = weight;
        else if(conn_key_frame_weights_[kf] != weight)
            conn_key_frame_weights_[kf] = weight;
        else
            return;
    }

    updateBestCovisibles();
}

std::vector<KeyFramePtr> KeyFrame::getCovisibles() const
{
    LOCK_CONNECTIONS;
    return ordered_conn_key_frames_;
}

std::vector<KeyFramePtr> KeyFrame::getBestCovisibles(const int n) const
{
    LOCK_CONNECTIONS;
    if (ordered_conn_key_frames_.size() < n) {
        return ordered_conn_key_frames_;
    } else {
        return
            std::vector<KeyFramePtr>(
                ordered_conn_key_frames_.begin(),
                ordered_conn_key_frames_.end() + n);
    }
}

const int KeyFrame::nTrackedPoints(const int n_min_obs) const
{
    LOCK_FRAME_MAP;
    int n_tracked = 0;
    const bool check_obs = n_min_obs > 0;
    for (const auto& mp: frame_->obs_map_points_) {
        if (mp && !mp->isBad()) {
            if (check_obs) {
                if (mp->nObservations() >= n_min_obs)
                    n_tracked++;
            } else {
                n_tracked++;
            }
        }
    }
    return n_tracked;
}

void KeyFrame::updateBestCovisibles() {
    LOCK_CONNECTIONS;
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
        LOCK_FRAME_MAP;
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
            if(it->first->id() == id())
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
        LOCK_CONNECTIONS;

        // mspConnectedKeyFrames = spConnectedKeyFrames;
        conn_key_frame_weights_ = kf_counter;
        ordered_conn_key_frames_ =
            std::vector<KeyFramePtr>(
                key_frames_list.begin(),key_frames_list.end());
        conn_weights_ =
            std::vector<int>(weights_list.begin(), weights_list.end());

        if(first_connection_ && id() != 0) {
            parent_ = ordered_conn_key_frames_.front();
            parent_->addChild(shared_from_this());
            first_connection_ = false;
        }
    }
}

// frame access definitions
// frame related methods for thread safe access
std::vector<MapPointPtr> KeyFrame::obsMapPoints() const {
    LOCK_FRAME_MAP;
    return frame_->obs_map_points_;
}
const cv::Mat KeyFrame::cameraInWorldT() const
    { LOCK_FRAME_POSE; return frame_->w_T_c_.clone(); }
const cv::Mat KeyFrame::cameraInWorldR() const
    { LOCK_FRAME_POSE; return frame_->w_R_c_.clone(); }
const cv::Mat KeyFrame::cameraInWorldt() const
    { LOCK_FRAME_POSE; return frame_->w_t_c_.clone(); }
const cv::Mat KeyFrame::worldInCameraT() const
    { LOCK_FRAME_POSE; return frame_->c_T_w_.clone(); }
const cv::Mat KeyFrame::worldInCameraR() const
    { LOCK_FRAME_POSE; return frame_->c_R_w_.clone(); }
const cv::Mat KeyFrame::worldInCamerat() const
    { LOCK_FRAME_POSE; return frame_->c_t_w_.clone(); }

const cv::Mat& KeyFrame::cameraInWorldTLocal() const { return w_T_c_local_; }
const cv::Mat& KeyFrame::cameraInWorldRLocal() const { return w_R_c_local_; }
const cv::Mat& KeyFrame::cameraInWorldtLocal() const { return w_t_c_local_; }
const cv::Mat& KeyFrame::worldInCameraTLocal() const { return c_T_w_local_; }
const cv::Mat& KeyFrame::worldInCameraRLocal() const { return c_R_w_local_; }
const cv::Mat& KeyFrame::worldInCameratLocal() const { return c_t_w_local_; }

template <typename T>
cv::Point3_<T> KeyFrame::cameraToWorld(const cv::Point3_<T>& p) {
    LOCK_FRAME_POSE;
    return frame_->cameraToWorld(p);
}

template <typename T>
cv::Point3_<T> KeyFrame::cameraToWorld(const cv::Mat_<T>& p) {
    LOCK_FRAME_POSE;
    return frame_->cameraToWorld(p);
}

template <typename T>
cv::Point3_<T> KeyFrame::worldToCamera(const cv::Point3_<T>& p) {
    LOCK_FRAME_POSE;
    return frame_->worldToCamera(p);
}

template <typename T>
cv::Point3_<T> KeyFrame::worldToCamera(const cv::Mat_<T>& p) {
    LOCK_FRAME_POSE;
    return frame_->worldToCamera(p);
}

template <typename U, typename V>
cv::Point3_<U> KeyFrame::frameToWorld(const cv::Point_<V>& p, const float& depth) {
    LOCK_FRAME_POSE;
    return cameraToWorld<U>(frameToCamera<U, V>(p, depth));
}

template <typename U, typename V>
cv::Point3_<U> KeyFrame::worldToFrame(const cv::Point3_<V>& p) {
    LOCK_FRAME_POSE;
    return cameraToFrame<U, V>(worldToCamera<V>(p));
}

template <typename T>
cv::Point3_<T> KeyFrame::cameraToWorldLocal(const cv::Point3_<T>& p) {
    return cv::Point3_<T>(
        cv::Mat(w_R_c_local_ * cv::Mat(p) + w_t_c_local_));
}

template <typename T>
cv::Point3_<T> KeyFrame::cameraToWorldLocal(const cv::Mat_<T>& p) {
    return cv::Point3_<T>(cv::Mat(w_R_c_local_ * p + w_t_c_local_));
}

template <typename T>
cv::Point3_<T> KeyFrame::worldToCameraLocal(const cv::Point3_<T>& p) {
    return cv::Point3_<T>(cv::Mat(c_R_w_local_ * cv::Mat(p) + c_t_w_local_));
}

template <typename T>
cv::Point3_<T> KeyFrame::worldToCameraLocal(const cv::Mat_<T>& p) {
    return cv::Point3_<T>(cv::Mat(c_R_w_local_ * p + c_t_w_local_));
}

template <typename U, typename V>
cv::Point3_<U> KeyFrame::frameToWorldLocal(const cv::Point_<V>& p, const float& depth) {
    return cameraToWorldLocal<U>(frameToCameraLocal<U, V>(p, depth));
}

template <typename U, typename V>
cv::Point3_<U> KeyFrame::worldToFrameLocal(const cv::Point3_<V>& p) {
    return cameraToFrameLocal<U, V>(worldToCameraLocal<V>(p));
}

void KeyFrame::setCamInWorld(const cv::Mat& w_T_c) {
    LOCK_FRAME_POSE;
    frame_->setCamInWorld(w_T_c);
}

void KeyFrame::setWorldInCam(const cv::Mat& c_T_w) {
    LOCK_FRAME_POSE;
    frame_->setWorldInCam(c_T_w);
}

void KeyFrame::updateWorldInCamLocal() {
    LOCK_FRAME_POSE;
    c_T_w_local_ = frame_->c_T_w_.clone();
    c_R_w_local_ = frame_->c_R_w_.clone();
    c_t_w_local_ = frame_->c_t_w_.clone();
}

void KeyFrame::updateCamInWorldLocal() {
    LOCK_FRAME_POSE;
    w_T_c_local_ = frame_->w_T_c_.clone();
    w_R_c_local_ = frame_->w_R_c_.clone();
    w_t_c_local_ = frame_->w_t_c_.clone();
}

bool KeyFrame::isInCameraView(
    const MapPointPtr& mp,
    TrackProperties& track_res,
    const float view_cos_limit = 0.5)
{
    // get 3d position of map point
    auto pos = worldToCamera<float>(mp->worldPos());

    // check depth to see if the point is in front or not
    if (pos.z < 0.0f)
        return false;

    // project the point to image to see if it lies inside the bounds
    auto img = frame_->cameraToFrame<float, float>(pos);

    if (!frame_->pointWithinBounds(img))
        return false;

    // check point distance is within the scale invariance region
    const auto& max_dist = mp->maxScaleInvDist();
    const auto& min_dist = mp->minScaleInvDist();
    const float dist = cv::norm(pos);

    // return false if out of range
    if(dist < min_dist || dist > max_dist)
        return false;

    // check viewing angle
    cv::Mat view_vec =  mp->viewVector();

    // view_vec is already norm 1
    // cos (angle) = a.b / |a| |b|
    const float cosine = (cv::Mat(pos) - frame_->w_t_c_).dot(view_vec) / dist;
    if(cosine < view_cos_limit) // outside angle range...
        return false;

    // predict the scale of this point in this image
    const int predicted_scale_level = mp->predictScale(dist);

    // Data used by the tracking
    track_res.in_view_ = true;
    track_res.proj_xy_ = img;
    //if (camera_->type() == geometry::CameraType::STEREO)
        //track_properties.proj_xr = toRightCam(img.x);
    track_res.pred_scale_level_ = predicted_scale_level;
    track_res.view_cosine_ = cosine;

    return true;
}

bool KeyFrame::isInCameraViewLocal(
    const MapPointPtr& mp,
    TrackProperties& track_res,
    const float view_cos_limit = 0.5)
{
    // get 3d position of map point
    auto pos = worldToCameraLocal<float>(mp->worldPos());

    // check depth to see if the point is in front or not
    if (pos.z < 0.0f)
        return false;

    // project the point to image to see if it lies inside the bounds
    auto img = frame_->cameraToFrame<float, float>(pos);

    if (!frame_->pointWithinBounds(img))
        return false;

    // check point distance is within the scale invariance region
    const auto& max_dist = mp->maxScaleInvDist();
    const auto& min_dist = mp->minScaleInvDist();
    const float dist = cv::norm(pos);

    // return false if out of range
    if(dist < min_dist || dist > max_dist)
        return false;

    // check viewing angle
    cv::Mat view_vec =  mp->viewVector();

    // view_vec is already norm 1
    // cos (angle) = a.b / |a| |b|
    const float cosine = (cv::Mat(pos) - w_t_c_local_).dot(view_vec) / dist;
    if(cosine < view_cos_limit) // outside angle range...
        return false;

    // predict the scale of this point in this image
    const int predicted_scale_level = mp->predictScale(dist);

    // Data used by the tracking
    track_res.in_view_ = true;
    track_res.proj_xy_ = img;
    //if (camera_->type() == geometry::CameraType::STEREO)
        //track_properties.proj_xr = toRightCam(img.x);
    track_res.pred_scale_level_ = predicted_scale_level;
    track_res.view_cosine_ = cosine;

    return true;
}

void KeyFrame::computeFundamentalMat(
    cv::Mat& f_mat,
    const FramePtr& frame)
{
    Frame::computeFundamentalMat(
        f_mat,
        worldInCameraR(),
        worldInCamerat(),
        frame->cameraInWorldR(),
        frame->cameraInWorldt(),
        this->frame_->camera()->intrinsicMatrix(),
        frame->camera()->intrinsicMatrix());
}

void KeyFrame::computeFundamentalMat(
    cv::Mat& f_mat,
    const KeyFramePtr& key_frame)
{
    Frame::computeFundamentalMat(
        f_mat,
        worldInCameraR(),
        worldInCamerat(),
        key_frame->cameraInWorldR(),
        key_frame->cameraInWorldt(),
        this->frame_->camera()->intrinsicMatrix(),
        key_frame->frame()->camera()->intrinsicMatrix());
}

} // namespace orb_slam
