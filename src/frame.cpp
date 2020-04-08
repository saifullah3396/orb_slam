/**
 * Implements the Frame class.
 */

#include <opencv2/highgui/highgui.hpp>
#include <orb_slam/frame.h>
#include <orb_slam/key_frame.h>
#include <orb_slam/map_point.h>
#include <thread>

namespace orb_slam {

//! static variable definitions
long unsigned int Frame::id_global_ = 0;
geometry::CameraConstPtr<float> Frame::camera_;
geometry::ORBExtractorConstPtr Frame::orb_extractor_;
geometry::ORBMatcherConstPtr Frame::orb_matcher_;
ORBVocabularyConstPtr Frame::orb_vocabulary_;

// grid
int Frame::grid_rows_;
int Frame::grid_cols_;
int Frame::grid_size_x_;
int Frame::grid_size_y_;

Frame::Frame(
    const ros::Time& time_stamp) :
    time_stamp_(time_stamp)
{
    id_ = id_global_++;
}

Frame::~Frame()
{
}

void Frame::setRefKeyFrame(const KeyFramePtr& ref_key_frame) {
    ref_key_frame_ = ref_key_frame;
}

const std::vector<MapPointPtr>& Frame::obsMapPoints() const {
    assert(thread_safe_ == false);
    return obs_map_points_;
}


void Frame::resizeMap(const size_t& n)
{
    assert(thread_safe_ == false);
    obs_map_points_.resize(n);
}

void Frame::resetMap()
{
    assert(thread_safe_ == false);
    for (auto& mp: obs_map_points_) {
        mp.reset();
    }
}

void Frame::setMapPointAt(const MapPointPtr& mp, const size_t& idx)
{
    assert(thread_safe_ == false);
    obs_map_points_[idx] = mp;
}

void Frame::removeMapPointAt(const unsigned long& idx) {
    assert(thread_safe_ == false);
    obs_map_points_[idx].reset();
}

bool Frame::pointWithinBounds(const cv::Point2f& p)
{
    return camera_->pointWithinBounds(p);
}

bool Frame::getBoxAroundPoint(
    const cv::Point2f& p,
    const float& box_radius,
    int& left,
    int& right,
    int& up,
    int& down)
{
    left =
        std::max( //
            0, (int) floor(p.x - camera_->minX() - box_radius) / grid_size_x_);
    if (left >= grid_cols_) return false;

    right =
        std::min(
            grid_cols_ - 1,
            (int) ceil(p.x - camera_->minX() + box_radius) / grid_size_x_);
    if (right < 0) return false;

    up =
        std::max( //
            0, (int) floor(p.y - camera_->minY() - box_radius) / grid_size_y_);
    if (up >= grid_rows_) return false;

    down =
        std::min(
            grid_rows_ - 1,
            (int) ceil(p.y - camera_->minY() + box_radius) / grid_size_y_);
    if (down < 0) return false;
}

bool Frame::getFeaturesAroundPoint(
    const cv::Point2f& p,
    const float& radius,
    std::vector<size_t>& matches)
{
    int left, right, up, down;
    if (getBoxAroundPoint(p, radius, left, right, up, down)) {
        for (size_t x = left; x <= right; ++x) {
            for (size_t y = up; y <= down; ++y) {
                const auto& cell = grid_[x][y];
                // take the points in the box:
                // --r-----r--
                // r         r
                // |    X    |
                // r         r
                // --r-----r--
                if (cell.empty()) continue;
                for (size_t i = 0; i < cell.size(); ++i) {
                    const auto& key_point = undist_key_points_[cell[i]];
                    const auto diff_x = fabsf(key_point.pt.x - x);
                    const auto diff_y = fabsf(key_point.pt.y - y);
                    if (diff_x < radius && diff_y < radius)
                        matches.push_back(cell[i]);
                }
            }
        }
        return true;
    }
    return false;
}

bool Frame::getFeaturesAroundPoint(
    const cv::Point2f& p,
    const float& radius,
    const int& min_level,
    std::vector<size_t>& matches)
{
    int left, right, up, down;
    if (getBoxAroundPoint(p, radius, left, right, up, down)) {
        for (size_t x = left; x <= right; ++x) {
            for (size_t y = up; y <= down; ++y) {
                const auto& cell = grid_[x][y];
                // take the points in the box with given scale range:
                // --r-----r--
                // r         r
                // |    X    |
                // r         r
                // --r-----r--
                if (cell.empty()) continue;
                for (size_t i = 0; i < cell.size(); ++i) {
                    const auto& key_point = undist_key_points_[cell[i]];
                    const auto& scale_level = key_point.octave;
                    if (scale_level < min_level) continue;
                    const auto diff_x = fabsf(key_point.pt.x - x);
                    const auto diff_y = fabsf(key_point.pt.y - y);
                    if (diff_x < radius && diff_y < radius)
                        matches.push_back(cell[i]);
                }
            }
        }
        return true;
    }
    return false;
}

bool Frame::getFeaturesAroundPoint(
    const cv::Point2f& p,
    const float& radius,
    const int& min_level,
    const int& max_level,
    std::vector<size_t>& matches)
{
    int left, right, up, down;
    if (getBoxAroundPoint(p, radius, left, right, up, down)) {
        for (size_t x = left; x <= right; ++x) {
            for (size_t y = up; y <= down; ++y) {
                const auto& cell = grid_[x][y];
                // take the points in the box with given scale range:
                // --r-----r--
                // r         r
                // |    X    |
                // r         r
                // --r-----r--
                if (cell.empty()) continue;
                for (size_t i = 0; i < cell.size(); ++i) {
                    const auto& key_point = undist_key_points_[cell[i]];
                    if (key_point.octave < min_level || // octave = scale level
                        key_point.octave > max_level) continue;
                    const auto diff_x = fabsf(key_point.pt.x - p.x);
                    const auto diff_y = fabsf(key_point.pt.y - p.y);
                    if (diff_x < radius && diff_y < radius)
                        matches.push_back(cell[i]);
                }
            }
        }
        return true;
    }
    return false;
}

bool Frame::isInCameraView(
    const MapPointPtr& mp,
    TrackProperties& track_res,
    const float view_cos_limit)
{
    // get 3d position of map point
    auto pos = worldToCamera<float>(mp->worldPos());

    // check depth to see if the point is in front or not
    if (pos.z < 0.0f)
        return false;

    // project the point to image to see if it lies inside the bounds
    auto img = cameraToFrame<float, float>(pos);

    if (!pointWithinBounds(img))
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
    const float cosine = (cv::Mat(pos) - w_t_c_).dot(view_vec) / dist;
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

void Frame::computeFundamentalMat(
    cv::Mat& f_mat,
    const FramePtr& frame)
{
    computeFundamentalMat(
        f_mat,
        worldInCameraR(),
        worldInCamerat(),
        frame->cameraInWorldR(),
        frame->cameraInWorldt(),
        this->camera()->intrinsicMatrix(),
        frame->camera()->intrinsicMatrix());
}

void Frame::computeFundamentalMat(
    cv::Mat& f_mat,
    const KeyFramePtr& key_frame)
{
    computeFundamentalMat(
        f_mat,
        worldInCameraR(),
        worldInCamerat(),
        key_frame->cameraInWorldR(),
        key_frame->cameraInWorldt(),
        this->camera()->intrinsicMatrix(),
        key_frame->frame()->camera()->intrinsicMatrix());
}

void Frame::setupFirstFrame() {
    // since this is the first frame it acts as reference for others
    // there we set it as identity matrix
    w_T_c_ = cv::Mat::eye(4, 4, CV_32F);
}

void Frame::setupGrid(const ros::NodeHandle& nh)
{
    std::string prefix = "/orb_slam/tracker/";
    nh.getParam(prefix + "grid_rows", grid_rows_);
    nh.getParam(prefix + "grid_cols", grid_cols_);
    grid_size_x_ = camera_->undistWidth() / grid_cols_;
    grid_size_y_ = camera_->undistHeight() / grid_rows_;
}

void Frame::computeBow() {
    if(bow_vec_.empty() || feature_vec_.empty()) {
        std::vector<cv::Mat> vec_descriptors;
        utils::matToVectorMat(undist_descriptors_, vec_descriptors);
        // Same as in orb_slam original repository
        // Feature vector associate features with nodes in the 4th level
        // (from leaves up). We assume the vocabulary tree has 6 levels,
        // change the 4 otherwise
        orb_vocabulary_->transform(
            vec_descriptors, bow_vec_, feature_vec_, 4);
    }
}

void Frame::assignFeaturesToGrid(
    std::vector<cv::KeyPoint>& key_points_,
    Grid<std::vector<size_t>>& grid)
{
    // Create an empty grid
    int n_reserve = 0.5f * key_points_.size() / (grid_cols_ * grid_rows_);
    for(unsigned int i = 0; i < grid_cols_; i++)
        for (unsigned int j = 0; j < grid_rows_; j++)
            grid[i][j].reserve(n_reserve);

    // Insert keypoints to grid. If not full, insert this cv::KeyPoint to result
    std::vector<cv::KeyPoint> filt_key_points_;
    int count = 0;
    for (int i = 0; i < key_points_.size(); ++i) {
        const auto& key_point = key_points_[i];
        int row, col;
        if (pointInGrid(key_point, col, row)) {
            filt_key_points_.push_back(key_point);
            // stores indices to filt_key_points_
            grid[col][row].push_back(count++);
        }
    }

    // removes the key points that are not in grid
    key_points_ = filt_key_points_;
}

bool Frame::pointInGrid(
    const cv::KeyPoint& key_point, int& pos_x, int& pos_y)
{
    pos_y = (key_point.pt.y - camera_->minY()) / grid_size_y_;
    pos_x = (key_point.pt.x - camera_->minX()) / grid_size_x_;
    if(
        pos_y < 0 ||
        pos_y >= grid_rows_ ||
        pos_x < 0 ||
        pos_x >= grid_cols_)
    {
        return false;
    }
    return true;
}

void Frame::match(
    const std::shared_ptr<Frame>& ref_frame)
{
    ref_frame_ = ref_frame;
    orb_matcher_->match(
        descriptorsUndist(),
        ref_frame_->descriptorsUndist(),
        matches_);
}

void Frame::matchByBowFeatures(
    const std::shared_ptr<KeyFrame>& ref_key_frame,
    const bool check_orientation,
    const float nn_ratio,
    const bool filter_matches)
{
    ref_key_frame_ = ref_key_frame;
    orb_matcher_->matchByBowFeatures( // 0.7 taken from original orb slam code
        shared_from_this(),
        ref_key_frame,
        matches_,
        check_orientation,
        nn_ratio);
}

void Frame::matchByProjection(
    const std::shared_ptr<Frame>& prev_frame,
    const bool check_orientation,
    const float radius,
    const bool filter_matches)
{
    ref_frame_ = prev_frame;
    orb_matcher_->matchByProjection(
        shared_from_this(),
        prev_frame,
        matches_,
        check_orientation,
        radius);
}

void Frame::matchByProjection(
    const std::vector<MapPointPtr>& map_points,
    const bool compute_track_info,
    const float nn_ratio,
    const float radius,
    const bool filter_matches)
{
    orb_matcher_->matchByProjection(
        shared_from_this(),
        map_points,
        local_matches_,
        nn_ratio,
        radius);
}

} // namespace orb_slam