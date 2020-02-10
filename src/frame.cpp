/**
 * Implements the Frame class.
 */

#ifdef DEBUG
#include <opencv2/highgui/highgui.hpp>
#endif
#include <orb_slam/frame.h>
#include <thread>

namespace orb_slam {

//! static variable definitions
int Frame::id_global_ = 0;
geometry::CameraPtr<float> Frame::camera_;
geometry::ORBExtractorPtr Frame::orb_extractor_;
geometry::ORBMatcherPtr Frame::orb_matcher_;

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

void Frame::setupFirstFrame() {
    // since this is the first frame it acts as reference for others
    // there we set it as identity matrix
    c_T_w_ = cv::Mat::eye(4, 4, CV_64F);
}

void Frame::setupGrid(const ros::NodeHandle& nh)
{
    std::string prefix = "/orb_slam/tracker/";
    nh.getParam(prefix + "grid_rows", grid_rows_);
    nh.getParam(prefix + "grid_cols", grid_cols_);
    grid_size_x_ = camera_->undistWidth() / grid_cols_;
    grid_size_y_ = camera_->undistHeight() / grid_rows_;
}

void Frame::assignFeaturesToGrid(
    std::vector<cv::KeyPoint>& key_points,
    Grid<std::vector<size_t>>& grid)
{
    // Create an empty grid
    int n_reserve = 0.5f * key_points.size() / (grid_cols_ * grid_rows_);
    for(unsigned int i = 0; i < grid_cols_; i++)
        for (unsigned int j = 0; j < grid_rows_; j++)
            grid[i][j].reserve(n_reserve);

    // Insert keypoints to grid. If not full, insert this cv::KeyPoint to result
    std::vector<cv::KeyPoint> filt_key_points;
    int count = 0;
    for (int i = 0; i < key_points.size(); ++i) {
        const auto& key_point = key_points[i];
        int row, col;
        if (pointInGrid(key_point, row, col)) {
            filt_key_points.push_back(key_point);
            // stores indices to filt_key_points
            grid[row][col].push_back(count++);
        }
    }

    // removes the key points that are not in grid
    key_points = filt_key_points;
}

bool Frame::pointInGrid(
    const cv::KeyPoint& key_point, int& pos_x, int& pos_y)
{
    auto row = (int)((key_point.pt.y - camera_->minY()) / grid_size_y_);
    auto col = (int)((key_point.pt.x - camera_->minX()) / grid_size_x_);
    if(
        row < 0 ||
        row >= grid_rows_ ||
        col < 0 ||
        col >= grid_cols_)
    {
        return false;
    }
    return true;
}

void Frame::match(const std::shared_ptr<Frame>& ref_frame)
{
    ref_frame_ = ref_frame;
    orb_matcher_->match(FramePtr(this), ref_frame_, matches_);
}

MonoFrame::MonoFrame(const ros::Time& time_stamp) : Frame(time_stamp)
{
}

MonoFrame::~MonoFrame()
{
}

void MonoFrame::extractFeatures() {
    // find orb features in the image
    orb_extractor_->detect(camera_->image(), key_points);
    #ifdef DEBUG
    ROS_DEBUG_STREAM("Number of features extracted: " << key_points.size());
    cv::Mat draw_image = camera_->image().clone();
    cv::drawKeypoints(
        camera_->image(),
        key_points,
        draw_image,
        cv::Scalar(255, 0, 0),
        cv::DrawMatchesFlags::DEFAULT);
    cv::imshow("key_points", draw_image);
    cv::waitKey(0);
    #endif

    if (key_points.empty())
        return;

    // undistort key points so they are the in the correct positions
    camera_->undistortPoints(
        key_points, undist_key_points);

    #ifdef DEBUG
    ROS_DEBUG_STREAM(
        "Number of undistorted features: " << undist_key_points.size());
    ROS_DEBUG_STREAM(
        "undist_intrinsic_matrix:\n" << undist_intrinsic_matrix);
    cv::drawKeypoints(
        draw_image,
        undist_key_points,
        draw_image,
        cv::Scalar(0, 0, 255),
        cv::DrawMatchesFlags::DEFAULT);
    cv::imshow("undist_key_points", draw_image);
    cv::waitKey(0);
    #endif

    // update the key points so that they are uniformly accumulated over
    // the image. Note that the points are undistorted and the grid is also
    // within non-black region of the undisorted image.
    assignFeaturesToGrid(undist_key_points, grid_);
    ROS_INFO("5");

    // find the orb descriptors for undistorted points
    orb_extractor_->compute(
        camera_->image(), undist_key_points, undist_descriptors_);
}

geometry::MonoCameraPtr<float> MonoFrame::camera()
{
    return std::static_pointer_cast<geometry::MonoCamera<float>>(camera_);
}

} // namespace orb_slam