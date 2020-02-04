/**
 * Implements the Frame class.
 */

#include <orb_slam/frame.h>

namespace orb_slam {

//! static variable definitions
int Frame::id_global_ = 0;
geometry::CameraPtr<float> Frame::camera_;
geometry::ORBExtractorPtr Frame::orb_extractor_;
int Frame::max_key_points_;
int Frame::uniform_key_points_grid_size_;
int Frame::max_key_points_per_grid_;
int Frame::grid_rows_;
int Frame::grid_cols_;

Frame::Frame(
    const ros::Time& time_stamp) :
    time_stamp_(time_stamp)
{
    id_ = id_global_++;
}

Frame::~Frame()
{
}

void Frame::setupUniformKeyPointsExtractor(const ros::NodeHandle& nh)
{
    nh.getParam("extract_uniform_key_points", extract_uniform_key_points_);
    if (extract_uniform_key_points_) {
        nh.getParam("max_key_points", max_key_points_);
        nh.getParam(
            "uniform_key_points_grid_size", uniform_key_points_grid_size_);
        nh.getParam("max_key_points_per_grid", max_key_points_per_grid_);
        grid_rows_ = camera_->height() / uniform_key_points_grid_size_;
        grid_cols_ = camera_->width() / uniform_key_points_grid_size_;
    }
}

void Frame::extractUniformKeyPointsInGrid(
    std::vector<cv::KeyPoint>& key_points)
{
    // Create an empty grid
    static cv::Mat grid = cv::Mat::zeros(grid_rows_, grid_cols_, CV_8UC1);

    // set grid to zero
    grid = cv::Scalar(0);

    // Insert keypoints to grid. If not full, insert this cv::KeyPoint to result
    std::vector<cv::KeyPoint> extracted_key_points;
    int total_count = 0;
    for (const auto& key_point : key_points)
    {
        auto row = ((int) key_point.pt.y) / uniform_key_points_grid_size_;
        auto col = ((int) key_point.pt.x) / uniform_key_points_grid_size_;
        auto& count = grid.at<int>(row, col);
        if (count < max_key_points_per_grid_)
        {
            extracted_key_points.push_back(key_point);
            count = count + 1;
            total_count++;
            if (total_count > max_key_points_)
                break;
        }
    }

    // return
    key_points = extracted_key_points;
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

    if (key_points.empty())
        return;

    // undistort key points so they are the in the correct positions
    camera_->undistortPoints(
        key_points);
        std::vector<cv::KeyPoint>& key_points,
    std::vector<cv::KeyPoint>& undist_key_points,
    cv::Mat& undist_intrinsic_matrix)


    // update the key points so that they are uniformly accumulated over
    // the image
    if (extract_uniform_key_points_)
        extractUniformKeyPointsInGrid(key_points);

}

geometry::MonoCameraPtr<float> MonoFrame::camera()
{
    return std::static_pointer_cast<geometry::MonoCamera<float>>(camera_);
}

} // namespace orb_slam