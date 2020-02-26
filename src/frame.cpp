/**
 * Implements the Frame class.
 */

#include <opencv2/highgui/highgui.hpp>
#include <orb_slam/frame.h>
#include <thread>

namespace orb_slam {

//! static variable definitions
int Frame::id_global_ = 0;
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

void Frame::setupFirstFrame() {
    // since this is the first frame it acts as reference for others
    // there we set it as identity matrix
    c_T_w_ = cv::Mat::eye(4, 4, CV_32F);
}

void Frame::setupGrid(const ros::NodeHandle& nh)
{
    std::string prefix = "/orb_slam/tracker/";
    nh.getParam(prefix + "grid_rows", grid_rows_);
    nh.getParam(prefix + "grid_cols", grid_cols_);
    grid_size_x_ = camera_->undistWidth() / grid_cols_;
    grid_size_y_ = camera_->undistHeight() / grid_rows_;
}

/**
 * Computes the bag of words from orb vocabulary and frame features
 */
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
    const std::shared_ptr<Frame>& ref_frame,
    const geometry::OrbMatcherTypes type)
{
    ref_frame_ = ref_frame;
    orb_matcher_->match(shared_from_this(), ref_frame_, matches_, type);
}

MonoFrame::MonoFrame(
    const cv_bridge::CvImageConstPtr& image,
    const ros::Time& time_stamp) :
    Frame(time_stamp),
    image_(image)
{
    grid_.resize(grid_cols_);
    for(unsigned int i = 0; i < grid_cols_; i++)
        grid_[i].resize(grid_rows_);
}

MonoFrame::~MonoFrame()
{
}

void MonoFrame::drawFeatures(cv::Mat& image)
{
    ROS_DEBUG_STREAM("Number of features extracted: " << key_points_.size());
    cv::drawKeypoints(
        image,
        key_points_,
        image,
        cv::Scalar(255, 0, 0),
        cv::DrawMatchesFlags::DEFAULT);
}

void MonoFrame::showImageWithFeatures(
    const std::string& name)
{
    cv::Mat draw_image = image_->image.clone();
    drawFeatures(draw_image);
    cv::imshow(name, draw_image);
}

void MonoFrame::showMatchesWithRef(const std::string& name)
{
    if (!ref_frame_ || matches_.empty()) {
        return;
    }
    cv::Mat image_match;
    drawMatches(
        image_->image,
        undist_key_points_,
        ref_frame_->image()->image,
        ref_frame_->featuresUndist(),
        matches_,
        image_match);
    cv::imshow(name, image_match);
}

void MonoFrame::extractFeatures()
{
    // find orb features in the image
    orb_extractor_->detect(image_->image, key_points_);
    #ifdef HARD_DEBUG
    ROS_DEBUG_STREAM("Number of features extracted: " << key_points_.size());
    showImageWithFeatures("key_points_");
    #endif

    if (key_points_.empty())
        return;

    // undistort key points so they are the in the correct positions
    camera_->undistortPoints(
        key_points_, undist_key_points_);

    #ifdef HARD_DEBUG
    ROS_DEBUG_STREAM(
        "Number of undistorted features: " << undist_key_points_.size());
    showImageWithFeatures("undist_key_points_");
    #endif

    // update the key points so that they are uniformly accumulated over
    // the image. Note that the points are undistorted and the grid is also
    // within non-black region of the undisorted image.
    assignFeaturesToGrid(undist_key_points_, grid_);

    ROS_DEBUG_STREAM("Finding orb features...");
    // find the orb descriptors for undistorted points
    orb_extractor_->compute(
        image_->image, undist_key_points_, undist_descriptors_);
}

geometry::MonoCameraConstPtr<float> MonoFrame::camera()
{
    return std::static_pointer_cast<const geometry::MonoCamera<float>>(camera_);
}

RGBDFrame::RGBDFrame(
    const cv_bridge::CvImageConstPtr& image,
    const cv_bridge::CvImageConstPtr& depth,
    const ros::Time& time_stamp) :
    Frame(time_stamp),
    image_(image),
    depth_(depth)
{
    grid_.resize(grid_cols_);
    for(unsigned int i = 0; i < grid_cols_; i++)
        grid_[i].resize(grid_rows_);
}

RGBDFrame::~RGBDFrame()
{
}

void RGBDFrame::drawFeatures(cv::Mat& image)
{
    ROS_DEBUG_STREAM("Number of features extracted: " << key_points_.size());
    cv::drawKeypoints(
        image,
        key_points_,
        image,
        cv::Scalar(255, 0, 0),
        cv::DrawMatchesFlags::DEFAULT);
}

void RGBDFrame::showImageWithFeatures(
    const std::string& name)
{
    cv::Mat draw_image = image_->image.clone();
    drawFeatures(draw_image);
    cv::imshow(name, draw_image);
}

void RGBDFrame::showMatchesWithRef(const std::string& name)
{
    if (!ref_frame_ || matches_.empty()) {
        return;
    }
    cv::Mat image_match;
    drawMatches(
        image_->image,
        undist_key_points_,
        ref_frame_->image()->image,
        ref_frame_->featuresUndist(),
        matches_,
        image_match);
    cv::imshow(name, image_match);
}

void RGBDFrame::extractFeatures()
{
    // convert image to gray
    if (image_->image.channels() == 3) {
        if(rgb_) // rgb or bgr
            cv::cvtColor(image_->image, gray_image_, CV_RGB2GRAY);
        else
            cv::cvtColor(image_->image, gray_image_, CV_BGR2GRAY);
    } else {
        gray_image_ = image_->image;
    }

    // find orb features in the image
    orb_extractor_->detect(gray_image_, key_points_);
    #ifdef HARD_DEBUG
    ROS_DEBUG_STREAM("Number of features extracted: " << key_points_.size());
    showImageWithFeatures("key_points_");
    #endif

    if (key_points_.empty())
        return;

    // undistort key points so they are the in the correct positions
    camera_->undistortPoints(
        key_points_, undist_key_points_);

    #ifdef HARD_DEBUG
    ROS_DEBUG_STREAM(
        "Number of undistorted features: " << undist_key_points_.size());
    showImageWithFeatures("undist_key_points_");
    #endif

    // update the key points so that they are uniformly accumulated over
    // the image. Note that the points are undistorted and the grid is also
    // within non-black region of the undisorted image.
    assignFeaturesToGrid(undist_key_points_, grid_);

    // assign depth for each key point
    const auto n = undist_key_points_.size();
    undist_key_point_depths_ = std::vector<float>(n, -1);
    for (int i = 0; i < n; ++i) {
        const auto& kp = undist_key_points_[i];
        const auto depth = depth_->image.at<float>(kp.pt.y, kp.pt.x);
        if (depth > 0) {
            undist_key_point_depths_[i] = depth;
        }
    }

    ROS_DEBUG_STREAM("Finding orb features...");
    // find the orb descriptors for undistorted points
    orb_extractor_->compute(
        gray_image_, undist_key_points_, undist_descriptors_);
}

geometry::RGBDCameraConstPtr<float> RGBDFrame::camera()
{
    return std::static_pointer_cast<const geometry::RGBDCamera<float>>(camera_);
}

} // namespace orb_slam