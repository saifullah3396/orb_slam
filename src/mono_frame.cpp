/**
 * Implements the Frame class.
 */

#include <opencv2/highgui/highgui.hpp>
#include <orb_slam/mono_frame.h>
#include <orb_slam/key_frame.h>
#include <orb_slam/map_point.h>
#include <thread>

namespace orb_slam {

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

void MonoFrame::drawFeatures(cv::Mat& image) const
{
    cv::drawKeypoints(
        image,
        key_points_,
        image,
        cv::Scalar(255, 0, 0),
        cv::DrawMatchesFlags::DEFAULT);
}

void MonoFrame::showImageWithFeatures(
    const std::string& name) const
{
    cv::Mat draw_image = image_->image.clone();
    drawFeatures(draw_image);
    cv::imshow(name, draw_image);
}

void MonoFrame::showMatchesWithRef(const std::string& name, const size_t n) const
{
    if (!ref_frame_ || matches_.empty()) {
        return;
    }
    cv::Mat image_match;
    std::vector<char> matches_mask;
    if (n > 0) {
        matches_mask = std::vector<char>(undist_key_points_.size(), 0);
        for (int i = 0; i < n; ++i) {
            matches_mask[i] = 1;
        }
    }
    drawMatches(
        image_->image,
        undist_key_points_,
        ref_frame_->image()->image,
        ref_frame_->featuresUndist(),
        matches_,
        image_match,
        cv::Scalar::all(-1),
        cv::Scalar::all(-1),
        matches_mask);
    cv::imshow(name, image_match);
}

void MonoFrame::extractFeatures()
{
    // find orb features in the image
    orb_extractor_->detect(image_->image, key_points_);
    #ifdef HARD_DEBUG
    //ROS_DEBUG_STREAM_NAMED(name_tag_, "Number of features extracted: " << key_points_.size());
    showImageWithFeatures("key_points_");
    #endif

    if (key_points_.empty())
        return;

    // undistort key points so they are the in the correct positions
    camera_->undistortPoints(
        key_points_, undist_key_points_);

    #ifdef HARD_DEBUG
    //ROS_DEBUG_STREAM_NAMED(name_tag_,
        "Number of undistorted features: " << undist_key_points_.size());
    showImageWithFeatures("undist_key_points_");
    #endif

    // update the key points so that they are uniformly accumulated over
    // the image. Note that the points are undistorted and the grid is also
    // within non-black region of the undisorted image.
    assignFeaturesToGrid(undist_key_points_, grid_);

    //ROS_DEBUG_STREAM_NAMED(name_tag_, "Finding orb features...");

    // find the orb descriptors for undistorted points
    orb_extractor_->compute(
        image_->image, undist_key_points_, undist_descriptors_);

    //ROS_DEBUG_STREAM_NAMED(name_tag_, "Resizing frame obs map and outliers");
    // resize the map equal to the feature points
    const auto n = undist_key_points_.size();
    resizeMap(n);
    outliers_.resize(n);
}

geometry::MonoCameraConstPtr<float> MonoFrame::camera()
{
    return std::static_pointer_cast<const geometry::MonoCamera<float>>(camera_);
}

} // namespace orb_slam