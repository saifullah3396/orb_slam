/**
 * Defines the Frame class.
 */

#pragma once

#include <array>
#include <memory>
#include <orb_slam/frame.h>

namespace orb_slam {

class RGBDFrame : public Frame {
public:
    /**
     * Constructor
     *
     * @param image: Image assigned to this frame
     * @param depth: Image depth assigned to this frame
     * @param time_stamp: Frame time stamp on creation
     */
    RGBDFrame(
        const cv_bridge::CvImageConstPtr& image,
        const cv_bridge::CvImageConstPtr& depth,
        const ros::Time& time_stamp);

    /**
     * Destructor
     */
    ~RGBDFrame();

    /**
     * Extracts orb features from the frame image
     */
    virtual void extractFeatures();

    void drawFeatures(cv::Mat& image) const;
    void showImageWithFeatures(const std::string& name) const;
    void showMatchesWithRef(const std::string& name, const size_t n = 0) const;

    /**
     * Getters
     */
    const cv_bridge::CvImageConstPtr& image() const { return image_; }
    const cv_bridge::CvImageConstPtr& depth() const { return depth_; }
    const std::vector<float>& featureDepthsUndist() const {
        return undist_key_point_depths_;
    }

private:
    /**
     * Returns the derived camera class
     */
    geometry::RGBDCameraConstPtr<float> camera();

    cv_bridge::CvImageConstPtr image_; // Frame image
    cv_bridge::CvImageConstPtr depth_; // Frame image depth
    cv::Mat gray_image_; // gray_scale image
    bool rgb_ = {false};
    std::vector<float> undist_key_point_depths_;
};

} // namespace orb_slam