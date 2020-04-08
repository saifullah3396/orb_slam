/**
 * Defines the Frame class.
 */

#pragma once

#include <array>
#include <memory>
#include <orb_slam/frame.h>

namespace orb_slam {

class MonoFrame : public Frame {
public:
    /**
     * Constructor
     *
     * @param image: Image assigned to this frame
     * @param time_stamp: Frame time stamp on creation
     */
    MonoFrame(
        const cv_bridge::CvImageConstPtr& image, const ros::Time& time_stamp);

    /**
     * Destructor
     */
    ~MonoFrame();

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

private:
    /**
     * Returns the derived camera class
     */
    geometry::MonoCameraConstPtr<float> camera();

    cv_bridge::CvImageConstPtr image_; // Frame image
};

} // namespace orb_slam