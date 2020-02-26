/**
 * This file declares the ORBExtractor class.
 *
 * @author <A href="mailto:saifullah3396@gmail.com">Saifullah</A>
 */

#pragma once

#include <iostream>
#include <ros/ros.h>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d.hpp>

namespace orb_slam
{

namespace geometry
{

/**
 * @struct ORBExtractor
 * @brief The class that is used to extract orb features from an image
 */
class ORBExtractor
{
public:
    ORBExtractor(const ros::NodeHandle& nh): nh_(nh) {
        std::string prefix = "orb_slam/orb_extractor/";
        nh_.param<int>(prefix + "n_key_points", n_key_points_, 8000);
        nh_.param<float>(prefix + "scale_factor", scale_factor_, 1.200000048F);
        nh_.param<int>(prefix + "level_pyramid", level_pyramid_, 4);
        nh_.param<int>(prefix + "edge_threshold", edge_threshold_, 31);
        nh_.param<int>(prefix + "first_level", first_level_, 0);
        nh_.param<int>(prefix + "wta_k", wta_k_, 2);
        nh_.param<int>(prefix + "patch_size", patch_size_, 31);
        nh_.param<int>(prefix + "score_threshold", score_threshold_, 20);

        scale_factors_.resize(level_pyramid_);
        scale_sigma_sqrd_.resize(level_pyramid_);
        inv_scale_factors_.resize(level_pyramid_);
        inv_scale_sigma_sqrd_.resize(level_pyramid_);
        scale_factors_[0] = 1.0f;
        scale_sigma_sqrd_[0] = 1.0f;
        inv_scale_factors_[0] = 1.0f / scale_factors_[0];
        inv_scale_sigma_sqrd_[0] = 1.0f / scale_sigma_sqrd_[0];
        for (int i = 1; i < level_pyramid_; ++i) {
            scale_factors_[i] = scale_factors_[i-1] * scale_factor_;
            inv_scale_factors_[i] = 1.0f / scale_factors_[i];

            scale_sigma_sqrd_[i] = scale_factors_[i] * scale_factors_[i];
            inv_scale_sigma_sqrd_[i] = 1.0f / scale_sigma_sqrd_[i];
        }

        // initialize the detector
        cv_orb_detector_ =
            cv::ORB::create(
                n_key_points_,
                scale_factor_,
                level_pyramid_,
                edge_threshold_,
                first_level_,
                wta_k_,
                score_type_,
                patch_size_,
                score_threshold_);

        // initialize the descriptor
        cv_orb_descriptor_ =
            cv::ORB::create(
                n_key_points_,
                scale_factor_,
                level_pyramid_,
                edge_threshold_,
                first_level_,
                wta_k_,
                score_type_,
                patch_size_,
                score_threshold_);
    }

    /**
     * Getters
     */
    const std::vector<float>& scaleFactors() const { return scale_factors_; }
    const int& levels() const { return level_pyramid_; }
    const std::vector<float>& scaleSigmas() const { return scale_sigma_sqrd_; }
    const std::vector<float>& invScaleSigmas() const
        { return inv_scale_sigma_sqrd_; }

    /**
     * Extracts orb features from the input image
     *
     * @param image: Input image
     * @param key_points: Output feature points
     */
    void detect(
        const cv::Mat& image, std::vector<cv::KeyPoint>& key_points) const
    {
        cv_orb_detector_->detect(image, key_points);
    }

    /**
     * Computres orb descriptors from the input image and key points
     *
     * @param image: Input image
     * @param key_points: Input feature points
     * @param descriptors: Output descriptors
     */
    void compute(
        const cv::Mat& image,
        std::vector<cv::KeyPoint>& key_points,
        cv::Mat& descriptors) const
    {
        cv_orb_descriptor_->compute(image, key_points, descriptors);
    }

    ~ORBExtractor() {

    }

private:
    //! ros node handle for reading parameters
    ros::NodeHandle nh_;

    //! orb extractor parameters
    int n_key_points_; // number of key points to extract
    float scale_factor_; // feature scale factor
    int level_pyramid_; // image pyramid level
    int edge_threshold_ = {31}; // edge threshold
    int first_level_ = {0}; // first level
    int wta_k_ = {2}; // wta_k
    int score_type_ = {cv::ORB::HARRIS_SCORE}; // score type
    int patch_size_ = {31}; // patch size
    int score_threshold_; // score threshold

    std::vector<float> scale_factors_;
    std::vector<float> inv_scale_factors_;
    std::vector<float> scale_sigma_sqrd_;
    std::vector<float> inv_scale_sigma_sqrd_;

    //! opencv orb extractors
    cv::Ptr<cv::ORB> cv_orb_detector_;
    cv::Ptr<cv::ORB> cv_orb_descriptor_;
};

using ORBExtractorPtr = std::shared_ptr<ORBExtractor>;
using ORBExtractorConstPtr = std::shared_ptr<const ORBExtractor>;

} // namespace geometry

} // namespace orb_slam