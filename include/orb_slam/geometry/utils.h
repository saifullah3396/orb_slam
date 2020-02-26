/**
 * Declares the geometry and other utility functions
 */

#pragma once

#include <vector>
#include <numeric>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace orb_slam {

namespace geometry {

/**
 * @brief Normalizes the points based on mean and standard deviation
 *
 * @param points: Input points
 * @param normalized: Output normalized points
 * @param T: Output transformation that performs the normalization operation
 */
void normalizePoints(
    const std::vector<cv::Point2f>& points,
    std::vector<cv::Point2f>& normalized,
    cv::Mat& T);

/**
 * Computes the fundamental matrix using 8-point algorithm.
 *
 * @param f_mat: Output fundamental matrix
 * @param points: Input points in the first frame
 * @param ref_points: Input points in reference frame
 */
void computeFundamentalMat(
    cv::Mat& f_mat,
    const std::vector<cv::Point2f>& points,
    const std::vector<cv::Point2f>& ref_points);

/**
 * Computes the homography matrix using 8-point algorithm.
 *
 * @param h_mat: Output fundamental matrix
 * @param points: Input points in the first frame
 * @param ref_points: Input points in reference frame
 */
void computeHomographyMat(
    cv::Mat& h_mat,
    const std::vector<cv::Point2f> &points,
    const std::vector<cv::Point2f> &ref_points);

void drawEpilines(
    std::vector<cv::Vec3f> lines,
    cv::Mat f_mat,
    cv::Mat image,
    std::vector<cv::Point2f> pts,
    std::vector<bool> inliers);

int descriptorDistance(const cv::Mat &a, const cv::Mat &b);

} // namespace geometry

namespace utils {
    /**
     * Converts a matrix to vector of matrices row-wise
     * @param mat: Matrix
     * @param vec_mat: Outout vector of matrices
     */
    void matToVectorMat(const cv::Mat& mat, std::vector<cv::Mat>& vec_mat);
} // namespace utils

} // namespace orb_slam