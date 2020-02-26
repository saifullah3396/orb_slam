/**
 * Declares the Initializer class.
 */

#pragma once

#include <ros/ros.h>
#include <random>
#include <thread>

namespace orb_slam {

namespace geometry {
    template <typename T>
    class Camera;
    template <typename T>
    using CameraPtr = std::shared_ptr<Camera<T>>;
}

class Frame;
using FramePtr = std::shared_ptr<Frame>;

/**
 * The initializer used in monocular slam.
 */
class Initializer
{
public:
    Initializer(
        const FramePtr& ref_frame,
        const geometry::CameraPtr<float>& camera,
        double sigma = 1.0,
        int iterations = 200);
    ~Initializer();

    /**
     * Tries to initialize the mono-slam with a new frame with respect to the
     * reference frame. This is based on the orb-slam repository.
     *
     * @param frame: The second frame that is matched with reference frame
     * @param best_rot_mat: The best rotation matrix from this frame to
     *     reference frame
     * @param best_trans_mat: The best translation matrix from this frame to
     *     reference frame
     * @param inlier_points: Inlier matched points
     * @param inlier_ref_points: Inlier matched reference points
     * @param inlier_points_3d: Points found after triangulation with R-t found
     * @param inliers_mask: A bool mask for inlier points from input points
     * @returns True if initialized successfully, false otherwise
     */
    bool tryToInitialize(
        const FramePtr& frame,
        cv::Mat& best_rot_mat,
        cv::Mat& best_trans_mat,
        std::vector<cv::Point2d>& inlier_points,
        std::vector<cv::Point2d>& inlier_ref_points,
        std::vector<cv::Point3d>& inlier_points_3d, // 3-dimensional
        std::vector<size_t>& inliers_idxs);

    /**
     * Getters
     */
    cv::Mat getFundamentalMat() { return fundamental_mat_; }
    cv::Mat getHomographyMat() { return homography_mat_; }
    std::vector<bool> getInliersF() { return inliers_f_; }
    std::vector<bool> getInliersH() { return inliers_h_; }

    /**
     * Setters
     */
    void setRefPoints(const std::vector<cv::Point2f>& ref_points)
        { ref_points_ = ref_points; }
    void setPoints(const std::vector<cv::Point2f>& points)
        { points_ = points; }

    /**
     * Finds the fundamental and homography matrices.
     */
    void findFundamentalAndHomography();

    /**
     * Finds the rotation and translation matrix of frame in reference frame
     * from homography matrix.
     *
     * @param inlier_points: Inlier points found from homography.
     * @param inlier_ref_points: Inlier reference points found from homography.
     * @param R: Output rotation matrix
     * @param t: Output translation matrix
     * @returns False on failure
     */
    bool findRtWithHomography(
        const std::vector<cv::Point2d>& inlier_points,
        const std::vector<cv::Point2d>& inlier_ref_points,
        cv::Mat& R,
        cv::Mat& t);

    /**
     * Finds the rotation and translation matrix of frame in reference frame
     * from fundamental matrix.
     *
     * @param inlier_points: Inlier points found from fundamental.
     * @param inlier_ref_points: Inlier reference points found from fundamental.
     * @param R: Output rotation matrix
     * @param t: Output translation matrix
     * @returns False on failure
     */
    bool findRtWithFundamental(
        const std::vector<cv::Point2d>& inlier_points,
        const std::vector<cv::Point2d>& inlier_ref_points,
        cv::Mat& R,
        cv::Mat& t);

    /**
     * Find the triangulated points for given R, t and points in two frames.
     * @param inlier_points: Inlier points found from fundamental.
     * @param inlier_ref_points: Inlier reference points found from fundamental.
     * @param R: Rotation matrix from 1 to 2
     * @param t: Translation matrix from 1 to 2
     * @param inlier_points_3d: Output points in 3d
     * @returns False on failure
     */
    bool triangulatePoints(
        const std::vector<cv::Point2d>& inlier_points,
        const std::vector<cv::Point2d>& inlier_ref_points,
        const cv::Mat& R,
        const cv::Mat& t,
        std::vector<cv::Point3d>& inlier_points_3d);

private:
    /**
     * Finds the fundamental matrix according to the orb-slam paper.
     *
     * @param points: Input points in the first frame
     * @param ref_points: Input points in reference frame
     * @param points_norm: Normalized points in the first frame
     * @param ref_points_norm: Normalized points in reference frame
     * @param T: The normalization matrix for the frame
     * @param ref_T: The normalization matrix for the reference frame
     */
    void findFundamentalMat(
        const std::vector<cv::Point2f>& points,
        const std::vector<cv::Point2f>& ref_points,
        const std::vector<cv::Point2f>& points_norm,
        const std::vector<cv::Point2f>& ref_points_norm,
        const cv::Mat& T,
        const cv::Mat& ref_T);

    /**
     * Checks the score of the fundamental matrix as done in orb-slam paper.
     * Taken directly from orb-slam repository.
     *
     * @param f_mat: Fundamental matrix
     * @param points: Points of current frame
     * @param ref_points: Points in reference frame
     * @param inliers: Inliers as found while computing score
     */
    double checkFundamentalScore(
        const cv::Mat& f_mat,
        const std::vector<cv::Point2f>& points,
        const std::vector<cv::Point2f>& ref_points,
        std::vector<bool>& inliers);

    /**
     * Finds the homography matrix according to the orb-slam paper.
     *
     * @param points: Input points in the first frame
     * @param ref_points: Input points in reference frame
     * @param points_norm: Normalized points in the first frame
     * @param ref_points_norm: Normalized points in reference frame
     * @param T: The normalization matrix for the frame
     * @param ref_T: The normalization matrix for the reference frame
     */
    void findHomographyMat(
        const std::vector<cv::Point2f>& points,
        const std::vector<cv::Point2f>& ref_points,
        const std::vector<cv::Point2f>& points_norm,
        const std::vector<cv::Point2f>& ref_points_norm,
        const cv::Mat& T,
        const cv::Mat& ref_T);

    /**
     * Checks the score of the homography matrix as done in orb-slam paper.
     * Taken directly from orb-slam repository.
     *
     * @param points: Points of current frame
     * @param ref_points: Points in reference frame
     * @param h_mat: Homography matrix
     * @param h_mat_inv: The inverse homography matrix
     * @param inliers: Inliers as found while computing score
     */
    double checkHomographyScore(
        const std::vector<cv::Point2f>& points,
        const std::vector<cv::Point2f>& ref_points,
        const cv::Mat& h_mat,
        const cv::Mat& h_mat_inv,
        std::vector<bool>& inliers);

    FramePtr frame_; // The first frame to initialize against
    FramePtr ref_frame_; // The initial reference frame
    std::vector<cv::Point2f> points_; // Points in first frame
    std::vector<cv::Point2f> ref_points_; // Points in reference frame

    double sigma_; // std dev?
    double sigma_squared_; // variance?
    int iterations_; // total iterations
    // a vector of 8-points sets made of randomized indices for feature point
    // matching
    std::vector<std::vector<size_t>> ransac_sets_;

    // fundamental and essential matrices required for 2d-2d epipolar geometry
    cv::Mat fundamental_mat_; // fundamental matrix
    std::vector<bool> inliers_f_; // inliers found by RANSAC while finding fmat.
    double f_score_; // fundamental matrix score as in orb-slam paper

    // homography matrix required for finding R-t directly from 2d-2d points
    cv::Mat homography_mat_; // homography matrix
    std::vector<bool> inliers_h_; // inliers found by RANSAC while finding hmat.
    double h_score_; // homography matrix score as in orb-slam paper

    geometry::CameraPtr<float> camera_;
};

class Initializer;
using InitializerPtr = std::unique_ptr<Initializer>;

} // namespace orb_slam