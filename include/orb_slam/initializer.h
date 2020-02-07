/**
 * Declares the Initializer class.
 */

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
     */
    void tryToInitialize(
        const FramePtr& frame, cv::Mat& best_rot_mat, cv::Mat& best_trans_mat);

    /**
     * setters
     */
    static void setCamera(const geometry::CameraPtr<float>& camera)
        { camera_ = camera; }

private:
    /**
     * Finds the fundamental matrix according to the orb-slam paper.
     *
     * @param points: Input points in the first frame
     * @param ref_points: Input points in reference frame
     * @param T: The normalization matrix for the frame
     * @param ref_T: The normalization matrix for the reference frame
     */
    void findFundamentalMat(
        const std::vector<cv::Point2f>& points,
        const std::vector<cv::Point2f>& ref_points,
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
     * @param T: The normalization matrix for the frame
     * @param ref_T: The normalization matrix for the reference frame
     */
    void findHomographyMat(
        const std::vector<cv::Point2f>& points,
        const std::vector<cv::Point2f>& ref_points,
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

    static geometry::CameraPtr<float> camera_;
};

class Initializer;
using InitializerPtr = std::unique_ptr<Initializer>;

} // namespace orb_slam