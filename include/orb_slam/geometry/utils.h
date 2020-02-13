/**
 * Defines the geometry utility functions
 */

#include <vector>
#include <numeric>
#include <opencv2/core/core.hpp>

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
    cv::Mat& T)
{
    const auto n = points.size();
    normalized.resize(n);
    auto sum =
        std::accumulate(points.begin(), points.end(), cv::Point2f(0.f, 0.f));
    auto n_inv = 1.0 / n;
    auto mean_x = sum.x * n_inv;
    auto mean_y = sum.y * n_inv;

    // use same normalization done in original paper
    // in place of 's' parameter, there are 'sx', and 'sy' parameters.
    // see https://cs.adelaide.edu.au/~wojtek/papers/pami-nals2.pdf
    double scale = 0.0;
    for (int i = 0; i < n; ++i) {
        const auto& p = points[i];
        normalized[i] = cv::Point2f(p.x - mean_x, p.y - mean_y);
        scale += cv::norm(normalized[i]);
    }
    scale *= n_inv;
    scale = std::sqrt(2.0) / scale;

    for (int i = 0; i < n; ++i) {
        normalized[i] *= scale;
    }

    T = cv::Mat::eye(3, 3, CV_32F);
    T.at<float>(0,0) = scale;
    T.at<float>(1,1) = scale;
    T.at<float>(0,2) = -mean_x * scale;
    T.at<float>(1,2) = -mean_y * scale;
}

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
    const std::vector<cv::Point2f>& ref_points)
{
    const int n = points.size();
    // see http://www.cs.cmu.edu/~16385/s17/Slides/12.4_8Point_Algorithm.pdf
    cv::Mat A(n, 9,CV_32F);

    // create an n x 9 matrix filled with the points
    // step 1: Construct A
    for(int i = 0; i< n; i++) {
        const auto& u1 = points[i].x;
        const auto& v1 = points[i].y;
        const auto& u2 = ref_points[i].x;
        const auto& v2 = ref_points[i].y;

        A.at<float>(i,0) = u2*u1;
        A.at<float>(i,1) = u2*v1;
        A.at<float>(i,2) = u2;
        A.at<float>(i,3) = v2*u1;
        A.at<float>(i,4) = v2*v1;
        A.at<float>(i,5) = v2;
        A.at<float>(i,6) = u1;
        A.at<float>(i,7) = v1;
        A.at<float>(i,8) = 1;
    }

    // compute SVD and take the last row of vt
    // step 2: compute SVD of A'A
    cv::Mat u, w, vt;
    cv::SVDecomp(A,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    // step 3: Entries of F are elements of column v for least singular values
    cv::Mat Fpre = vt.row(8).reshape(0, 3);

    // step 4: Enforce constraint of rank 2
    cv::SVDecomp(Fpre,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
    // set the last eigen value to 0. This make the matrix of rank 2
    w.at<float>(2) = 0;
    // remake the matrix with rank 2
    f_mat = u * cv::Mat::diag(w) * vt;
}

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
    const std::vector<cv::Point2f> &ref_points)
{
    const int n = points.size();

    // construct A matrix
    cv::Mat A(2 * n, 9, CV_32F);
    for(int i = 0; i < n; i++) {
        const float u1 = points[i].x;
        const float v1 = points[i].y;
        const float u2 = ref_points[i].x;
        const float v2 = ref_points[i].y;

        A.at<float>(2*i,0) = 0.0;
        A.at<float>(2*i,1) = 0.0;
        A.at<float>(2*i,2) = 0.0;
        A.at<float>(2*i,3) = -u1;
        A.at<float>(2*i,4) = -v1;
        A.at<float>(2*i,5) = -1;
        A.at<float>(2*i,6) = v2*u1;
        A.at<float>(2*i,7) = v2*v1;
        A.at<float>(2*i,8) = v2;

        A.at<float>(2*i+1,0) = u1;
        A.at<float>(2*i+1,1) = v1;
        A.at<float>(2*i+1,2) = 1;
        A.at<float>(2*i+1,3) = 0.0;
        A.at<float>(2*i+1,4) = 0.0;
        A.at<float>(2*i+1,5) = 0.0;
        A.at<float>(2*i+1,6) = -u2*u1;
        A.at<float>(2*i+1,7) = -u2*v1;
        A.at<float>(2*i+1,8) = -u2;

    }

    // perform SVD
    cv::Mat u, w, vt;
    cv::SVDecomp(A,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    // entries of F are elements of column v for least singular values
    h_mat = vt.row(8).reshape(0, 3);
}

} // namespace geometry

} // namespace orb_slam