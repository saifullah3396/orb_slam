/**
 * Implements the geometry and other utility functions
 */

#include <orb_slam/geometry/utils.h>

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
        const auto& u1 = ref_points[i].x;
        const auto& v1 = ref_points[i].y;
        const auto& u2 = points[i].x;
        const auto& v2 = points[i].y;

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
        const float u1 = ref_points[i].x;
        const float v1 = ref_points[i].y;
        const float u2 = points[i].x;
        const float v2 = points[i].y;

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

    // entries of H are elements of column v for least singular values
    h_mat = vt.row(8).reshape(0, 3);
}

void drawEpilines(
    std::vector<cv::Vec3f> lines,
    cv::Mat f_mat,
    cv::Mat image,
    std::vector<cv::Point2f> pts,
    std::vector<bool> inliers)
{
    cv::Mat p = cv::Mat_<float>(3, 1);
    for(std::size_t i = 0; i < lines.size(); ++i) {
        if (!inliers[i]) continue;
        auto l = lines.at(i);
        float a = l.val[0];
        float b = l.val[1];
        float c = l.val[2];

        float x0,y0,x1,y1;
        x0=0;
        y0=(-c-a*x0)/b;
        x1=image.cols;
        y1=(-c-a*x1)/b;

        cv::circle(image, pts[i], 5, cv::Scalar(0, 255, 0), -1);
        cv::line(image, cvPoint(x0,y0), cvPoint(x1,y1), cv::Scalar(0,255,0), 1);
    }
    cv::imshow("image:", image);
    cv::waitKey(0);
}

int descriptorDistance(const cv::Mat &a, const cv::Mat &b)
{
    const int *pa = a.ptr<int32_t>();
    const int *pb = b.ptr<int32_t>();

    int dist=0;

    for(int i=0; i<8; i++, pa++, pb++)
    {
        unsigned  int v = *pa ^ *pb;
        v = v - ((v >> 1) & 0x55555555);
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }

    return dist;
}

void computeThreeMaxima(
    std::vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3)
{
    int max1=0;
    int max2=0;
    int max3=0;

    for (int i = 0; i < L; i++)
    {
        const int s = histo[i].size();
        if(s > max1) {
            max3=max2;
            max2=max1;
            max1=s;
            ind3=ind2;
            ind2=ind1;
            ind1=i;
        } else if(s>max2) {
            max3=max2;
            max2=s;
            ind3=ind2;
            ind2=i;
        } else if(s>max3) {
            max3=s;
            ind3=i;
        }
    }

    if(max2 < 0.1f * (float) max1) {
        ind2=-1;
        ind3=-1;
    } else if(max3 < 0.1f * (float) max1) {
        ind3=-1;
    }
}

} // namespace geometry

namespace utils {
    /**
     * Converts a matrix to vector of matrices row-wise
     * @param mat: Matrix
     * @param vec_mat: Outout vector of matrices
     */
    void matToVectorMat(const cv::Mat& mat, std::vector<cv::Mat>& vec_mat) {
        vec_mat.resize(mat.rows);
        for (int i = 0; i < mat.rows; ++i) {
            vec_mat[i] = mat.row(i);
        }
    }

} // namespace utils

} // namespace orb_slam