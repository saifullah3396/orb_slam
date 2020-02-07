/**
 * Implements the Initializer class.
 */

#include <orb_slam/frame.h>
#include <orb_slam/initializer.h>
#include <orb_slam/geometry/camera.h>
#include <orb_slam/geometry/utils.h>

namespace orb_slam {

// static variable definitions
geometry::CameraPtr<float> Initializer::camera_;

Initializer::Initializer(
    const FramePtr& ref_frame,
    double sigma,
    int iterations) :
    ref_frame_(ref_frame), sigma_(sigma), iterations_(iterations)
{
    sigma_squared_ = sigma * sigma;
}

Initializer::~Initializer() {

}

void Initializer::tryToInitialize(
    const FramePtr& frame, cv::Mat& best_rot_mat, cv::Mat& best_trans_mat)
{
    frame_ = frame;
    if (!ref_frame_) {
        ROS_WARN("No reference frame assigned to estimate relative pose from");
    }

    const auto n = frame->nMatches();
    const auto& matches = frame->matches();
    const auto& undist_key_points = frame->featuresUndist();
    const auto& undist_ref_key_points = ref_frame_->featuresUndist();
    std::vector<cv::Point2f> ref_points(n);
    std::vector<cv::Point2f> points(n);
    for (int i = 0; i < n; ++i) {
        ref_points[i] = undist_ref_key_points[matches[i].trainIdx].pt;
        points[i] = undist_key_points[matches[i].queryIdx].pt;
    }

    // generate sets of 8 points for each RANSAC iteration
    // since F and H are to be compared, same set of randomized points are
    // used for computing both matrices. As in paper, this is done to ensure
    // the procedure is homogenous.
    std::vector<size_t> all_indices;
    all_indices.reserve(n);
    std::generate(
        all_indices.begin(),
        all_indices.end(), [value = 0]() mutable { return value++; });

    ransac_sets_ =
        std::vector<std::vector<size_t>>(
            iterations_, std::vector<size_t>(8, 0));

    for (int it = 0; it < iterations_; it++) {
        std::shuffle(
            all_indices.begin(),
            all_indices.end(),
            std::mt19937{std::random_device{}()});
        ransac_sets_.push_back(
            std::vector<size_t>(
                all_indices.begin(), all_indices.begin() + 8));
    }

    // see http://www.cs.cmu.edu/~16385/s17/Slides/12.4_8Point_Algorithm.pdf
    // for details on fundamental matrix computation

    // normalize the points
    std::vector<cv::Point2f> points_norm, ref_points_norm;
    cv::Mat T, ref_T;
    geometry::normalizePoints(points, points_norm, T);
    geometry::normalizePoints(ref_points, ref_points_norm, ref_T);

    // find fundamental matrix using RANSAC
    std::thread computeF(
        [&] {
            findFundamentalMat(points_norm, ref_points_norm, T, ref_T); });

    // find homography matrix
    std::thread computeH(
        [&] {
            findHomographyMat(points_norm, ref_points_norm, T, ref_T); });

    // wait for both threads to finish...
    computeF.join();
    computeH.join();

    std::vector<cv::Point2f> inlier_points, inlier_ref_points;
    for (int i = 0; i < n; ++i) {
        if (!inliers_h_[i])
            continue;
        inlier_points.push_back(points[i]);
        inlier_ref_points.push_back(ref_points[i]);
    }

    // compute ratio of scores
    float r_score = h_score_/(h_score_ + f_score_);

    // try to reconstruct from homography or fundamental depending
    // on the ratio (0.40-0.45)
    if(r_score > 0.40) {
        // find R-t from homography matrix
        std::vector<cv::Mat> rot_mats;
        std::vector<cv::Mat> trans_mats;
        std::vector<cv::Mat> normals;
        decomposeHomographyMat(
            homography_mat_,
            camera_->intrinsicMatrix(),
            rot_mats,
            trans_mats,
            normals);

        // Remove wrong rotations and translations
        // R, t is wrong if a point ends up behind the camera
        std::vector<cv::Mat> res_Rs, res_ts, res_normals;
        cv::Mat sols;
        filterHomographyDecompByVisibleRefpoints(
            rot_mats,
            normals,
            inlier_points,
            inlier_ref_points,
            sols);

        if (!sols.empty()) {
            int idx = sols.at<int>(0, 0);
            best_rot_mat = rot_mats[idx];
            best_trans_mat = trans_mats[idx];
        }
    } else { //if(pF_HF>0.6)
        const auto& K = camera_->intrinsicMatrix();
        auto principal_point =
            cv::Point2f(camera_->centerX(), camera_->centerY());
        double focal_length =
            (camera_->focalX() + camera_->focalY()) / 2;

        cv::Mat essential_mat = K.t() * fundamental_mat_ * K;
        // Recover R,t from essential matrix
        recoverPose(
            essential_mat,
            inlier_points,
            inlier_ref_points,
            best_rot_mat,
            best_trans_mat,
            focal_length,
            principal_point);
    }
}

void Initializer::findFundamentalMat(
    const std::vector<cv::Point2f>& points,
    const std::vector<cv::Point2f>& ref_points,
    const cv::Mat& T,
    const cv::Mat& ref_T)
{
    const auto n = points.size();
    cv::Mat ref_T_t = ref_T.t();

    // best results variables
    f_score_ = 0.0;
    inliers_f_ = std::vector<bool>(n, false);

    // iteration variables
    std::vector<cv::Point2f> iter_points(8);
    std::vector<cv::Point2f> iter_ref_points(8);

    // perform all RANSAC iterations and save the solution with highest score
    cv::Mat f_mat;
    std::vector<bool> current_inliers(n, false);
    float current_score;
    for(int it = 0; it < iterations_; it++) {
        for(int j = 0; j < 8; j++) {
            const int idx = ransac_sets_[it][j];
            iter_points[j] = points[idx];
            iter_ref_points[j] = ref_points[idx];
        }

        cv::Mat f_normalized;
        geometry::computeFundamentalMat(
            f_normalized, iter_points, iter_ref_points);
        // unnormalize the fundamental matrix
        f_mat = ref_T_t * f_normalized * T;
        current_score =
            checkFundamentalScore(f_mat, points, ref_points, current_inliers);

        if (current_score > f_score_) {
            // set best f_mat to fundamental_mat
            fundamental_mat_ = f_mat.clone();
            inliers_f_ = current_inliers;
            f_score_ = current_score;
        }
    }
}

double Initializer::checkFundamentalScore(
    const cv::Mat& f_mat,
    const std::vector<cv::Point2f>& points,
    const std::vector<cv::Point2f>& ref_points,
    std::vector<bool>& inliers)
{
    const auto n = points.size();

    const double f11 = f_mat.at<double>(0, 0);
    const double f12 = f_mat.at<double>(0, 1);
    const double f13 = f_mat.at<double>(0, 2);
    const double f21 = f_mat.at<double>(1, 0);
    const double f22 = f_mat.at<double>(1, 1);
    const double f23 = f_mat.at<double>(1, 2);
    const double f31 = f_mat.at<double>(2, 0);
    const double f32 = f_mat.at<double>(2, 1);
    const double f33 = f_mat.at<double>(2, 2);

    double score = 0;

    const double th = 3.841;
    const double th_score = 5.991;

    const double inv_sigma_square = 1.0 / sigma_squared_;

    inliers.resize(n);
    for (int i = 0; i < n; i++)
    {
        bool good_point = true;

        const auto& p1 = points[i];
        const auto& p2 = ref_points[i];

        const auto& u1 = p1.x;
        const auto& v1 = p1.y;
        const auto& u2 = p2.x;
        const auto& v2 = p2.y;

        // Reprojection error in second image == Epipolar constraint error
        // l2=F21x1=(a2,b2,c2)

        const double a2 = f11 * u1 + f12 * v1 + f13;
        const double b2 = f21 * u1 + f22 * v1 + f23;
        const double c2 = f31 * u1 + f32 * v1 + f33;

        const double num2 = a2 * u2 + b2 * v2 + c2;
        const double square_dist_1 = num2 * num2 / (a2 * a2 + b2 * b2);

        const double chi_square_1 = square_dist_1 * inv_sigma_square;
        if (chi_square_1 > th)
        {
            score += 0;
            good_point = false;
        }
        else
            score += th_score - chi_square_1;

        // Reprojection error in second image
        // l1 =x2tF21=(a1,b1,c1)

        const double a1 = f11 * u2 + f21 * v2 + f31;
        const double b1 = f12 * u2 + f22 * v2 + f32;
        const double c1 = f13 * u2 + f23 * v2 + f33;

        const double num1 = a1 * u1 + b1 * v1 + c1;
        const double square_dist_2 = num1 * num1 / (a1 * a1 + b1 * b1);
        const double chi_square_2 = square_dist_2 * inv_sigma_square;

        if (chi_square_2 > th) {
            good_point = false;
        } else {
            score += th_score - chi_square_2;
        }

        inliers[i] = good_point;
    }
    return score;
}

void Initializer::findHomographyMat(
    const std::vector<cv::Point2f>& points,
    const std::vector<cv::Point2f>& ref_points,
    const cv::Mat& T,
    const cv::Mat& ref_T)
{
    const auto n = points.size();
    cv::Mat ref_T_inv = ref_T.inv();

    // best results variables
    h_score_ = 0.0;
    inliers_h_ = std::vector<bool>(n, false);

    // iteration variables
    std::vector<cv::Point2f> iter_points(8);
    std::vector<cv::Point2f> iter_ref_points(8);

    // perform all RANSAC iterations and save the solution with highest score
    cv::Mat h_mat;
    std::vector<bool> current_inliers(n, false);
    float current_score;
    for(int it = 0; it < iterations_; it++) {
        for(int j = 0; j < 8; j++) {
            const int idx = ransac_sets_[it][j];
            iter_points[j] = points[idx];
            iter_ref_points[j] = ref_points[idx];
        }

        cv::Mat h_normalized;
        geometry::computeHomographyMat(
            h_normalized, iter_points, iter_ref_points);
        // unnormalize the fundamental matrix
        h_mat = ref_T_inv * h_normalized * T;
        cv::Mat h_mat_inv = h_mat.inv();
        current_score =
            checkHomographyScore(
                points, ref_points, h_mat, h_mat_inv, current_inliers);

        if(current_score > f_score_) {
            // set best f_mat to fundamental_mat
            homography_mat_ = h_mat.clone();
            inliers_h_ = current_inliers;
            h_score_ = current_score;
        }
    }
}

double Initializer::checkHomographyScore(
    const std::vector<cv::Point2f>& points,
    const std::vector<cv::Point2f>& ref_points,
    const cv::Mat& h_mat,
    const cv::Mat& h_mat_inv,
    std::vector<bool>& inliers)
{
    const int n = frame_->nMatches();

    const float h11 = h_mat.at<float>(0,0);
    const float h12 = h_mat.at<float>(0,1);
    const float h13 = h_mat.at<float>(0,2);
    const float h21 = h_mat.at<float>(1,0);
    const float h22 = h_mat.at<float>(1,1);
    const float h23 = h_mat.at<float>(1,2);
    const float h31 = h_mat.at<float>(2,0);
    const float h32 = h_mat.at<float>(2,1);
    const float h33 = h_mat.at<float>(2,2);

    const float h11inv = h_mat_inv.at<float>(0,0);
    const float h12inv = h_mat_inv.at<float>(0,1);
    const float h13inv = h_mat_inv.at<float>(0,2);
    const float h21inv = h_mat_inv.at<float>(1,0);
    const float h22inv = h_mat_inv.at<float>(1,1);
    const float h23inv = h_mat_inv.at<float>(1,2);
    const float h31inv = h_mat_inv.at<float>(2,0);
    const float h32inv = h_mat_inv.at<float>(2,1);
    const float h33inv = h_mat_inv.at<float>(2,2);

    inliers.resize(n);

    float score = 0;

    const float th = 5.991;

    const float invSigmaSquare = 1.0 / sigma_squared_;

    for(int i = 0; i < n; i++) {
        bool good_point = true;

        const auto& p1 = points[i];
        const auto& p2 = ref_points[i];

        const auto& u1 = p1.x;
        const auto& v1 = p1.y;
        const auto& u2 = p2.x;
        const auto& v2 = p2.y;

        // Reprojection error in first image
        // x2in1 = H12*x2

        const float w2in1inv = 1.0/(h31inv*u2+h32inv*v2+h33inv);
        const float u2in1 = (h11inv*u2+h12inv*v2+h13inv)*w2in1inv;
        const float v2in1 = (h21inv*u2+h22inv*v2+h23inv)*w2in1inv;
        const float squareDist1 = (u1-u2in1)*(u1-u2in1)+(v1-v2in1)*(v1-v2in1);
        const float chiSquare1 = squareDist1*invSigmaSquare;

        if(chiSquare1 > th) {
            good_point = false;
        } else {
            score += th - chiSquare1;
        }

        // Reprojection error in second image
        // x1in2 = H21*x1

        const float w1in2inv = 1.0/(h31*u1+h32*v1+h33);
        const float u1in2 = (h11*u1+h12*v1+h13)*w1in2inv;
        const float v1in2 = (h21*u1+h22*v1+h23)*w1in2inv;

        const float squareDist2 = (u2-u1in2)*(u2-u1in2)+(v2-v1in2)*(v2-v1in2);

        const float chiSquare2 = squareDist2*invSigmaSquare;

        if(chiSquare2>th) {
            good_point = false;
        } else {
            score += th - chiSquare2;
        }

        inliers[i] = good_point;
    }

    return score;
}

} // namespace orb_slam