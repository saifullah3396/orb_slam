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
    auto mean_x = sum.x / n;
    auto mean_y = sum.y / n;

    // use same normalization done in orb-slam repository
    // this looks similar to hartley's normalization but is a bit different
    // in place of 's' parameter, there are 'sx', and 'sy' parameters.
    // see https://cs.adelaide.edu.au/~wojtek/papers/pami-nals2.pdf
    float mean_dev_x = 0;
    float mean_dev_y = 0;
    for (int i = 0; i < n; i++) {
        normalized[i].x = points.x - mean_x;
        normalized[i].y = points.y - mean_y;
        mean_dev_x += fabs(normalized[i].x);
        mean_dev_y += fabs(normalized[i].y);
    }
    mean_dev_x = mean_dev_x / n;
    mean_dev_y = mean_dev_y / n;
    float s_x = 1.0 / mean_dev_x;
    float s_y = 1.0 / mean_dev_y;
    for (int i = 0; i < n; i++) {
        normalized[i].x = normalized[i].x * s_x;
        normalized[i].y = normalized[i].y * s_y;
    }
    T = cv::Mat::eye(3, 3, CV_32F);
    T.at<float>(0,0) = s_x;
    T.at<float>(1,1) = s_y;
    T.at<float>(0,2) = -mean_x * s_x;
    T.at<float>(1,2) = -mean_y * s_y;
}

} // namespace geometry

} // namespace orb_slam