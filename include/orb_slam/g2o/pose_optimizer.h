/**
 * Declares the PoseOptimizer class.
 */

#include <eigen3/Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <orb_slam/g2o/g2o_types.h>
#include <orb_slam/frame.h>

namespace orb_slam
{

class PoseOptimizer {
public:
    PoseOptimizer() {
        solver = new g2o::OptimizationAlgorithmLevenberg(
            g2o::make_unique<BlockSolverType>(
                g2o::make_unique<LinearSolverType>()));
        optimizer = new g2o::SparseOptimizer();
        optimizer->setAlgorithm(solver);
    }

    ~PoseOptimizer() {
        delete solver;
        delete optimizer;
        solver = NULL;
        optimizer = NULL;
    }

    /**
     * Solves the optimization problem of finding the pose of a single frame in
     * 3d space (world coordinates) with 2d-3d point correspondences also in
     * world coordiantes.
     *
     * @param frame: Frame pose to optimize
     * @param opt_pose: Output optimized pose of the frame
     * @returns The number of features key points - the outliers found by
     *     optimization
     */
    int solve(const FramePtr& frame, cv::Mat& opt_pose) {
        optimizer->clear(); // reset optimizer

        // create the vertex for camera pose
        VertexPose *vertex_pose = new VertexPose();
        vertex_pose->setId(0); // first vertex
        vertex_pose->setFixed(false);

        Eigen::Matrix3d R;
        Eigen::Vector3d t;
        // get eigen pose from cv pose
        cv::cv2eigen<double>(frame->getCamInWorldR(), R);
        cv::cv2eigen<double>(frame->getCamInWorldt(), t);
        vertex_pose->setEstimate(Sophus::SE3d(R, t));

        // add camera pose as first vertex in graph
        optimizer->addVertex(vertex_pose);

        // get camera matrix in eigen
        Eigen::Matrix3d K;
        cv::cv2eigen<double>(frame->camera()->intrinsicMatrix(), K);

        // the more the scale factor -> the greater the down sampling ->
        // the greater variance. inv_scale_sigmas is inverse of variance
        const auto& inv_scale_sigmas = frame->orbExtractor()->invScaleSigmas();

        // create edges for all the observed 3d points from this frame
        const auto& key_points = frame->featuresUndist();
        const auto map_points = frame->obsMapPoints();
        const auto n = key_points.size();

        // create edges
        int index = 1;
        std::vector<EdgeProjectionPoseOnly*> edges;
        // Indices of edges corresponding to frame key points
        std::vector<size_t> edge_to_key_point;
        edges.reserve(n);
        edge_to_key_point.reserve(n);

        for (size_t i = 0; i < n; ++i) {
            const auto& mp = map_points[i];
            if (mp) { // if point exists
                //features.push_back(frame_->features_left_[i]);
                // get world position of point in eigen
                Eigen::Vector3d world_pos;
                cv::cv2eigen<double>(mp->worldPos(), world_pos);
                EdgeProjectionPoseOnly* edge = // create an edge for point pose
                    new EdgeProjectionPoseOnly(world_pos, K);
                edge->setId(index);
                // set vertex start point as camera pose vertex
                edge->setVertex(0, vertex_pose);
                // get point image position in eigen
                Eigen::Vector2d key_point_eigen(
                    key_points[i].pt.x, key_points[i].pt.y);
                // set point seen in camera as measured or observed position
                edge->setMeasurement(key_point_eigen);
                // use scale variance to determine information matrix
                // remember information matrix is the opposite of covariance
                // matrix
                Eigen::Matrix2d info =
                    Eigen::Matrix2d::Identity()*
                    inv_scale_sigmas[key_points[i].octave];
                edge->setInformation(info);
                // set loss function as huber loss
                edge->setRobustKernel(new g2o::RobustKernelHuber);
                edges.push_back(edge); // add edge to vector
                optimizer->addEdge(edge); // add edge to graph
                index++;
            }
        }

        // optimizer and determine the outliers for the given pose
        const double chi2_th = 5.991;
        int cnt_outlier = 0;
        const auto& outliers = frame->outliers();
        for (int iteration = 0; iteration < 4; ++iteration) {
            // get eigen pose from cv pose
            cv::cv2eigen<double>(frame->getCamInWorldR(), R);
            cv::cv2eigen<double>(frame->getCamInWorldt(), t);
            vertex_pose->setEstimate(Sophus::SE3d(R, t));
            optimizer->initializeOptimization();
            optimizer->optimize(10);
            cnt_outlier = 0;

            // count the outliers
            for (size_t i = 0; i < edges.size(); ++i) {
                auto e = edges[i];
                if (outliers[i]) {
                    e->computeError();
                }
                if (e->chi2() > chi2_th) {
                    frame->setOutlier(i, true);
                    e->setLevel(1);
                    cnt_outlier++;
                } else {
                    frame->setOutlier(i, false);
                    e->setLevel(0);
                };

                if (iteration == 2) {
                    e->setRobustKernel(nullptr);
                }
            }
        }

        ROS_DEBUG_STREAM(
            "Outlier/Inlier in pose estimating: "
            << cnt_outlier << "/" << key_points.size() - cnt_outlier);

        // set optimized pose and outliers to the frame
        cv::Mat opt_pose;
        cv::eigen2cv(vertex_pose->estimate().matrix(), opt_pose);
        frame->setPose(opt_pose);

        ROS_DEBUG_STREAM(
            "Current frame pose = \n" << frame->getCamInWorldT());

        return key_points.size() - cnt_outlier;
    }

private:
    // setup g2o
    typedef g2o::BlockSolver_6_3 BlockSolverType;
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType>
        LinearSolverType;
    g2o::OptimizationAlgorithmLevenberg* solver;
    g2o::SparseOptimizer* optimizer;
};

using PoseOptimizerPtr = std::shared_ptr<PoseOptimizer>;

} // namespace orb_slam
