/**
 * Declares the PoseOptimizer class.
 */

#include <eigen3/Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <orb_slam/g2o/g2o_types.h>
#include <orb_slam/frame.h>

namespace orb_slam
{

class PoseOptimizer_ {
public:
    PoseOptimizer_() {
        solver = new g2o::OptimizationAlgorithmLevenberg(
            g2o::make_unique<BlockSolverType>(
                g2o::make_unique<LinearSolverType>()));
        optimizer = new g2o::SparseOptimizer();
        optimizer->setAlgorithm(solver);
    }

    ~PoseOptimizer_() {
        delete solver;
        delete optimizer;
        solver = NULL;
        optimizer = NULL;
    }

    virtual int solve(const FramePtr& frame, cv::Mat& opt_pose) = 0;

protected:
    // setup g2o
    typedef g2o::BlockSolver_6_3 BlockSolverType;
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType>
        LinearSolverType;
    g2o::OptimizationAlgorithmLevenberg* solver;
    g2o::SparseOptimizer* optimizer;
};

template <class Edge>
class PoseOptimizer : public PoseOptimizer_ {
public:
    PoseOptimizer() {}
    ~PoseOptimizer() {}

    void createEdges(
        VertexPose* camera_pose,
        const FramePtr& frame,
        std::vector<EdgeProjectionPoseOnlyMono*>& edges,
        std::vector<size_t>& edge_to_key_point,
        int& start_node_idx)
    {
        // get camera matrix in eigen
        Eigen::Matrix3d K;
        cv::cv2eigen<double>(frame->camera()->intrinsicMatrix(), K);

        // the more the scale factor -> the greater the down sampling ->
        // the greater variance. inv_scale_sigmas is inverse of variance
        const auto& inv_scale_sigmas = frame->orbExtractor()->invScaleSigmas();

        // create edges for all the observed 3d points from this frame
        const auto& key_points = frame->featuresUndist();
        const auto& map_points = frame->obsMapPoints();
        const auto n = key_points.size();

        edges.reserve(n);
        edge_to_key_point.reserve(n);

        for (size_t idx = 0; idx < n; ++idx) {
            const auto& mp = map_points[idx];
            const auto& kp = key_points[idx];
            if (mp) { // if point exists
                //features.push_back(frame_->features_left_[i]);
                // get world position of point in eigen
                Eigen::Vector3d world_pos;
                cv::cv2eigen<double>(mp->worldPos(), world_pos);
                auto edge = // create an edge for point pose
                    new EdgeProjectionPoseOnlyMono(world_pos, K);
                edge->setId(start_node_idx);
                // set vertex start point as camera pose vertex
                edge->setVertex(0, camera_pose);
                // get point image position in eigen
                Eigen::Vector2d key_point_eigen(kp.pt.x, kp.pt.y);
                // set point seen in camera as measured or observed position
                edge->setMeasurement(key_point_eigen);
                // use scale variance to determine information matrix
                // remember information matrix is the opposite of covariance
                // matrix
                Eigen::Matrix2d info =
                    Eigen::Matrix2d::Identity() * inv_scale_sigmas[kp.octave];
                edge->setInformation(info);
                // set robust loss function as huber loss
                // this is done for outlier rejection
                // See https://qcloud.coding.net/u/vincentqin/p/blogResource/
                // git/raw/master/slam/g2o-details.pdf for details
                edge->setRobustKernel(new g2o::RobustKernelHuber);
                edges.push_back(edge); // add edge to vector
                edge_to_key_point.push_back(idx);
                optimizer->addEdge(edge); // add edge to graph
                start_node_idx++;
            }
        }
    }

    void createEdges(
        VertexPose* camera_pose,
        const FramePtr& frame,
        std::vector<EdgeProjectionPoseOnlyDepth*>& edges,
        std::vector<size_t>& edge_to_key_point,
        int& start_node_idx)
    {
        // get camera matrix in eigen
        Eigen::Matrix3d K;
        cv::cv2eigen<double>(frame->camera()->intrinsicMatrix(), K);

        // the more the scale factor -> the greater the down sampling ->
        // the greater variance. inv_scale_sigmas is inverse of variance
        const auto& inv_scale_sigmas = frame->orbExtractor()->invScaleSigmas();

        // create edges for all the observed 3d points from this frame
        const auto& key_points = frame->featuresUndist();
        const auto& depths = frame->featureDepthsUndist();
        const auto& map_points = frame->obsMapPoints();
        const auto n = key_points.size();

        edges.reserve(n);
        edge_to_key_point.reserve(n);

        for (size_t idx = 0; idx < n; ++idx) {
            const auto& mp = map_points[idx];
            const auto& kp = key_points[idx];
            const auto& depth = depths[idx];
            if (mp) { // if point exists
                //features.push_back(frame_->features_left_[i]);
                // get world position of point in eigen
                Eigen::Vector3d world_pos;
                cv::cv2eigen<double>(mp->worldPos(), world_pos);
                auto edge = // create an edge for point pose
                    new EdgeProjectionPoseOnlyDepth(world_pos, K);
                edge->setId(start_node_idx);
                // set vertex start point as camera pose vertex
                edge->setVertex(0, camera_pose);
                // get point image position in eigen
                Eigen::Vector3d key_point_eigen(kp.pt.x, kp.pt.y, depth);
                // set point seen in camera as measured or observed position
                edge->setMeasurement(key_point_eigen);
                // use scale variance to determine information matrix
                // remember information matrix is the opposite of covariance
                // matrix
                Eigen::Matrix3d info =
                    Eigen::Matrix3d::Identity() * inv_scale_sigmas[kp.octave];
                edge->setInformation(info);
                // set robust loss function as huber loss
                // this is done for outlier rejection
                // See https://qcloud.coding.net/u/vincentqin/p/blogResource/
                // git/raw/master/slam/g2o-details.pdf for details
                edge->setRobustKernel(new g2o::RobustKernelHuber);
                edges.push_back(edge); // add edge to vector
                edge_to_key_point.push_back(idx);
                optimizer->addEdge(edge); // add edge to graph
                start_node_idx++;
            }
        }
    }

    /**
     * Solves the optimization problem of finding the pose of a single frame in
     * 3d space (world coordinates) with 2d-3d point correspondences also in
     * world coordiantes.
     *
     * @param frame: Frame pose to optimize
     * @param opt_pose: Output optimized pose of the frame
     * @returns The number of feature inliers found by
     *     optimization
     */
    int solve(const FramePtr& frame, cv::Mat& opt_pose) {
        optimizer->clear(); // reset optimizer

        // create the vertex for camera pose
        auto camera_pose = new VertexPose();
        camera_pose->setId(0); // first vertex
        camera_pose->setFixed(false);

        Eigen::Matrix3d R;
        Eigen::Vector3d t;
        // get eigen pose from cv pose
        cv::cv2eigen<double>(frame->worldInCameraR(), R);
        cv::cv2eigen<double>(frame->worldInCamerat(), t);

        // reinforce orthogonality
        R = Eigen::AngleAxisd(R).toRotationMatrix();

        auto init_pose_se3 = Sophus::SE3d(R, t);
        camera_pose->setEstimate(init_pose_se3);

        // add camera pose as first vertex in graph
        optimizer->addVertex(camera_pose);

        std::vector<Edge*> edges;
        std::vector<size_t> edge_to_key_point;
        int node_idx = 1;
        createEdges(
            camera_pose, frame, edges, edge_to_key_point, node_idx);

        // optimizer and determine the outliers for the given pose
        const double chi2_th = 5.991;
        int cnt_outlier = 0;
        const auto& outliers = frame->outliers();
        //optimizer->setVerbose(true);
        for (int iteration = 0; iteration < 4; ++iteration) {
            // first three iterations, the robust kernel is used which is used
            // to give more weight to inliers as explained in g2o docs.
            // the last iteration the kernel is removed to find the real error.

            // reset the initial pose and optimize each iteration from start
            camera_pose->setEstimate(init_pose_se3);
            optimizer->initializeOptimization();
            optimizer->optimize(10);
            cnt_outlier = 0;

            // count the outliers
            for (size_t i = 0; i < edges.size(); ++i) {
                auto& e = edges[i];
                const auto& idx = edge_to_key_point[i];
                if (outliers[idx]) {
                    e->computeError();
                }
                if (e->chi2() > chi2_th) {
                    // if the error is greater than threshold, remove the point
                    frame->setOutlier(idx, true);
                    // we remove this point from optimization level 0, so that
                    // it is not included in optimization in the next run
                    e->setLevel(1);
                    cnt_outlier++;
                } else { // else keep it
                    frame->setOutlier(idx, false);
                    // we add this point to optimization level 0, so that
                    // it is included in optimization in the next run
                    e->setLevel(0);
                };

                // remove robust kernel in the last iteration to find the
                // real quadratic error
                if (iteration == 2) {
                    e->setRobustKernel(nullptr);
                }
            }
        }

        ROS_DEBUG_STREAM("Mean error over number of points:" <<
            (float) optimizer->activeChi2() / (float) optimizer->activeEdges().size());

        const auto& n = frame->nFeaturesUndist();
        ROS_DEBUG_STREAM(
            "Outlier/Inlier in pose estimating: "
            << cnt_outlier << "/" << n - cnt_outlier);

        // set optimized pose to the output
        cv::eigen2cv(camera_pose->estimate().matrix(), opt_pose);
        opt_pose.convertTo(opt_pose, CV_32F); // convert to float

        return n - cnt_outlier;
    }
};

template class PoseOptimizer<EdgeProjectionPoseOnlyMono>;
template class PoseOptimizer<EdgeProjectionPoseOnlyDepth>;
using PoseOptimizerMono = PoseOptimizer<EdgeProjectionPoseOnlyMono>;
using PoseOptimizerRGBD = PoseOptimizer<EdgeProjectionPoseOnlyDepth>;
using PoseOptimizerMonoPtr = std::shared_ptr<PoseOptimizerMono>;
using PoseOptimizerRGBDPtr = std::shared_ptr<PoseOptimizerRGBD>;
using PoseOptimizerPtr = std::shared_ptr<PoseOptimizer_>;

} // namespace orb_slam
