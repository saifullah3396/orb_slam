/**
 * Declares the LocalBundleAdjuster class.
 */

#include <eigen3/Eigen/Dense>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <opencv2/core/eigen.hpp>
#include <orb_slam/g2o/g2o_types.h>
#include <orb_slam/frame.h>
#include <orb_slam/key_frame.h>
#include <orb_slam/map_point.h>
#include <orb_slam/map.h>

namespace orb_slam
{

class LocalBundleAdjuster {
public:
    LocalBundleAdjuster(const MapPtr& map) : map_(map) {
        solver = new g2o::OptimizationAlgorithmLevenberg(
            g2o::make_unique<g2o::BlockSolver_6_3>(
                g2o::make_unique<g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>>()));
        optimizer = new g2o::SparseOptimizer();
        optimizer->setAlgorithm(solver);
    }

    ~LocalBundleAdjuster() {
        delete solver;
        delete optimizer;
        solver = NULL;
        optimizer = NULL;
    }

    /**
     * Solves the optimization problem of adjusting the poses of all the key frames
     * in the local map based on 3d space (world coordinates) of the frames and with
     * 2d-3d point correspondences also in world coordiantes.
     *
     * @param frame: Frame pose to optimize
     * @param opt_pose: Output optimized pose of the frame
     * @returns The number of feature inliers found by
     *     optimization
     */
    void solve(const KeyFramePtr& key_frame, bool* abort_ba) {
        // add local key frames: first breath search from current key frame
        std::vector<KeyFramePtr> local_key_frames;
        local_key_frames.push_back(key_frame);
        key_frame->setInLocalAdjustmentOf(key_frame->id());

        const auto cov_key_frames = key_frame->getCovisibles();
        for (const auto& kf: cov_key_frames) {
            kf->setInLocalAdjustmentOf(key_frame->id());
            if (!kf->isBad()) {
                local_key_frames.push_back(kf);
            }
        }

        // add local map points seen in local key frames
        std::vector<MapPointPtr> local_map_points;
        for (const auto& kf: local_key_frames) {
            const auto& map_points = kf->obsMapPoints();
            for (const auto& mp: map_points) {
                if (mp && !mp->isBad() && !mp->inLocalAdjustmentOf(key_frame->id())) {
                    local_map_points.push_back(mp);
                    mp->setInLocalAdjustmentOf(key_frame->id());
                }
            }
        }

        // add fixed key frames. Key frames that see local map points but are
        // not local key frames themselves
        std::vector<KeyFramePtr> fixed_key_frames;
        for (const auto& mp: local_map_points)
        {
            const auto observations = mp->observations();
            for (const auto& obs: observations) {
                const auto kf = obs.first;
                // if it is not in adjustment and
                if (!kf->inLocalAdjustmentOf(key_frame->id()) && kf->inLocalFixedAdjustmentOf(key_frame->id())) {
                    kf->setInLocalFixedAdjustmentOf(key_frame->id());
                    if (!kf->isBad()) {
                        fixed_key_frames.push_back(kf);
                    }
                }
            }
        }

        optimizer->clear(); // reset optimizer
        optimizer->setForceStopFlag(abort_ba);

        unsigned long max_key_frame_id = 0;
        // set local key frame vertices
        for (const auto& kf: local_key_frames) {
            const auto& id = kf->id();
            auto vertex_pose = new VertexPose();

            // create se3 pose from R and t and set it as estimate
            Eigen::Matrix3d R;
            Eigen::Vector3d t;
            cv::cv2eigen<double>(key_frame->worldInCameraR(), R);
            cv::cv2eigen<double>(key_frame->worldInCamerat(), t);
            R = Eigen::AngleAxisd(R).toRotationMatrix();
            vertex_pose->setEstimate(Sophus::SE3d(R, t));
            vertex_pose->setId(id);
            vertex_pose->setFixed(id==0);
            optimizer->addVertex(vertex_pose);
            if(id > max_key_frame_id)
                max_key_frame_id = id;
        }

        // set fixed key frame vertices
        for (const auto& kf: fixed_key_frames) {
            const auto& id = kf->id();
            auto vertex_pose = new VertexPose();

            // create se3 pose from R and t and set it as estimate
            Eigen::Matrix3d R;
            Eigen::Vector3d t;
            cv::cv2eigen<double>(key_frame->worldInCameraR(), R);
            cv::cv2eigen<double>(key_frame->worldInCamerat(), t);
            R = Eigen::AngleAxisd(R).toRotationMatrix();
            vertex_pose->setEstimate(Sophus::SE3d(R, t));
            vertex_pose->setId(id);
            vertex_pose->setFixed(true);
            optimizer->addVertex(vertex_pose);
            if(id > max_key_frame_id)
                max_key_frame_id = id;
        }

        // set map point vertices
        const int expected_size =
            (local_key_frames.size() + fixed_key_frames.size()) *
            local_map_points.size();

        // get camera matrix in eigen
        Eigen::Matrix3d K;
        cv::cv2eigen<double>(key_frame->frame()->camera()->intrinsicMatrix(), K);

        // the more the scale factor -> the greater the down sampling ->
        // the greater variance. inv_scale_sigmas is inverse of variance
        const auto& inv_scale_sigmas =
            key_frame->frame()->orbExtractor()->invScaleSigmas();

        // create edges
        int node_idx = 1;
        std::vector<EdgeProjectionDepth*> edges;
        // Indices of edges corresponding to frame key points
        std::vector<MapPointPtr> edge_map_point;
        std::vector<KeyFramePtr> edge_key_frame;
        edges.reserve(expected_size);
        edge_map_point.reserve(expected_size);
        edge_key_frame.reserve(expected_size);

        for (auto it = local_map_points.begin(); it != local_map_points.end(); ++it) {
            const auto& mp = *it;
            auto vertex = new VertexXYZ();

            // add point as vertex
            Eigen::Vector3d world_pos;
            // get world position of point in eigen
            cv::cv2eigen<double>(mp->worldPos(), world_pos);
            int id = mp->id() + max_key_frame_id + 1;
            vertex->setId(id);
            vertex->setMarginalized(true);
            optimizer->addVertex(vertex);

            static const float th_huber = sqrt(7.815);

            // now add observations of map point as edges
            const auto observations = mp->observations();
            for (const auto& obs: observations) {
                const auto& kf = obs.first;
                const auto& key_points = kf->frame()->featuresUndist();
                const auto& depths = kf->frame()->featureDepthsUndist();
                if (!kf->isBad()) {
                    const auto& kp = key_points[obs.second];
                    const auto& depth = depths[obs.second];
                    // depth observation
                    Eigen::Vector3d measurement(kp.pt.x, kp.pt.y, depth);

                    // create edge for map point to depth camera projection
                    auto edge = new EdgeProjectionDepth(K);

                    // set first vertex to be map point id
                    edge->setVertex(
                        0,
                        dynamic_cast<g2o::OptimizableGraph::Vertex*>(
                            optimizer->vertex(id)));

                    // set second vertex to be key frame id
                    edge->setVertex(
                        1,
                        dynamic_cast<g2o::OptimizableGraph::Vertex*>(
                            optimizer->vertex(kf->id())));

                    // set measurement for map point position in 3d from
                    // key frame camera
                    edge->setMeasurement(measurement);

                    // set information matrix based on scale of key point
                    // associated with map point in this key frame
                    const auto& inv_sigma = inv_scale_sigmas[kp.octave];
                    Eigen::Matrix3d info =
                        Eigen::Matrix3d::Identity() * inv_sigma;
                    edge->setInformation(info);

                    // set robust loss function as huber loss
                    // this is done for outlier rejection
                    // See https://qcloud.coding.net/u/vincentqin/p/blogResource/
                    // git/raw/master/slam/g2o-details.pdf for details
                    g2o::RobustKernelHuber* kernel = new g2o::RobustKernelHuber;
                    edge->setRobustKernel(new g2o::RobustKernelHuber);
                    kernel->setDelta(th_huber);

                    optimizer->addEdge(edge);
                    edges.push_back(edge);
                    edge_key_frame.push_back(kf);
                    edge_map_point.push_back(mp);
                }
            }
        }

        if (abort_ba) // stop here if required
            return;

        optimizer->initializeOptimization();
        optimizer->optimize(5);

        bool continued = true;

        if (abort_ba) // stop here if required
            continued = false;

        const double chi2_th = 5.991;
        if (continued) {
            // count the outliers
            for (size_t i = 0; i < edges.size(); ++i) {
                auto& e = edges[i];
                const auto& mp = edge_map_point[i];
                if (mp->isBad()) continue;

                if (e->chi2() > chi2_th || !e->isDepthPositive()) {
                    // if the error is greater than threshold or depth is
                    // negative, remove the point
                    // we remove this point from optimization level 0, so that
                    // it is not included in optimization in the next run
                    e->setLevel(1);
                }

                e->setRobustKernel(0);
            }

            // optimize again without the outliers
            optimizer->initializeOptimization(0);
            optimizer->optimize(5);
        }

        std::map<KeyFramePtr, MapPointPtr> bad_vertices;

        // check inlier observations
        for (size_t i = 0; i < edges.size(); ++i) {
            auto& e = edges[i];
            const auto& mp = edge_map_point[i];
            if (mp->isBad()) continue;

            if (e->chi2() > chi2_th || !e->isDepthPositive()) {
                bad_vertices[edge_key_frame[i]] = mp;
            }
        }

        // lock the map for update
        map_->lock();
        if (!bad_vertices.empty()) {
            for (const auto& v : bad_vertices) {
                auto kf = v.first;
                auto mp = v.second;
                kf->removeMapPoint(mp);
                mp->removeObservation(kf);
            }
        }

        // reset new estimates of local key frames
        for (const auto& kf : local_key_frames) {
            const auto pose_se3 =
                static_cast<VertexPose*>(optimizer->vertex(kf->id()))->estimate();
            cv::Mat opt_pose;
            // set optimized pose to the output
            cv::eigen2cv(pose_se3.matrix(), opt_pose);
            opt_pose.convertTo(opt_pose, CV_32F); // convert to float
            kf->setWorldInCam(opt_pose);
        }

        // reset new estimates of local map points
        for (const auto& mp : local_map_points) {
            int id = mp->id() + max_key_frame_id + 1;
            const auto point_xyz =
                static_cast<VertexXYZ*>(optimizer->vertex(id))->estimate();
            cv::Mat opt_xyz;
            // set optimized pose to the output
            cv::eigen2cv(point_xyz, opt_xyz);
            opt_xyz.convertTo(opt_xyz, CV_32F); // convert to float
            mp->setWorldPos(opt_xyz);
            mp->updateNormalAndScale();
        }
    }

private:
    // setup g2o
    typedef g2o::BlockSolver_6_3 BlockSolverType;
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType>
        LinearSolverType;
    g2o::OptimizationAlgorithmLevenberg* solver;
    g2o::SparseOptimizer* optimizer;

    MapPtr map_;

    std::string name_tag_ = {"LocalBundleAdjuster"};
};

using LocalBundleAdjusterPtr = std::shared_ptr<LocalBundleAdjuster>;

} // namespace orb_slam
