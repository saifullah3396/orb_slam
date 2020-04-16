/**
 * Defines the Tracker class.
 */

#include <ros/ros.h>
#include <ros/package.h>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "orb_slam/frame.h"
#include "orb_slam/key_frame.h"
#include "orb_slam/map.h"
#include "orb_slam/map_point.h"
#include "orb_slam/tracker.h"
#include "orb_slam/mono_tracker.h"
#include "orb_slam/motion_model.h"
#include "orb_slam/rgbd_tracker.h"
#include "orb_slam/initializer.h"
#include "orb_slam/geometry/camera.h"
#include "orb_slam/geometry/orb_extractor.h"
#include "orb_slam/geometry/orb_matcher.h"
#include "orb_slam/g2o/pose_optimizer.h"
#include "orb_slam/local_mapping/local_mapper.h"
#include "orb_slam/viewer/viewer.h"

namespace orb_slam
{

Tracker::Tracker(const ros::NodeHandle& nh, const int& camera_type) : nh_(nh)
{
    // initialize the camera
    ROS_DEBUG("Initializing camera...");
    camera_ =
        geometry::Camera<float>::makeCamera(
            nh_,
            static_cast<geometry::CameraType>(camera_type));
    camera_->readParams();
    camera_->setup();
    camera_->setupCameraStream();

    n_min_frames_ = 0;
    n_max_frames_ = camera_->fps();

    ROS_DEBUG("Initializing orb features vocabulary...");
    auto pkg_path = ros::package::getPath("orb_slam");
    orb_vocabulary_ = ORBVocabularyPtr(new ORBVocabulary());
    std::string vocabulary_path;
    ROS_DEBUG_STREAM("pkg_path:" << pkg_path);
    nh_.getParam("/orb_slam/tracker/vocabulary_path", vocabulary_path);
    vocabulary_path = pkg_path + "/" + vocabulary_path;
    ROS_DEBUG_STREAM("vocabulary_path:" << vocabulary_path);
    try {
        orb_vocabulary_->loadFromTextFile(vocabulary_path);
    } catch (std::exception& e) {
        ROS_FATAL_STREAM(e.what());
        ROS_FATAL_STREAM(
            "Failed to load ORB vocabulary from path:" <<
                pkg_path + "/" + vocabulary_path);
        exit(-1);
    }
    ROS_DEBUG("ORB vocabulary successfully loaded...");

    ROS_DEBUG("Initializing orb features extractor...");
    orb_extractor_ =
        geometry::ORBExtractorPtr(new geometry::ORBExtractor(nh_));

    ROS_DEBUG("Initializing orb features matcher...");
    orb_matcher_ =
        geometry::ORBMatcherPtr(new geometry::ORBMatcher(nh_));

    ROS_DEBUG("Initializing frame base variables...");
    Frame::setCamera(
        std::const_pointer_cast<const geometry::Camera<float>>(camera_));
    Frame::setORBExtractor(
        std::const_pointer_cast<const geometry::ORBExtractor>(orb_extractor_));
    Frame::setORBMatcher(
        std::const_pointer_cast<const geometry::ORBMatcher>(orb_matcher_));
    Frame::setORBVocabulary(
        std::const_pointer_cast<const ORBVocabulary>(orb_vocabulary_));
    Frame::setupGrid(nh_);

    ROS_DEBUG("Initializing the global map...");
    map_ = MapPtr(new Map());

    ROS_DEBUG("Setting orb extractors and matchers...");
    MapPoint::setORBExtractor(
        std::const_pointer_cast<const geometry::ORBExtractor>(orb_extractor_));
    MapPoint::setORBMatcher(
        std::const_pointer_cast<const geometry::ORBMatcher>(orb_matcher_));

    ROS_DEBUG("Initializing motion model...");
    motion_model_ = MotionModelPtr<float>(new MotionModel<float>());

    ROS_DEBUG("Initializing viewer...");
    viewer_ = ViewerPtr(new Viewer(map_));
    viewer_->startThread();

    ROS_DEBUG("Initializing local mapper...");
    local_mapper_ = LocalMapperPtr(new LocalMapper(map_));

    state_ = NO_IMAGES_YET;
    ROS_DEBUG("Tracker node successfully initialized...");
}

Tracker::~Tracker()
{
}

void Tracker::reset()
{

}

std::unique_ptr<Tracker> Tracker::createTracker(
    const ros::NodeHandle& nh, const int& camera_type) {
    if (camera_type == static_cast<int>(geometry::CameraType::MONO)) {
        return std::unique_ptr<Tracker>(new MonoTracker(nh, camera_type));
    } else if (camera_type == static_cast<int>(geometry::CameraType::RGBD)) {
        return std::unique_ptr<Tracker>(new RGBDTracker(nh, camera_type));
    }
}

void Tracker::trackFrame()
{
    if (state_ == TrackingState::LOST) {
        ROS_DEBUG_STREAM("Tracking lost...");
        exit(1);
    }

    // means first image is yet to be processed
    if (state_ == NO_IMAGES_YET) {
        state_ = NOT_INITIALIZED;
    }
    last_proc_state_ = state_;

    // map is frozen and cannot be accessed by other threads
    std::unique_lock<std::mutex> lock(map_->mapUpdateMutex());

    // reset local map matches
    local_map_matches_ = 0;

    if(state_ == NOT_INITIALIZED) {
        initializeTracking();
    } else {
        bool tracking_good = false;
        // initialization done, starting tracking...
        if(state_ == OK) {
            if (motion_model_ && motion_model_->initialized()) {
                ROS_DEBUG_STREAM("Motion model initialized. Tracking with motion model.");
                tracking_good = trackWithMotionModel();
            } else {
                ROS_DEBUG_STREAM("Tracking the reference key frame");
                tracking_good = trackReferenceKeyFrame();
            }
        } else {
            ROS_DEBUG_STREAM("Tracking bad. Relocalizing...");
            tracking_good = relocalize();
            tracking_good = false;
        }

        if (tracking_good) {
            ROS_DEBUG_STREAM("Tracking good. updating local map...");
            // update the local map of this frame...
            tracking_good = updateLocalMap();
        }

        if (tracking_good) {
            state_ = TrackingState::OK;

            ROS_DEBUG_STREAM("Tracking good. updating motion model...");
            auto current_pose = current_frame_->worldInCameraT();
            motion_model_->updateModel(current_pose, current_frame_->timeStamp());

            viewer_->addFrame(current_frame_);
            viewer_->updateMap();

            // checking if we need to insert a new keyframe
            ROS_DEBUG_STREAM("Checking if we need to insert a new keyframe...");
            if (needNewKeyFrame()) {
                ROS_DEBUG_STREAM("Creating new keyframe...");
                createNewKeyFrame();
            }
        } else {
            ROS_DEBUG_STREAM("Tracking bad...");
            state_ = TrackingState::LOST;

            if(map_->nKeyFrames() <= MIN_REQ_KEY_FRAMES_RELOC) {
                ROS_WARN_STREAM(
                    "Tracking lost with very few key frames. \
                    Cannot relocalize...");
                reset();
                return;
            }
        }

        if (current_frame_->refKeyFrame())
            ROS_DEBUG_STREAM(
                "current_frame_->refKeyFrame():" <<
                current_frame_->refKeyFrame()->id());

        // if no reference frame exists for this frame
        if(!current_frame_->refKeyFrame()) // why do this?
            current_frame_->setRefKeyFrame(ref_key_frame_);

        last_frame_ = current_frame_;
    }
    //ROS_DEBUG("HERE");
    //current_frame_->showImageWithFeatures("Current Frame");
    //cv::waitKey(0);
}

bool Tracker::trackReferenceKeyFrame()
{
    ROS_DEBUG_STREAM("Reference key frame: " << ref_key_frame_->id());
    ROS_DEBUG_STREAM("Current frame: " << current_frame_->id());

    ROS_DEBUG_STREAM("Computing orb bow features...");
    // compute bag of words vector for current frame
    current_frame_->computeBow();
    ROS_DEBUG_STREAM("Bow: " << current_frame_->bow().size());
    ROS_DEBUG_STREAM("Bow Features: " << current_frame_->bowFeatures().size());

    // find matches between current and reference frame.
    ROS_DEBUG_STREAM("Matching bow features between frames...");

    // 0.7 taken from original orb slam code
    current_frame_->matchByBowFeatures(ref_key_frame_, true, 0.7);
    const auto& matches = current_frame_->matches();

    ROS_DEBUG_STREAM("Matches: " << matches.size());

    if (matches.size() < MIN_REQ_MATCHES)
        return false;

    //current_frame_->showMatchesWithRef("Matched points.");
    //cv::waitKey(0);
    //ROS_DEBUG_STREAM("Adding resultant map points to map.");

    const auto ref_map_points = ref_key_frame_->frame()->obsMapPoints();
    for (const auto& m: matches) {
        // assign matched map points from reference to current frame
        current_frame_->setMapPointAt(ref_map_points[m.trainIdx], m.queryIdx);
    }

    ROS_DEBUG_STREAM("Optimizing current frame pose...");
    // set initial pose of this frame to last frame. This acts as starting point
    // for pose optimization using graph
    current_frame_->setWorldInCam(ref_key_frame_->frame()->worldInCameraT());
    cv::Mat opt_pose;
    pose_optimizer_->solve(current_frame_, opt_pose);
    current_frame_->setWorldInCam(opt_pose);

    // discard outliers
    ROS_DEBUG_STREAM("Discarding outliers points...");
    int map_matches = 0;
    const auto map_points = current_frame_->obsMapPoints();
    const auto& outliers = current_frame_->outliers();
    for (int i = 0; i < map_points.size(); i++) {
        // copy since it has to be updated after removal from current frame
        const auto mp = map_points[i];
        if(!mp) continue;
        if(outliers[i]) {
            // remove the matched map point from the frame since it is an
            // outlier
            current_frame_->removeMapPointAt(i);
            // reset the feature as inlier for usage next time with maybe
            // another reference matching
            current_frame_->setOutlier(i, false);

            // Note that even though the point is removed from current frame
            // it still exists in map. We reset its status as untracked and
            // last seen in current frame. This is used by local mapping
            mp->setTrackedInFrame(current_frame_->id());
            mp->resetTrackProperties();
        } else if (mp->nObservations() > 0) {
            // only consider matched if there is at least one key frame
            // observation for this point
            map_matches++;
        }
    }
    ROS_DEBUG_STREAM("map_matches:" << map_matches);
    return map_matches >= 10;
}

bool Tracker::trackWithMotionModel()
{
    ROS_DEBUG_STREAM("Tracking with motion model...");
    ROS_DEBUG_STREAM("Last frame:" << last_frame_->id());
    ROS_DEBUG_STREAM("Current frame:" << current_frame_->id());
    // compute predicted camera pose...
    cv::Mat predicted_pose;
    // predict the pose at this time stamp
    if (!motion_model_->predict(predicted_pose, current_frame_->timeStamp())) {
        ROS_WARN_STREAM("Failed to predict next pose.");
        return false;
    }
    current_frame_->setWorldInCam(predicted_pose);
    current_frame_->resetMap();

    //ROS_DEBUG_STREAM("Camera in world:" << current_frame_->cameraInWorldT());

    // project points seen in previous frame to current frame//
    int radius; // match search radius
    if(camera_->type() == geometry::CameraType::MONO)
        radius = 15;
    else
        radius = 7;

    current_frame_->matchByProjection(last_frame_, true, radius);
    const auto& matches = current_frame_->matches();
    ROS_DEBUG_STREAM("Matches: " << matches.size());
    if (matches.size() < MIN_REQ_MATCHES_PROJ) {
        radius *= 2.0;
        current_frame_->resetMap();
        current_frame_->matchByProjection(last_frame_, true, radius);
    }

    //current_frame_->showMatchesWithRef("Matched points.");
    //cv::waitKey(0);

    if (matches.size() < MIN_REQ_MATCHES_PROJ) {
        return false;
    }

    const auto last_map_points = last_frame_->obsMapPoints();
    for (const auto& m: matches) {
        // add matched map points from reference to current frame
        current_frame_->setMapPointAt(
            last_map_points[m.trainIdx], m.queryIdx);
    }

    ROS_DEBUG_STREAM("Optimizing current frame pose...");
    cv::Mat opt_pose;
    pose_optimizer_->solve(current_frame_, opt_pose);
    current_frame_->setWorldInCam(opt_pose);

    // discard outliers
    ROS_DEBUG_STREAM("Discarding outliers points...");
    int inlier = 0;
    const auto map_points = current_frame_->obsMapPoints();
    const auto& outliers = current_frame_->outliers();
    for (int i = 0; i < map_points.size(); i++) {
        const auto& mp = map_points[i];
        if(!mp) continue;
        if(outliers[i]) {
            // remove the matched map point since it is an outlier
            current_frame_->removeMapPointAt(i);
            // reset the feature is inlier for usage next time with maybe
            // another reference matching
            current_frame_->setOutlier(i, false);
            //mp->setTrackInView(false); used in orb_slam
            //mp->setLastSeenFrame(current_frame_->id()); used in orb_slam
        } else if (mp->nObservations() > 0) {
            inlier++;
        }
    }
    ROS_DEBUG_STREAM("inliers:" << inlier);
    return inlier >= 10;
}

bool Tracker::updateLocalMap()
{
    // add all key frames that observe points in current frame to local map
    ROS_DEBUG("Updating local map key frames...");
    updateLocalMapKeyFrames();

    // add all points that are observed in newly added local map key frames
    // to local map
    ROS_DEBUG("Updating local map points...");
    updateLocalMapPoints();

    // project local points to current frame, those other than already found in
    // the current frame
    ROS_DEBUG("Projecting local map points...");
    if (!projectLocalPoints())
        return false;

    ROS_DEBUG_STREAM("Optimizing current frame pose with local map...");

    // optimize the frame pose with newly added map points
    cv::Mat opt_pose; // initial pose for optimization is already set
    pose_optimizer_->solve(current_frame_, opt_pose);
    current_frame_->setWorldInCam(opt_pose);

    // update map point stats, we don't discard them here to keep them for
    // bundle adjustment
    const auto& map_points = current_frame_->obsMapPoints();
    const auto& outliers = current_frame_->outliers();

    // We copy the map points currently found to store them for usage in bundle
    // adjusment
    ROS_DEBUG_STREAM("Storing map points for bundle adjustment...");
    current_frame_->copyMapPointsForBA();

    ROS_DEBUG_STREAM("Discarding outliers for current frame...");
    for (int i = 0; i < current_frame_->nFeaturesUndist(); i++) {
        const auto& mp = map_points[i];
        if(!mp) continue;
        if(outliers[i]) {
            // remove the matched map point since it is an outlier
            current_frame_->removeMapPointAt(i);

            // reset the feature is inlier for usage next time with maybe
            // another reference matching
            current_frame_->setOutlier(i, false);
        } else if (mp->nObservations() > 0) {
            mp->increaseFound();
            local_map_matches_++;
        }
    }

    // Decide if the tracking was succesful /////// from orb_slam 2
    // More restrictive if there was a relocalization recently /////// from orb_slam 2
    //if (current_frame_->id() < last_reloc_frame_id_ + max_frames_ && map_matches < 50) /////// from orb_slam 2
    //    return false;

    ROS_DEBUG_STREAM("map_matches: " << local_map_matches_);
    if (local_map_matches_ < 30)
        return false;
    else
        return true;
}

void Tracker::findObservingKeyFrames(
    const FramePtr& frame, map<KeyFramePtr, int>& obs_key_frame_map) const
{
    const auto map_points = frame->obsMapPoints();
    // find all map points that are matched in frame and see if they are
    // observed in any key frame. Based on this make a
    // key_frame -> number of observations map
    for (int i = 0; i < frame->nFeaturesUndist(); i++) {
        const auto& mp = map_points[i];
        if(mp) {
            if(!mp->isBad()) { // if it is a good map point
                // get all key frames this point is observed in
                auto observations = mp->observations();
                for(auto it = observations.begin();
                    it != observations.end(); it++)
                {
                    obs_key_frame_map[it->first]++;
                }
            } else { // remove if a bad point
                frame->removeMapPointAt(i);
            }
        }
    }
}

#define MAX_KEY_FRAMES_LOCAL_MAP 80
void Tracker::updateLocalMapKeyFrames()
{
    // find the key frames that observe points and the number of points each
    // key frame observes...
    map<KeyFramePtr, int> key_frame_counter;
    ROS_DEBUG("Finding key frames that observe current_frame_ points...");
    findObservingKeyFrames(current_frame_, key_frame_counter);
    if(key_frame_counter.empty()) {
        // no map point of current_frame_ is observed in any frame...
        return;
    }

    int max = 0;
    KeyFramePtr best_key_frame;

    // reset the local key frames map
    key_frames_local_map_.clear();
    key_frames_local_map_.reserve(3 * key_frame_counter.size());

    // all keyframes that observe a map point are included in the local map.
    // also check which keyframe shares most points
    for (const auto& kf_counter: key_frame_counter)
    {
        auto& key_frame = kf_counter.first;
        if(key_frame->isBad()) // ignore if bad
            continue;
        if(kf_counter.second > max) { // find max observations
            max = kf_counter.second;
            best_key_frame = key_frame;
        }

        // add key frame to local map
        key_frames_local_map_.push_back(kf_counter.first);
        key_frame->setIsInLocalMapOf(current_frame_->id());
    }

    if(best_key_frame)
    {
        ref_key_frame_ = best_key_frame;
        current_frame_->setRefKeyFrame(best_key_frame);
    }

    // include some not-already-included keyframes that are neighbors to
    // already-included keyframes
    const auto& current_frame_id = current_frame_->id();
    for(const auto& kf: key_frames_local_map_)
    {
        // Limit the number of keyframes
        if(key_frames_local_map_.size() > MAX_KEY_FRAMES_LOCAL_MAP)
            break;
        const auto covisible_kfs = kf->getBestCovisibles(10);
        for (const auto& c_kf: covisible_kfs) {
            // if it is not bad frame and is not already in local map
            if (!c_kf->isBad() && !c_kf->isInLocalMapOf(current_frame_id)) {
                key_frames_local_map_.push_back(c_kf); // add to map
                c_kf->setIsInLocalMapOf(current_frame_id);
                break; // why break? dunno yet. maybe to limit the number of neighbours to 1?
            }
        }

        // get chlids of this frame
        for(const auto& child: kf->children()) {
            // if it is not bad frame and is not already in local map
            if (!child->isBad() && !child->isInLocalMapOf(current_frame_id)) {
                key_frames_local_map_.push_back(child); // add to map
                child->setIsInLocalMapOf(current_frame_id);
                break; // why break? dunno yet. maybe to limit the number of children to 1?
            }
        }

        // if parent exists, add it as well
        const auto& parent = kf->parent();
        if(parent) {
            if (!parent->isBad() && !parent->isInLocalMapOf(current_frame_id)) {
                key_frames_local_map_.push_back(parent); // add to map
                parent->setIsInLocalMapOf(current_frame_id);
                break; // why break? dunno yet
            }
        }
    }

    ROS_DEBUG_STREAM("Key frames in local map:" << key_frames_local_map_.size());
}

void Tracker::updateLocalMapPoints()
{
    map_points_local_map_.clear();
    // get all map points observed by all key frames in local map
    for (const auto& kf: key_frames_local_map_) {
        const auto kf_map_points = kf->obsMapPoints();
        for (const auto& mp: kf_map_points) {
            // if it exists and does not lie already in local map and is not bad
            // add the point to local map
            if (mp && !mp->isInLocalMapOf(current_frame_->id()) && !mp->isBad()) {
                map_points_local_map_.push_back(mp);
                mp->setIsInLocalMapOf(current_frame_->id());
            }
        }
    }
}

bool Tracker::projectLocalPoints()
{
    // set flag for already seen points in this frame
    const auto& map_points = current_frame_->obsMapPoints();
    for (int idx = 0; idx < map_points.size(); ++idx) {
        const auto& mp = map_points[idx];
        if (mp) {
            if (!mp->isBad()) {
                mp->increaseVisibility(); // why add this? no idea
                // setting current frame as last seen frame removes these points
                // from projecting since they are already tracked in this frame.
                mp->setTrackedInFrame(this->current_frame_->id());
            } else { // remove if bad
                current_frame_->removeMapPointAt(idx);
            }
        }
    }

    int n_mp_in_view = 0;
    // project points that are in local map and not already matched in this frame
    // also check their visibility
    for (const auto& mp: map_points_local_map_) {
        // if already tracked or bad, ignore...
        if(mp->trackedInFrame(this->current_frame_->id()) || mp->isBad())
            continue;

        // see if the point is within camera view
        TrackProperties track_results;
        mp->resetTrackProperties();
        if (current_frame_->isInCameraView(mp, track_results, 0.5)) {
            mp->setTrackProperties(track_results);
            // increase visibility again... where is this used?
            mp->increaseVisibility();
            // number of points in local map that are in view of current
            // camera frame
            n_mp_in_view++;
        }
    }

    ROS_DEBUG_STREAM("n_mp_in_view:" << n_mp_in_view);
    if (n_mp_in_view > 0) {
        int radius = 1;
        if (camera_->type() == geometry::CameraType::RGBD) {
            radius = 3;
        }

        // project points found in camera view to image and match them using
        // orb features
        ROS_DEBUG("Matching frame with local map points by projection...");

        // use_track_info here means the points are already projected on image
        // and so TrackProperties of the map point should be used.
        // this is through by isInCameraView
        bool compute_track_info = false;
        current_frame_->matchByProjection(
            map_points_local_map_, compute_track_info, 0.8, radius);
        const auto& matches = current_frame_->localMatches();
        ROS_DEBUG_STREAM("Matches with local map: " << matches.size());
        if (matches.size() < MIN_REQ_MATCHES)
            return false;

        //current_frame_->showMatchesWithRef("Matched points.");
        //cv::waitKey(0);
        //ROS_DEBUG_STREAM("Adding resultant map points to map.");

        for (const auto& m: matches) {
            // add matched map points from reference to current frame
            current_frame_->setMapPointAt(
                map_points_local_map_[m.trainIdx],
                m.queryIdx);
        }
        return true;
    }
    return false;
}

bool Tracker::relocalize()
{
    /*ROS_DEBUG_STREAM("Computing orb bow features...");
    // compute bag of words vector for current frame
    current_frame_->computeBow();
    ROS_DEBUG_STREAM("Bow: " << current_frame_->bow().size());
    ROS_DEBUG_STREAM("Bow Features: " << current_frame_->bowFeatures().size());

    // relocalization is performed when tracking is lost
    // query key frame database to find keyframe candidates for relocalization
    auto key_frames_ =
        mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame);

    if(vpCandidateKFs.empty())
        return false;

    const int nKFs = vpCandidateKFs.size();

    // We perform first an ORB matching with each candidate
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.75,true);

    vector<PnPsolver*> vpPnPsolvers;
    vpPnPsolvers.resize(nKFs);

    vector<vector<MapPoint*> > vvpMapPointMatches;
    vvpMapPointMatches.resize(nKFs);

    vector<bool> vbDiscarded;
    vbDiscarded.resize(nKFs);

    int nCandidates=0;

    for(int i=0; i<nKFs; i++)
    {
        KeyFrame* pKF = vpCandidateKFs[i];
        if(pKF->isBad())
            vbDiscarded[i] = true;
        else
        {
            int nmatches = matcher.SearchByBoW(pKF,mCurrentFrame,vvpMapPointMatches[i]);
            if(nmatches<15)
            {
                vbDiscarded[i] = true;
                continue;
            }
            else
            {
                PnPsolver* pSolver = new PnPsolver(mCurrentFrame,vvpMapPointMatches[i]);
                pSolver->SetRansacParameters(0.99,10,300,4,0.5,5.991);
                vpPnPsolvers[i] = pSolver;
                nCandidates++;
            }
        }
    }

    // Alternatively perform some iterations of P4P RANSAC
    // Until we found a camera pose supported by enough inliers
    bool bMatch = false;
    ORBmatcher matcher2(0.9,true);

    while(nCandidates>0 && !bMatch)
    {
        for(int i=0; i<nKFs; i++)
        {
            if(vbDiscarded[i])
                continue;

            // Perform 5 Ransac Iterations
            vector<bool> vbInliers;
            int nInliers;
            bool bNoMore;

            PnPsolver* pSolver = vpPnPsolvers[i];
            cv::Mat Tcw = pSolver->iterate(5,bNoMore,vbInliers,nInliers);

            // If Ransac reachs max. iterations discard keyframe
            if(bNoMore)
            {
                vbDiscarded[i]=true;
                nCandidates--;
            }

            // If a Camera Pose is computed, optimize
            if(!Tcw.empty())
            {
                Tcw.copyTo(mCurrentFrame.mTcw);

                set<MapPoint*> sFound;

                const int np = vbInliers.size();

                for(int j=0; j<np; j++)
                {
                    if(vbInliers[j])
                    {
                        mCurrentFrame.mvpMapPoints[j]=vvpMapPointMatches[i][j];
                        sFound.insert(vvpMapPointMatches[i][j]);
                    }
                    else
                        mCurrentFrame.mvpMapPoints[j]=NULL;
                }

                int nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                if(nGood<10)
                    continue;

                for(int io =0; io<mCurrentFrame.N; io++)
                    if(mCurrentFrame.mvbOutlier[io])
                        mCurrentFrame.mvpMapPoints[io]=static_cast<MapPoint*>(NULL);

                // If few inliers, search by projection in a coarse window and optimize again
                if(nGood<50)
                {
                    int nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,10,100);

                    if(nadditional+nGood>=50)
                    {
                        nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                        // If many inliers but still not enough, search by projection again in a narrower window
                        // the camera has been already optimized with many points
                        if(nGood>30 && nGood<50)
                        {
                            sFound.clear();
                            for(int ip =0; ip<mCurrentFrame.N; ip++)
                                if(mCurrentFrame.mvpMapPoints[ip])
                                    sFound.insert(mCurrentFrame.mvpMapPoints[ip]);
                            nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,3,64);

                            // Final optimization
                            if(nGood+nadditional>=50)
                            {
                                nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                                for(int io =0; io<mCurrentFrame.N; io++)
                                    if(mCurrentFrame.mvbOutlier[io])
                                        mCurrentFrame.mvpMapPoints[io]=NULL;
                            }
                        }
                    }
                }


                // If the pose is supported by enough inliers stop ransacs and continue
                if(nGood>=50)
                {
                    bMatch = true;
                    break;
                }
            }
        }
    }

    if(!bMatch)
    {
        return false;
    }
    else
    {
        mnLastRelocFrameId = mCurrentFrame.mnId;
        return true;
    }
    */
}

bool Tracker::needNewKeyFrame()
{
    // if local mapper is stopped or a stop request has been initiated
    // by a loop closure, do not insert keyframes
    if(local_mapper_->stopped() || local_mapper_->stopRequested())
        return false;

    const int n_key_frames = map_->nKeyFrames();

    // do not insert keyframes if not enough frames have passed from last
    // relocalization
    //if (current_frame_->id() < mnLastRelocFrameId + mMaxFrames && n_key_frames > mMaxFrames)
    //    return false;

    // find the map points in this key frame that are tracked by at least
    // n_min_obs key frames
    int n_min_obs = 3;
    if (n_key_frames <= 2)
        n_min_obs = 2;
    ROS_DEBUG_STREAM("Checking number of tracked points...");
    int n_tracked_in_ref = ref_key_frame_->nTrackedPoints(n_min_obs);
    ROS_DEBUG_STREAM("n_tracked_in_ref:" << n_tracked_in_ref);

    // is local mapper not busy?
    auto local_mapper_idle = !local_mapper_->isBusy();

    // check how many "close" points are being tracked and how many could
    // be potentially created
    int n_not_tracked_close = 0;
    int n_tracked_close = 0;

    ROS_DEBUG_STREAM("Checking points that are within track and are also close...");
    const auto map_points = current_frame_->obsMapPoints();
    const auto& outliers = current_frame_->outliers();
    const auto& depths = current_frame_->featureDepthsUndist();
    if (camera_->type() != geometry::CameraType::MONO)  {
        for (int i = 0; i < depths.size(); ++i) {
            if (depths[i] > 0 && depths[i] < close_depth_threshold) {
                if (map_points[i] && !outliers[i]) n_tracked_close++;
                else n_not_tracked_close++;
            }
        }
    }

    bool need_to_insert_close =
        (n_tracked_close < 100) && (n_not_tracked_close > 70);

    // thresholds
    float ref_ratio_threshold = 0.75f;
    if (n_key_frames < 2)
        ref_ratio_threshold = 0.4f;

    if (camera_->type() == geometry::CameraType::MONO)
        ref_ratio_threshold = 0.9f;

    ROS_DEBUG_STREAM("Checking conditions...");
    ROS_DEBUG_STREAM("last_key_frame_:" << last_key_frame_);
    ROS_DEBUG_STREAM("current_frame_->id():" << current_frame_->id());
    ROS_DEBUG_STREAM("last_key_frame_->id():" << last_key_frame_->id());
    ROS_DEBUG_STREAM("local_mapper_idle:" << local_mapper_idle);
    ROS_DEBUG_STREAM("need_to_insert_close:" << need_to_insert_close);

    // condition 1a: more than n_max_frames have passed since
    // last keyframe insertion
    const bool c1a =
        current_frame_->id() >= last_key_frame_->id() + n_max_frames_;

    // condition 1b: more than n_min_frames have passed and local mapper is idle
    const bool c1b =
        current_frame_->id() >= last_key_frame_->id() + n_min_frames_ &&
        local_mapper_idle;

    // condition 1c: tracking is weak
    const bool c1c =
        camera_->type() != geometry::CameraType::MONO &&
        (local_map_matches_ < n_tracked_in_ref * 0.25 || need_to_insert_close);

    // condition 2: few tracked points compared to reference keyframe.
    // lots of visual odometry compared to map matches.
    const bool c2 =
        ((local_map_matches_ < n_tracked_in_ref * ref_ratio_threshold ||
            need_to_insert_close) &&
        local_map_matches_ > 15);

    ROS_DEBUG_STREAM("c1:" << c1a);
    ROS_DEBUG_STREAM("c1:" << c1b);
    ROS_DEBUG_STREAM("c1:" << c1c);
    ROS_DEBUG_STREAM("c2:" << c2);
    if((c1a || c1b || c1c) && c2)
    {
        ROS_DEBUG_STREAM("Need new key frame...");
        // if the local mapper is not busy, insert keyframe.
        // otherwise send a signal to interrupt bundle adjustmnet
        if(local_mapper_idle) {
            ROS_DEBUG_STREAM("Local mapper is idle...");
            return true;
        } else {
            ROS_DEBUG_STREAM("Aborting local mapper BA...");
            local_mapper_->abortBA();
            if(camera_->type() != geometry::CameraType::MONO) {
                if (local_mapper_->nKeyFramesInQueue() < 3)
                    return true;
                else
                    return false;
            } else {
                return false;
            }
        }
    } else {
        ROS_DEBUG_STREAM("Key frame not needed...");
        return false;
    }
}

void Tracker::createNewKeyFrame()
{
    if (local_mapper_->stopped()) // return if local mapper is stopped.
        return;

    // If it is not stopped, make it unstoppable until a new key frame is added
    local_mapper_->setStoppable(false);

    // create a new key frame from frame
    auto current_key_frame =
        KeyFramePtr(
            new KeyFrame(current_frame_, map_));

    // set reference key frame as this one
    ref_key_frame_ = current_key_frame;
    current_frame_->setRefKeyFrame(ref_key_frame_); // why do this?

    if(camera_->type() != geometry::CameraType::MONO) {
        // create new map points for all those features that are not matched in
        // another frame/key frame but are still very close.
        std::map<float, int, std::less<float>> depth_indices;
        const auto& depths = current_frame_->featureDepthsUndist();
        const auto& key_points = current_frame_->featuresUndist();
        for (int idx = 0; idx < current_frame_->nFeaturesUndist(); idx++) {
            auto depth = depths[idx];
            if (depth > 0)
                depth_indices.insert(std::pair<float, int>(depth, idx));
        }

        // if there are some points
        if (!depth_indices.empty()) {
            int n_points = 0;
            const auto map_points = current_frame_->obsMapPoints();

            // add all points with depths smaller than depth threshold to key
            // frame
            for (const auto& depth_idx: depth_indices) {
                const auto& idx = depth_idx.second;

                // see if the map point for this feature index is not already
                // assigned
                bool create_new_point = false;
                const auto& mp = map_points[idx];
                if (!mp) { // if map point does not exist
                    create_new_point = true;
                } else if (mp->nObservations() < 1) {
                    // if map point is observed in zero key frames
                    current_frame_->removeMapPointAt(idx);
                    create_new_point = true;
                }

                if (create_new_point) {
                    // transform key point to 3d world space
                    auto world_pos =
                        cv::Mat(
                            current_frame_->frameToWorld<float, float>(
                                key_points[idx].pt, depth_idx.first));

                    // create a map point given 2d-3d correspondence
                    auto mp =
                        MapPointPtr(
                            new MapPoint(world_pos, current_key_frame, map_));

                    // add the map point in local observation map of key frame
                    // i corresponds to index of the map point here
                    current_key_frame->setMapPointAt(mp, idx);

                    // add the key frame to map point in which it is observed
                    // i corresponds to index of the key point in frame class
                    mp->addObservation(current_key_frame, idx);

                    // compute map point best descriptor out of all observing key frames
                    mp->computeBestDescriptor();

                    // compute normal vector, and scale distance for map point
                    mp->updateNormalAndScale();

                    // add map points to map and to reference map
                    map_->addMapPoint(mp);

                    // add these map points to unmatched_map_points_
                    current_frame_->addUnmatchedMapPoint(mp);
                }
                n_points++;

                // if the depth is over minimum depth threshold and 100 points
                // are added then stop adding new points, otherwise keep adding
                // more points until a minimum depth threshold is reached
                if(depth_idx.first > close_depth_threshold && n_points > 100)
                    break;
            }
        }
    }

    // block all operations of frame that can cause conflicts
    // with other threads since after addition to local map the key frame can be
    // accessed by other threads simultaneously
    current_key_frame->frame()->setThreadSafe(true);

    // add the key frame to local mapper
    local_mapper_->addKeyFrameToQueue(current_key_frame);

    // make it stoppable now
    local_mapper_->setStoppable(true);

    last_key_frame_ = current_key_frame;
}

} // namespace orb_slam