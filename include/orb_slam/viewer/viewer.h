/**
 * Defines the Viewer class.
 */

#pragma once

#include <thread>
#include <orb_slam/map.h>
#include <orb_slam/frame.h>
#include <pangolin/pangolin.h>
#include <sophus/se3.hpp>
#include <opencv2/core/eigen.hpp>

namespace orb_slam
{

#define LOCK_VIEWER std::unique_lock<std::mutex> lock(viewer_mutex_)

class Viewer
{
public:
    Viewer(const MapPtr& map);
    ~Viewer();

    void startThread();
    void threadCall();

    /**
     * Performs viewer setup
     */
    void setup();

    /**
     * The update loop of this class.
     */
    void draw();

    cv::Mat plotFrameImage();
    void drawFrame(const FrameConstPtr& frame, const float* color);
    void followCurrentFrame(pangolin::OpenGlRenderState& vis_camera);
    void drawMapPoints();

    /**
     * Adds a frame to the viewer.
     * @param frame: Camera frame
     */
    void addFrame(const FramePtr& frame);

    /**
     * Updates the map with current data
     */
    void updateMap();
    void close();

private:
    bool close_ = {false}; // whether to close the viewer

    pangolin::View vis_display_;
    pangolin::OpenGlRenderState vis_camera_;

    const float blue_[3] = {0, 0, 1};
    const float green_[3] = {0, 1, 0};

    std::set<KeyFramePtr> key_frames_;
    std::set<MapPointPtr> map_points_;
    MapConstPtr map_; // Pointer to map
    FrameConstPtr frame_; // Pointer to current frame
    std::thread viewer_thread_; // Thread of this class
    std::mutex viewer_mutex_; // Thread access mutex

    std::string name_tag_ = {"Viewer"};
};

} // namespace orb_slam

