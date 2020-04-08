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
    Viewer(const MapPtr& map) {
        map_ = std::const_pointer_cast<const Map>(map);
        viewer_thread_ = std::thread(std::bind(&Viewer::update, this));
    }

    ~Viewer() {
        viewer_thread_.join();
    }

    /**
     * The update loop of this class.
     */
    void update() {
        // taken from slam book viewer class
        pangolin::CreateWindowAndBind("ORB SLAM", 1024, 768);
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        pangolin::OpenGlRenderState vis_camera(
            pangolin::ProjectionMatrix(
                1024, 768, 400, 400, 512, 384, 0.1, 1000),
            pangolin::ModelViewLookAt(
                0, -5, -10, 0, 0, 0, 0.0, -1.0, 0.0));

        // add named OpenGL viewport to window and provide 3d handler
        pangolin::View& vis_display =
            pangolin::CreateDisplay()
                .SetBounds(0.0, 1.0, 0.0, 1.0, -1024.0f / 768.0f)
                .SetHandler(new pangolin::Handler3D(vis_camera));

        const float blue[3] = {0, 0, 1};
        const float green[3] = {0, 1, 0};

        while (!pangolin::ShouldQuit() && close_) {
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
            vis_display.Activate(vis_camera);

            { // shared resources
                LOCK_VIEWER;
                if (frame_) {
                    drawFrame(frame_, green);
                    followCurrentFrame(vis_camera);

                    cv::Mat img = plotFrameImage();
                    cv::imshow("image", img);
                    cv::waitKey(1);
                }

                if (map_) {
                    drawMapPoints();
                }
            }
            pangolin::FinishFrame();
            usleep(5000);
        }
        ROS_DEBUG_STREAM("Viewer stopped!");
    }

    cv::Mat plotFrameImage();
    void drawFrame(const FrameConstPtr& frame, const float* color);
    void followCurrentFrame(pangolin::OpenGlRenderState& vis_camera);
    void drawMapPoints();

    /**
     * Adds a frame to the viewer.
     * @param frame: Camera frame
     */
    void addFrame(const FramePtr& frame) {
        LOCK_VIEWER;
        frame_ = std::const_pointer_cast<const Frame>(frame);
    }

    /**
     * Updates the map with current data
     */
    void updateMap() {
        LOCK_VIEWER;
        if (map_) {
            key_frames_ = map_->keyFrames();
            map_points_ = map_->mapPoints();
        }

    }

    void close() {
        LOCK_VIEWER;
        close_ = false;
    }

private:
    bool close_; // whether to close the viewer

    std::vector<KeyFramePtr> key_frames_;
    std::vector<MapPointPtr> map_points_;
    MapConstPtr map_; // Pointer to map
    FrameConstPtr frame_; // Pointer to current frame
    std::thread viewer_thread_; // Thread of this class
    std::mutex viewer_mutex_; // Thread access mutex
};

cv::Mat Viewer::plotFrameImage() {
    cv::Mat img_out;
    cv::cvtColor(
        static_pointer_cast<RGBDFrame>(frame_)->image()->image,
        img_out,
        CV_GRAY2BGR);
    const auto& key_points = frame_->featuresUndist();
    const auto& key_map_points = frame_->obsMapPoints();
    for (size_t i = 0; i < key_points.size(); ++i) {
        if (key_map_points[i]) {
            auto feat = key_points[i];
            cv::circle(img_out, key_points[i].pt, 2, cv::Scalar(0, 250, 0), 2);
        }
    }
    return img_out;
}

void Viewer::followCurrentFrame(pangolin::OpenGlRenderState& vis_camera) {
    Eigen::Matrix4f w_T_c;
    cv::cv2eigen(frame_->getCamInWorldT(), w_T_c);
    pangolin::OpenGlMatrix m(w_T_c);
    vis_camera.Follow(m, true);
}

void Viewer::drawFrame(const FrameConstPtr& frame, const float* color)
{
    Eigen::Matrix4f w_T_c;
    cv::cv2eigen(frame->getCamInWorldT(), w_T_c);
    const float sz = 1.0;
    const int line_width = 2.0;
    const float fx = 400;
    const float fy = 400;
    const float cx = 512;
    const float cy = 384;
    const float width = 1080;
    const float height = 768;

    glPushMatrix();

    Sophus::Matrix4f m = w_T_c;
    glMultMatrixf((GLfloat*)m.data());

    if (color == nullptr) {
        glColor3f(1, 0, 0);
    } else
        glColor3f(color[0], color[1], color[2]);

    glLineWidth(line_width);
    glBegin(GL_LINES);
    glVertex3f(0, 0, 0);
    glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
    glVertex3f(0, 0, 0);
    glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
    glVertex3f(0, 0, 0);
    glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
    glVertex3f(0, 0, 0);
    glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);

    glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
    glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);

    glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
    glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);

    glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
    glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);

    glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
    glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);

    glEnd();
    glPopMatrix();
}


void Viewer::drawMapPoints() {
    const float red[3] = {1.0, 0, 0};
    for (auto& kf : key_frames_) {
        drawFrame(static_pointer_cast<const Frame>(kf), red);
    }

    glPointSize(2);
    glBegin(GL_POINTS);
    for (auto& mp : map_points_) {
        Eigen::Vector3f pos;
        cv::cv2eigen(mp->worldPos(), pos);
        glColor3f(red[0], red[1], red[2]);
        glVertex3d(pos[0], pos[1], pos[2]);
    }
    glEnd();
}

} // namespace orb_slam

