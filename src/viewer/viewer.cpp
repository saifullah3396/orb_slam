/**
 * Defines the Viewer class.
 */

#include <thread>
#include <orb_slam/viewer/viewer.h>
#include <orb_slam/map.h>
#include <orb_slam/frame.h>
#include <orb_slam/key_frame.h>

namespace orb_slam
{

Viewer::Viewer(const MapPtr& map) {
    map_ = std::const_pointer_cast<const Map>(map);
}

Viewer::~Viewer() {
    viewer_thread_.join();
}

void Viewer::startThread() {
    viewer_thread_ = std::thread(std::bind(&Viewer::threadCall, this));
}

void Viewer::threadCall() {
    setup();
    while (!pangolin::ShouldQuit() && !close_) {
        draw();
        usleep(5000);
    }
    //ROS_DEBUG_STREAM_NAMED(name_tag_, "Viewer stopped!");
}

void Viewer::setup() {
    // taken from slam book viewer class
    pangolin::CreateWindowAndBind("ORB SLAM", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    vis_camera_ = pangolin::OpenGlRenderState(
        pangolin::ProjectionMatrix(
            1024, 768, 400, 400, 512, 384, 0.1, 1000),
        pangolin::ModelViewLookAt(
            0, -5, -10, 0, 0, 0, 0.0, -1.0, 0.0));

    // add named OpenGL viewport to window and provide 3d handler
    vis_display_ =
        pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, 0.0, 1.0, -1024.0f / 768.0f)
            .SetHandler(new pangolin::Handler3D(vis_camera_));

}

void Viewer::draw() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    vis_display_.Activate(vis_camera_);
    { // shared resources
        LOCK_VIEWER;
        if (frame_) {
            drawFrame(frame_, green_);
            followCurrentFrame(vis_camera_);

            frame_->showImageWithFeatures("Frame");
            cv::waitKey(1);
        }

        if (map_) {
            drawMapPoints();
        }
    }
    pangolin::FinishFrame();
}

void Viewer::addFrame(const FramePtr& frame) {
    LOCK_VIEWER;
    frame_ = std::const_pointer_cast<const Frame>(frame);
}

void Viewer::updateMap() {
    LOCK_VIEWER;
    if (map_) {
        key_frames_ = map_->keyFrames();
        map_points_ = map_->mapPoints();
    }

}

void Viewer::close() {
    LOCK_VIEWER;
    close_ = false;
}

void Viewer::followCurrentFrame(pangolin::OpenGlRenderState& vis_camera) {
    Eigen::Matrix4f w_T_c;
    cv::cv2eigen(frame_->cameraInWorldT(), w_T_c);
    pangolin::OpenGlMatrix m(w_T_c);
    vis_camera.Follow(m, true);
}

void Viewer::drawFrame(const FrameConstPtr& frame, const float* color)
{
    Eigen::Matrix4f w_T_c;
    cv::cv2eigen(frame->cameraInWorldT(), w_T_c);
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
        drawFrame(static_pointer_cast<const Frame>(kf->frame()), red);
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

