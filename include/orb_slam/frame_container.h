/**
 * Defines the FrameContainer class.
 */

#include <orb_slam/frame.h>

namespace orb_slam
{

class FrameContainer {
public:
    FrameContainer(
        const FramePtr& frame, Frame::BehaviorPtr behavior);
    const FramePtr& frame();
    void setBehavior(Frame::BehaviorPtr behavior);
    Frame::BehaviorPtr behavior();

protected:
    FramePtr frame_;
    Frame::BehaviorPtr behavior_;
};

class ThreadSafeFrame : public FrameContainer {
public:
    ThreadSafeFrame(const FramePtr& frame);
    Frame::ThreadSafePtr behavior();
};

class ThreadUnsafeFrame : public FrameContainer {
public:
    ThreadUnsafeFrame(const FramePtr& frame);
    Frame::ThreadUnsafePtr behavior();
};

using FrameContainerPtr = std::shared_ptr<FrameContainer>;
using ThreadSafeFramePtr = std::shared_ptr<ThreadSafeFrame>;
using ThreadUnsafeFramePtr = std::shared_ptr<ThreadUnsafeFrame>;

} // namespace orb_slam