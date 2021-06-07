/**
 * Defines the FrameContainer class.
 */

#include <orb_slam/frame_container.h>

namespace orb_slam
{

FrameContainer::FrameContainer(
    const FramePtr& frame, Frame::BehaviorPtr behavior) :
    frame_(frame),
    behavior_(behavior)
{}
const FramePtr& FrameContainer::frame() { return frame_; }
void FrameContainer::setBehavior(Frame::BehaviorPtr behavior)
    { behavior_ = behavior; }
Frame::BehaviorPtr FrameContainer::behavior() { return behavior_; }

ThreadSafeFrame::ThreadSafeFrame(const FramePtr& frame) :
    FrameContainer(
        frame, std::shared_ptr<Frame::ThreadSafe>(new Frame::ThreadSafe(frame)))
{
}
Frame::ThreadSafePtr ThreadSafeFrame::behavior()
    { return static_pointer_cast<Frame::ThreadSafe>(behavior_); }

ThreadUnsafeFrame::ThreadUnsafeFrame(const FramePtr& frame) :
    FrameContainer(
        frame, std::shared_ptr<Frame::ThreadUnsafe>(new Frame::ThreadUnsafe(frame)))
{}
Frame::ThreadUnsafePtr ThreadUnsafeFrame::behavior()
    { return static_pointer_cast<Frame::ThreadUnsafe>(behavior_); }

} // namespace orb_slam