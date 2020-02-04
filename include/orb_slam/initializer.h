/**
 * Declares the Initializer class.
 */

#include "orb_slam/frame.h"

namespace orb_slam {

/**
 * The initializer used in monocular slam.
 */
class Initializer
{
public:
    Initializer(
        const FramePtr& frame, double sigma = 1.0, int iterations = 200) :
        frame_(frame), sigma_(sigma), iterations_(iterations)
    {
        sigma_squared_ = sigma * sigma;
    }

    ~Initializer() {

    }

private:
    FramePtr frame_; // The initial reference frame
    double sigma_; // std dev?
    double sigma_squared_; // variance?
    int iterations_; // total iterations
};

} // namespace orb_slam