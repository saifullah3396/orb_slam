/**
 * Declares the KeyFrameDatabase class.
 */

#pragma once

#include <list>
#include <mutex>
#include <set>
#include <vector>
#include <orb_slam/orb_vocabulary.h>

namespace orb_slam {

class KeyFrame;
using KeyFramePtr = std::shared_ptr<KeyFrame>;

class Frame;
using FramePtr = std::shared_ptr<Frame>;
using FrameConstPtr = std::shared_ptr<const Frame>;

class KeyFrameDatabase
{
public:
    /**
     * Constructor
     *
     * @param vocabulary: Orb vocabulary object which is queried or updated
     */
    KeyFrameDatabase(const ORBVocabularyPtr& vocabulary);

    /**
     * Adds a key frame to the database.
     *
     * @param key_frame: Key frame added
     */
    void add(const KeyFramePtr& key_frame);

    /**
     * Removes a key frame fromt the database.
     *
     * @param key_frame: Key frame removed
     */
    void remove(const KeyFramePtr& key_frame);

    /**
     * Clears the database.
     */
    void clear();

    /**
     * Finds candidate key frames that can be used for relocalization based on
     * the input frame.
     *
     * @param ref_frame: Input reference frame
     * @param candidate_key_frames: Output candidate key frames
     * @returns true if matches are found
     */
    bool findRelocCandidates(
        const FramePtr& ref_frame,
        std::vector<KeyFramePtr>& candidate_key_frames);

    // Loop Detection
    //std::vector<KeyFrame *> DetectLoopCandidates(KeyFrame* pKF, float minScore);

protected:
    const ORBVocabularyPtr vocabulary_; // Associated vocabulary
    std::vector<list<KeyFramePtr>> inverted_idxs_; // Inverted database indices
    std::mutex access_mutex_; // Mutex for database access
};

} //namespace orb_slam
