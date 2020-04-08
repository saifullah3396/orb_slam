/**
 * Implements the KeyFrameDatabase class.
 */

#include <orb_slam/key_frame_database.h>
#include <orb_slam/key_frame.h>
#include <orb_slam/frame.h>

#define LOCK_DATABASE std::unique_lock<std::mutex> db_lock(access_mutex_)

namespace orb_slam
{

KeyFrameDatabase::KeyFrameDatabase(const ORBVocabularyPtr& vocabulary) :
    vocabulary_(vocabulary)
{
    inverted_idxs_.resize(vocabulary_->size());
}


void KeyFrameDatabase::add(const KeyFramePtr& key_frame)
{
    LOCK_DATABASE;
    const auto& bow = key_frame->frame()->bow();
    for (const auto& word : bow)
        inverted_idxs_[word.first].push_back(key_frame);
}

void KeyFrameDatabase::remove(const KeyFramePtr& key_frame)
{
    LOCK_DATABASE;

    // erase elements for the given key frame entry
    const auto& bow = key_frame->frame()->bow();
    for(const auto& word : bow) {
        // list of key frames that share the word
        auto& key_frames = inverted_idxs_[word.first];
        for(auto it = key_frames.begin(); it != key_frames.end(); it++) {
            if (key_frame == *it) { // remove they input key frame from all words
                key_frames.erase(it);
                break;
            }
        }
    }
}

void KeyFrameDatabase::clear()
{
    inverted_idxs_.clear();
    inverted_idxs_.resize(vocabulary_->size());
}

bool KeyFrameDatabase::findRelocCandidates(
    const FramePtr& ref_frame,
    std::vector<KeyFramePtr>& candidate_key_frames)
{
    std::vector<KeyFramePtr> matching_key_frames;
    // search all keyframes that share a word with current frame
    { // shared
        LOCK_DATABASE;

        const auto& bow = ref_frame->bow();
        for (const auto& word : bow) {
            auto& key_frames = inverted_idxs_[word.first];
            for (const auto& kf: key_frames) {
                if (!kf->queriedInFrame(ref_frame->id())) {
                    // if this frame is not already queried, set it as queried
                    // for this frame
                    kf->setQueriedInFrame(ref_frame->id());
                    kf->setNMatchingWords(0);

                    matching_key_frames.push_back(kf);
                }
                kf->increaseNMatchingWords();
            }
        }
    }

    if (matching_key_frames.empty()) // no matches found
        return false;

    // find max and min word matches
    int max_req_word_matches = 0;
    for(const auto& kf: matching_key_frames)
    {
        const auto& n = kf->nMatchingWords();
        if (n > max_req_word_matches)
            max_req_word_matches = n;
    }
    int min_req_word_matches = 0.8f * max_req_word_matches;

    // filter out key frames based on max and min matching words
    std::map<KeyFramePtr, float> key_frame_score_map;

    // compute similarity score for each key frame
    for(const auto& kf: matching_key_frames) {
        if(kf->nMatchingWords() > min_req_word_matches) {
            auto score =
                vocabulary_->score(ref_frame->bow(), kf->frame()->bow());
            kf->setMatchingScore(score);
            key_frame_score_map.insert(
                std::pair<KeyFramePtr, float>(kf, score));
        }
    }

    if(key_frame_score_map.empty())
        return false;

    std::map<KeyFramePtr, float> key_frame_score_acc_map;
    float best_acc_score = 0.f;

    // accumulate the score by covisibility
    for(
        auto it = key_frame_score_map.begin();
        it != key_frame_score_map.end(); it++)
    {
        const auto& kf = it->first;
        auto best_cov_key_frames = vector<KeyFramePtr>();
            //kf->GetBestCovisibilityKeyFrames(10);

        auto best_score = it->second;
        auto acc_score = best_score;
        KeyFramePtr best_kf = kf;
        for (const auto& cov_kf: best_cov_key_frames) {
            if (!cov_kf->queriedInFrame(ref_frame->id()))
                continue;
            acc_score += cov_kf->matchingScore();
            if (cov_kf->matchingScore() > best_score) {
                best_kf = cov_kf;
                best_score = cov_kf->matchingScore();
            }
        }
        key_frame_score_acc_map.insert(
            std::pair<KeyFramePtr, float>(best_kf, acc_score));
        if (acc_score > best_acc_score)
            best_acc_score = acc_score;
    }

    // return all those keyframes with a score higher than 0.75*bestScore
    auto min_score_to_retain = 0.75f * best_acc_score;


    set<KeyFramePtr> key_frames_alr_added;
    candidate_key_frames.reserve(key_frame_score_acc_map.size());
    for (
        auto it = key_frame_score_acc_map.begin();
        it != key_frame_score_acc_map.end(); it++)
    {
        const auto& score = it->second;
        if (score > min_score_to_retain) {
            const auto& kf = it->first;
            if(!key_frames_alr_added.count(kf)) {
                candidate_key_frames.push_back(kf);
                key_frames_alr_added.insert(kf);
            }
        }
    }

    return true;
}

} // namespace orb_slam