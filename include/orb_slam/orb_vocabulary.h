/**
 * Defines the ORBVocabulary wrapper object from DBoW2
 */

#pragma once

#include <memory>
#include <DBoW2/FORB.h>
#include <DBoW2/TemplatedVocabulary.h>

namespace orb_slam {

using ORBVocabulary =
    DBoW2::TemplatedVocabulary<DBoW2::FORB::TDescriptor, DBoW2::FORB>;
using ORBVocabularyPtr = std::shared_ptr<ORBVocabulary>;
using ORBVocabularyConstPtr = std::shared_ptr<const ORBVocabulary>;

} // namespace orb_slam
