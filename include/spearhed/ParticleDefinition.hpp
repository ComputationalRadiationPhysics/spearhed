#pragma once

#include "spearhed/param/speciesDefinition.param"
#include "spmacc/Frame.hpp"
#include "spmacc/ListPointer.hpp"
#include "spmacc/meta/ComponentList.hpp"

#include <pmacc/meta/Pair.hpp>
#include <pmacc/meta/conversion/MakeSeq.hpp>
#include <pmacc/particles/memory/dataTypes/StaticArray.hpp>

namespace spearhed
{
    /** linked list pointer */
    using LinkedListPointer = pmacc::spearhed::meta::ComponentList<pmacc::spearhed::NextFramePtr>;

    /* extent particle description with pointer to a frame*/
    using FrameDescription = decltype(particleDesc.replaceFrameExtensionSeq<LinkedListPointer>());

    /** frame definition
     *
     * a group of particles is stored as frame
     */
    using FrameType = pmacc::spearhed::Frame<FrameDescription>;

} // namespace spearhed
