#pragma once

#include "llamaLite/llamaLite.hpp"
#include "spearhed/param/speciesDefinition.param"

namespace spearhed
{

    namespace particleView
    {
        using PartDescT = decltype(particleDesc);

        using SoaT = ll::SoA<PartDescT::ParticleRecord, PartDescT::numSlots>;

    } // namespace particleView

    template<auto... TagInstances>
    using ParticleView
        = ll::SoAIndexedView<particleView::SoaT, ll::to_path_t<std::remove_cvref_t<decltype(TagInstances)>>...>;

    /**
     * Alias for creating a ConstView type using constexpr tag INSTANCES (values).
     */
    template<auto... TagInstances>
    using ParticleViewConst
        = ll::SoAIndexedView<particleView::SoaT const, ll::to_path_t<std::remove_cvref_t<decltype(TagInstances)>>...>;
} // namespace spearhed
