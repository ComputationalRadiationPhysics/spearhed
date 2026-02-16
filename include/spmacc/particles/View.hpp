#pragma once

#include "llamaLite/llamaLite.hpp"
#include "spmacc/particles/traits.hpp"

namespace pmacc::spearhed
{
    template<typename TSoA, auto... TagInstances>
    using ParticleView = ll::SoAIndexedView<TSoA, ll::to_path_t<std::remove_cvref_t<decltype(TagInstances)>>...>;

} // namespace pmacc::spearhed
