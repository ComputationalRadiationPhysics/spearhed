#pragma once

#include <pmacc/particles/memory/dataTypes/Pointer.hpp>

#include <boost/mpl/placeholders.hpp>

namespace pmacc::spearhed
{
    template<typename T_Type = boost::mpl::_1>
    struct NextFramePtr
    {
        PMACC_ALIGN(next, T_Type*);
    };

} // namespace pmacc::spearhed
