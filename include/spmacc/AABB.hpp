#pragma once

#include <pmacc/math/vector/Vector.hpp>

namespace pmacc::spearhed
{
    /**
     * Axis aligned bounding box
     */
    template<typename TAxis, unsigned DIM>
    struct AABB
    {
        using Vec = math::Vector<TAxis, DIM>;

        // This may need to be optimized later
        friend constexpr bool intersects(const AABB& a, const AABB& b)
        {
            for(unsigned i = 0; i < DIM; ++i)
            {
                // Check for separation along axis i
                if(a.min[i] > b.max[i] || a.max[i] < b.min[i])
                {
                    return false;
                }
            }
            return true;
        }

    private:
        Vec min;
        Vec max;
    };
} // namespace pmacc::spearhed
