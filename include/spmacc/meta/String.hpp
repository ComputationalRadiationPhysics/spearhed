#pragma once

#include <algorithm>
#include <cstddef>
#include <string_view>

namespace pmacc::spearhed::meta
{
    namespace internal
    {
        template<size_t N>
        struct FixedString
        {
            char data[N];

            constexpr FixedString(char const (&str)[N])
            {
                std::copy_n(str, N, data);
            }

            [[nodiscard]] constexpr std::string_view view() const noexcept
            {
                // Exclude null terminator
                return {data, N - 1};
            }

            [[nodiscard]] constexpr auto size() const noexcept -> size_t
            {
                return N - 1;
            }

            [[nodiscard]] constexpr auto c_str() const noexcept -> char const*
            {
                return data;
            }
        };
    } // namespace internal

    /** compile time string
     *
     * The size of the instance is 1 byte.
     */
    template<internal::FixedString Str>
    struct String
    {
        static consteval auto view() noexcept -> std::string_view
        {
            return Str.view();
        }

        static consteval auto c_str() noexcept -> char const*
        {
            return Str.c_str();
        }

        static consteval auto size() noexcept -> size_t
        {
            return Str.size();
        }
    };
} // namespace pmacc::spearhed::meta

/** create a compile time string type
 *
 * usage example:
 * @code{.cpp}
 * // create an instance of the compile time string
 * auto particleName = PMACC_CSTRING( "electrons" ){};
 * // create a C++ type (can be used as template parameter)
 * using Electrons = PMACC_CSTRING( "electrons" );
 * @endcode
 */
#define SPMACC_CSTRING(str) pmacc::spearhed::meta::String<str>
