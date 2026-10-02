#include <llamaLite/llamaLite.hpp>

struct value_t : llama_lite::TagBase
{
};

inline constexpr value_t value{};

int main()
{
    using Record = llama_lite::Record<llama_lite::Field<value_t, int>>;
    llama_lite::SoA<Record, 4> values;
    values[value][0] = 42;
    return values[value][0] == 42 ? 0 : 1;
}
