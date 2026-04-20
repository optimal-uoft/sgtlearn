#include <catch2/catch_test_macros.hpp>

#include "sgtlearn/adder.hpp"

TEST_CASE("Adder adds doubles", "[adder]") {
    const sgtlearn::Adder adder;
    REQUIRE(adder.add(2.0, 3.0) == 5.0);
}
