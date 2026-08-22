/**
 * @file bench_mae_branch_assignment.cpp
 * @brief Wall-time comparison: sort-based vs BST AbsoluteError branch assignment.
 */

#include <BranchAssignmentObjectives/AbsoluteErrorBranchAssignment.h>
#include <algorithms/CoordinateDescent.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <chrono>
#include <iostream>
#include <random>
#include <vector>

using Catch::Matchers::WithinAbs;
using clock_type = std::chrono::steady_clock;

namespace {

struct MaeScenario {
  std::vector<size_t> assignments;
  std::vector<std::vector<std::vector<float>>> leafYs;
  std::vector<std::vector<float>> leafWs;
  std::vector<double> leafWeights;
  std::vector<size_t> leafSampleCounts;
  size_t numPartitions = 0;
};

MaeScenario makeScenario(size_t numBins, size_t samplesPerBin,
                         size_t numPartitions, size_t nOutputs,
                         uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<float> yDist(-100.0F, 100.0F);
  std::uniform_real_distribution<float> wDist(0.5F, 2.0F);

  MaeScenario s;
  s.numPartitions = numPartitions;
  s.assignments.resize(numBins);
  s.leafYs.resize(numBins);
  s.leafWs.resize(numBins);
  s.leafWeights.resize(numBins);
  s.leafSampleCounts.assign(numBins, samplesPerBin);

  for (size_t b = 0; b < numBins; ++b) {
    s.assignments[b] = b % numPartitions;
    s.leafYs[b].resize(nOutputs);
    s.leafWs[b].resize(samplesPerBin);
    double wSum = 0.0;
    for (size_t i = 0; i < samplesPerBin; ++i) {
      s.leafWs[b][i] = wDist(rng);
      wSum += static_cast<double>(s.leafWs[b][i]);
    }
    s.leafWeights[b] = wSum;
    for (size_t o = 0; o < nOutputs; ++o) {
      s.leafYs[b][o].resize(samplesPerBin);
      for (size_t i = 0; i < samplesPerBin; ++i)
        s.leafYs[b][o][i] = yDist(rng);
    }
  }
  return s;
}

struct BenchResult {
  double ms = 0.0;
  double objective = 0.0;
};

BenchResult timeBstCd(MaeScenario &base, uint64_t seed, int repeats) {
  double totalMs = 0.0;
  double lastObj = 0.0;
  for (int r = 0; r < repeats; ++r) {
    auto asg = base.assignments;
    AbsoluteErrorBranchAssignment obj(asg, base.numPartitions, base.leafYs,
                                      base.leafWs, base.leafWeights,
                                      base.leafSampleCounts);
    std::mt19937_64 rng(seed + static_cast<uint64_t>(r));
    const auto t0 = clock_type::now();
    lastObj = coordinateDescent(base.numPartitions, obj, rng, 8, 3);
    const auto t1 = clock_type::now();
    totalMs += std::chrono::duration<double, std::milli>(t1 - t0).count();
  }
  return {totalMs / static_cast<double>(repeats), lastObj};
}

BenchResult timeSortCd(MaeScenario &base, uint64_t seed, int repeats) {
  double totalMs = 0.0;
  double lastObj = 0.0;
  for (int r = 0; r < repeats; ++r) {
    auto asg = base.assignments;
    AbsoluteErrorBranchAssignmentSort obj(asg, base.numPartitions, base.leafYs,
                                          base.leafWs, base.leafWeights,
                                          base.leafSampleCounts);
    std::mt19937_64 rng(seed + static_cast<uint64_t>(r));
    const auto t0 = clock_type::now();
    lastObj = coordinateDescent(base.numPartitions, obj, rng, 8, 3);
    const auto t1 = clock_type::now();
    totalMs += std::chrono::duration<double, std::milli>(t1 - t0).count();
  }
  return {totalMs / static_cast<double>(repeats), lastObj};
}

} // namespace

TEST_CASE("AbsoluteError BST and sort objectives match under CD",
          "[branch_assignment][absolute_error][correctness]") {
  auto scenario = makeScenario(/*numBins=*/32, /*samplesPerBin=*/40,
                               /*numPartitions=*/4, /*nOutputs=*/1, /*seed=*/99);

  auto asgBst = scenario.assignments;
  auto asgSort = scenario.assignments;

  AbsoluteErrorBranchAssignment bst(asgBst, scenario.numPartitions,
                                    scenario.leafYs, scenario.leafWs,
                                    scenario.leafWeights,
                                    scenario.leafSampleCounts);
  AbsoluteErrorBranchAssignmentSort sortObj(asgSort, scenario.numPartitions,
                                            scenario.leafYs, scenario.leafWs,
                                            scenario.leafWeights,
                                            scenario.leafSampleCounts);

  REQUIRE_THAT(bst.objective(), WithinAbs(sortObj.objective(), 1e-6));

  std::mt19937_64 rngBst(123);
  std::mt19937_64 rngSort(123);
  const double bstFinal =
      coordinateDescent(scenario.numPartitions, bst, rngBst, 8, 3);
  const double sortFinal =
      coordinateDescent(scenario.numPartitions, sortObj, rngSort, 8, 3);
  REQUIRE_THAT(bstFinal, WithinAbs(sortFinal, 1e-5));
  REQUIRE(asgBst == asgSort);
}

TEST_CASE("Bench AbsoluteError branch assignment: sort vs BST",
          "[.benchmark]") {
  struct Case {
    const char *name;
    size_t bins;
    size_t samplesPerBin;
    size_t parts;
    size_t outputs;
    int repeats;
  };

  const Case cases[] = {
      {"small  64 bins x 20", 64, 20, 4, 1, 5},
      {"medium 128 bins x 50", 128, 50, 4, 1, 3},
      {"large  256 bins x 100", 256, 100, 8, 1, 2},
      {"multi-out 128 x 40 x 3", 128, 40, 4, 3, 3},
  };

  std::cout << '\n'
            << "AbsoluteError CD bench (sort = re-sort partition; bst = "
               "WeightedMAETree)\n";
  std::cout << "--------------------------------------------------------------"
               "--------\n";

  for (const Case &c : cases) {
    auto scenario =
        makeScenario(c.bins, c.samplesPerBin, c.parts, c.outputs, 2026);
    const auto sortRes = timeSortCd(scenario, 7, c.repeats);
    const auto bstRes = timeBstCd(scenario, 7, c.repeats);
    const double speedup = sortRes.ms > 0.0 ? (sortRes.ms / bstRes.ms) : 0.0;
    std::cout << c.name << ": sort " << sortRes.ms << " ms, bst " << bstRes.ms
              << " ms, speedup " << speedup << "x (obj sort=" << sortRes.objective
              << " bst=" << bstRes.objective << ")\n";

    REQUIRE(bstRes.ms > 0.0);
    REQUIRE(sortRes.ms > 0.0);
  }
  std::cout << std::flush;
}
