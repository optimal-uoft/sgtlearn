/**
 * @file bench_mae_branch_assignment.cpp
 * @brief CD wall-time: sort vs BST vs merge AbsoluteError backends.
 *
 * Writes a concise colleague-facing CSV under ``benchmarks/results/``.
 */

#include <BranchAssignmentObjectives/AbsoluteErrorBranchAssignment.h>
#include <BranchAssignmentObjectives/AbsoluteErrorBranchAssignmentBst.h>
#include <BranchAssignmentObjectives/AbsoluteErrorBranchAssignmentSort.h>
#include <algorithms/CoordinateDescent.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using Catch::Matchers::WithinAbs;
using clock_type = std::chrono::steady_clock;

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

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

template <typename MakeObj>
BenchResult timeCd(MaeScenario &base, uint64_t seed, int repeats,
                   MakeObj &&makeObj) {
  double totalMs = 0.0;
  double lastObj = 0.0;
  for (int r = 0; r < repeats; ++r) {
    auto asg = base.assignments;
    auto obj = makeObj(asg);
    std::mt19937_64 rng(seed + static_cast<uint64_t>(r));
    const auto t0 = clock_type::now();
    lastObj = coordinateDescent(base.numPartitions, obj, rng, 8, 3);
    const auto t1 = clock_type::now();
    totalMs += std::chrono::duration<double, std::milli>(t1 - t0).count();
  }
  return {totalMs / static_cast<double>(repeats), lastObj};
}

std::filesystem::path resultsDir() {
  namespace fs = std::filesystem;
  const fs::path candidates[] = {
      fs::path("benchmarks") / "results",
      fs::path("..") / "benchmarks" / "results",
      fs::path("..") / ".." / "benchmarks" / "results",
  };
  for (const auto &p : candidates) {
    std::error_code ec;
    if (fs::exists(p.parent_path(), ec))
      return p;
  }
  return fs::path("benchmarks") / "results";
}

} // namespace

TEST_CASE("AbsoluteError BST/sort/merge objectives match under CD",
          "[branch_assignment][absolute_error][correctness]") {
  auto scenario = makeScenario(/*numBins=*/32, /*samplesPerBin=*/40,
                               /*numPartitions=*/4, /*nOutputs=*/1, /*seed=*/99);

  auto asgBst = scenario.assignments;
  auto asgSort = scenario.assignments;
  auto asgMerge = scenario.assignments;

  AbsoluteErrorBranchAssignmentBst bst(asgBst, scenario.numPartitions,
                                       scenario.leafYs, scenario.leafWs,
                                       scenario.leafWeights,
                                       scenario.leafSampleCounts);
  AbsoluteErrorBranchAssignmentSort sortObj(asgSort, scenario.numPartitions,
                                            scenario.leafYs, scenario.leafWs,
                                            scenario.leafWeights,
                                            scenario.leafSampleCounts);
  AbsoluteErrorBranchAssignment mergeObj(asgMerge, scenario.numPartitions,
                                         scenario.leafYs, scenario.leafWs,
                                         scenario.leafWeights,
                                         scenario.leafSampleCounts);

  REQUIRE_THAT(bst.objective(), WithinAbs(sortObj.objective(), 1e-6));
  REQUIRE_THAT(mergeObj.objective(), WithinAbs(sortObj.objective(), 1e-6));

  std::mt19937_64 rngBst(123);
  std::mt19937_64 rngSort(123);
  std::mt19937_64 rngMerge(123);
  const double bstFinal =
      coordinateDescent(scenario.numPartitions, bst, rngBst, 8, 3);
  const double sortFinal =
      coordinateDescent(scenario.numPartitions, sortObj, rngSort, 8, 3);
  const double mergeFinal =
      coordinateDescent(scenario.numPartitions, mergeObj, rngMerge, 8, 3);
  REQUIRE_THAT(bstFinal, WithinAbs(sortFinal, 1e-5));
  REQUIRE_THAT(mergeFinal, WithinAbs(sortFinal, 1e-5));
  REQUIRE(asgBst == asgSort);
  REQUIRE(asgMerge == asgSort);
}

TEST_CASE("Bench AbsoluteError branch assignment: sort vs BST vs merge",
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
      {"small_64x20", 64, 20, 4, 1, 5},
      {"medium_128x50", 128, 50, 4, 1, 3},
      {"large_256x100", 256, 100, 8, 1, 2},
      {"multiout_128x40x3", 128, 40, 4, 3, 3},
  };

  const auto outDir = resultsDir();
  std::filesystem::create_directories(outDir);
  const auto csvPath = outDir / "mae_branch_cd_comparison.csv";
  std::ofstream csv(csvPath);
  csv << "case,n_bins,samples_per_bin,n_partitions,n_outputs,"
         "sort_ms,bst_ms,merge_ms,"
         "bst_speedup_vs_sort,merge_speedup_vs_sort,"
         "sort_obj,bst_obj,merge_obj\n";

  std::cout << '\n'
            << "AbsoluteError CD bench: sort | bst | merge (default)\n"
            << "CSV -> " << csvPath << '\n'
            << "--------------------------------------------------------------"
               "--------\n";

  for (const Case &c : cases) {
    auto scenario =
        makeScenario(c.bins, c.samplesPerBin, c.parts, c.outputs, 2026);

    const auto sortRes = timeCd(scenario, 7, c.repeats, [&](auto &asg) {
      return AbsoluteErrorBranchAssignmentSort(
          asg, scenario.numPartitions, scenario.leafYs, scenario.leafWs,
          scenario.leafWeights, scenario.leafSampleCounts);
    });
    const auto bstRes = timeCd(scenario, 7, c.repeats, [&](auto &asg) {
      return AbsoluteErrorBranchAssignmentBst(
          asg, scenario.numPartitions, scenario.leafYs, scenario.leafWs,
          scenario.leafWeights, scenario.leafSampleCounts);
    });
    const auto mergeRes = timeCd(scenario, 7, c.repeats, [&](auto &asg) {
      return AbsoluteErrorBranchAssignment(
          asg, scenario.numPartitions, scenario.leafYs, scenario.leafWs,
          scenario.leafWeights, scenario.leafSampleCounts);
    });

    const double bstSpeedup =
        bstRes.ms > 0.0 ? (sortRes.ms / bstRes.ms) : 0.0;
    const double mergeSpeedup =
        mergeRes.ms > 0.0 ? (sortRes.ms / mergeRes.ms) : 0.0;

    std::cout << std::fixed << std::setprecision(2) << c.name << ": sort "
              << sortRes.ms << " ms | bst " << bstRes.ms << " ms ("
              << bstSpeedup << "x) | merge " << mergeRes.ms << " ms ("
              << mergeSpeedup << "x)\n";

    csv << std::fixed << std::setprecision(3) << c.name << ',' << c.bins << ','
        << c.samplesPerBin << ',' << c.parts << ',' << c.outputs << ','
        << sortRes.ms << ',' << bstRes.ms << ',' << mergeRes.ms << ','
        << std::setprecision(3) << bstSpeedup << ',' << mergeSpeedup << ','
        << std::setprecision(6) << sortRes.objective << ',' << bstRes.objective
        << ',' << mergeRes.objective << '\n';

    REQUIRE(sortRes.ms > 0.0);
    REQUIRE(bstRes.ms > 0.0);
    REQUIRE(mergeRes.ms > 0.0);
    REQUIRE_THAT(bstRes.objective, WithinAbs(sortRes.objective, 1e-4));
    REQUIRE_THAT(mergeRes.objective, WithinAbs(sortRes.objective, 1e-4));
  }

  csv.flush();
  std::cout << "Wrote " << csvPath << std::endl;
}

#pragma GCC diagnostic pop
#pragma clang diagnostic pop
