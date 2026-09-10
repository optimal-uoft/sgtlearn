#include "Estimators/ShapeFunctions/ShapeFunctionSplitSearch.h"
#include "Discretizers/univariate/UnivariateDiscretizer.h"
#include "Discretizers/pair/PairClassificationDiscretizer.h"
#include "Discretizers/pair/PairRegressionDiscretizer.h"
#include "BranchAssignmentObjectives/BranchAssignmentFactory.h"
#include "algorithms/BinPartitionAssignments.h"
#include "algorithms/CoordinateDescent.h"
#include "Criterion.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <numeric>
#include <set>
#include <cmath>

using Catch::Matchers::WithinAbs;

namespace {
class Bins : public InnerDiscretizer<std::vector<double>>,
             public UnivariateThresholds {
public:
  std::vector<size_t> root;
  std::vector<double> cuts;
  bool missing;
  Bins(const std::vector<std::vector<double>> &histograms,
       std::vector<size_t> root, bool missing = false, bool numeric = false,
       size_t samplesPerBin = 0)
      : root(std::move(root)), missing(missing) {
    for (const auto &histogram : histograms) {
      leafStats_.push_back({histogram});
      const double weight = std::accumulate(histogram.begin(), histogram.end(), 0.0);
      leafNodeWeights_.push_back(weight);
      leafNumSamples_.push_back(samplesPerBin ? samplesPerBin : static_cast<size_t>(weight));
    }
    numLeaves_ = histograms.size() - missing;
    if (numeric) cuts.assign(numLeaves_, 0.0);
    markTrained();
  }
  const std::vector<double> &thresholds() const override { return cuts; }
  std::vector<size_t> rootBinAssignments(size_t missingBranch) const override {
    auto labels = root;
    if (missing && !labels.empty()) labels.back() = missingBranch;
    return labels;
  }
  void transform(const arma::fmat &, arma::Row<size_t> &) const override {}
  size_t routeToBin(const std::vector<float> &) const override { return 0; }
};

double impurity(Bins &bins, std::vector<size_t> labels, size_t k,
                LearningCriterion criterion) {
  return makeBranchAssignment(criterion, labels, k, bins.leafStats(),
      bins.leafNodeWeights(), bins.leafNumSamples(),
      {bins.leafStats()[0][0].size()})->objective();
}

ShapeBranchAssignmentSearchResult search(Bins &bins, size_t k, size_t minLeaf,
    double penalty, size_t iters, LearningCriterion criterion = LearningCriterion::Gini) {
  TreeBuildingParams outer;
  outer.minLeafSize = minLeaf;
  outer.branchingPenalty = penalty;
  CoordinateDescentParams cd;
  cd.maxIters = iters;
  std::mt19937_64 rng(42);
  const double parent = impurity(bins, std::vector<size_t>(bins.leafStats().size()), 1, criterion);
  return searchShapeBranchAssignmentFromDiscretizer(bins, criterion, parent, k,
      outer, cd, rng, {bins.leafStats()[0][0].size()}, 1,
      nullptr, nullptr, 0, bins.missing).best;
}
} // namespace

TEST_CASE("Regularized acceptance is strictly above fixed double epsilon", "[shape_search][outer_score]") {
  Bins bins({{1, 0}, {0, 1}}, {0, 1});
  TreeBuildingParams outer;
  CoordinateDescentParams cd;
  cd.maxIters = 0;
  const double eps = std::numeric_limits<double>::epsilon();
  for (double alpha : {1.0, 1.0 + eps, std::nextafter(1.0, 0.0), 1.0 - eps,
                       std::nextafter(1.0 - eps, 0.0)}) {
    outer.minGainSplit = alpha;
    std::mt19937_64 rng(42);
    const auto result = searchShapeBranchAssignmentFromDiscretizer(
        bins, LearningCriterion::Gini, 0.5, 2, outer, cd, rng, {2}, 1,
        nullptr, nullptr, 0, false).best;
    REQUIRE(result.found == (alpha < 1.0 - eps));
  }
  outer.minGainSplit = 0.0;
  for (double parent : {std::numeric_limits<double>::quiet_NaN(),
                        std::numeric_limits<double>::infinity(),
                        -std::numeric_limits<double>::infinity()}) {
    std::mt19937_64 rng(42);
    REQUIRE_FALSE(searchShapeBranchAssignmentFromDiscretizer(
        bins, LearningCriterion::Gini, parent, 2, outer, cd, rng, {2}, 1,
        nullptr, nullptr, 0, false).best.found);
  }
}

TEST_CASE("Branch costs charge only children beyond binary and do not reduce raw importance", "[shape_search][outer_score]") {
  Bins bins({{10, 0, 0}, {0, 10, 0}, {0, 0, 10}}, {0, 1, 1});
  TreeBuildingParams outer;
  outer.minGainSplit = 2.0;
  outer.branchingPenalty = 3.0;
  CoordinateDescentParams cd;
  cd.maxIters = 0;
  std::mt19937_64 rng(42);
  auto result = searchShapeBranchAssignmentFromDiscretizer(
      bins, LearningCriterion::Gini, 2.0 / 3.0, 3, outer, cd, rng, {3}, 1,
      nullptr, nullptr, 0, false).best;
  REQUIRE(result.chosenK == 3);
  REQUIRE_THAT(result.impurityDecrease, WithinAbs(20.0, 1e-12));
  REQUIRE_THAT(result.regularizedGain, WithinAbs(15.0, 1e-12));
  outer.branchingPenalty = 11.0;
  result = searchShapeBranchAssignmentFromDiscretizer(
      bins, LearningCriterion::Gini, 2.0 / 3.0, 3, outer, cd, rng, {3}, 1,
      nullptr, nullptr, 0, false).best;
  REQUIRE(result.chosenK == 2);
  REQUIRE_THAT(result.impurityDecrease, WithinAbs(10.0, 1e-12));
  REQUIRE_THAT(result.regularizedGain, WithinAbs(8.0, 1e-12));
  outer.branchingPenalty = 10.0;
  result = searchShapeBranchAssignmentFromDiscretizer(
      bins, LearningCriterion::Gini, 2.0 / 3.0, 3, outer, cd, rng, {3}, 1,
      nullptr, nullptr, 0, false).best;
  REQUIRE(result.chosenK == 2); // Equal score conserves children.
}

TEST_CASE("Search retains independently replayable candidates at each occupied arity", "[shape_search]") {
  Bins bins({{10, 0, 0}, {0, 10, 0}, {0, 0, 10}, {0, 0, 0}},
            {0, 1, 1, 0}, true);
  TreeBuildingParams outer;
  CoordinateDescentParams cd;
  std::mt19937_64 rng(42);
  const auto result = searchShapeBranchAssignmentFromDiscretizer(
      bins, LearningCriterion::Gini, 2.0 / 3.0, 4, outer, cd, rng, {3}, 1,
      nullptr, nullptr, 0, true);
  REQUIRE(result.best.chosenK == 3);
  REQUIRE_FALSE(result.byArity[4].found); // An empty missing bin is not a child.
  for (size_t k : {2, 3}) {
    const auto &candidate = result.byArity[k];
    REQUIRE(candidate.found);
    REQUIRE(candidate.chosenK == k);
    REQUIRE(candidate.partitionSampleCounts.size() == k);
    REQUIRE(candidate.partitionWeights.size() == k);
    REQUIRE(candidate.partitionClassCounts.size() == k);
    REQUIRE(*std::max_element(candidate.assignments.begin(), candidate.assignments.end()) < k);
    REQUIRE_THAT(30 * (2.0 / 3.0 - impurity(bins, candidate.assignments, k,
        LearningCriterion::Gini)), WithinAbs(candidate.impurityDecrease, 1e-12));
  }
}

TEST_CASE("Raw screening proxy ignores costs and can differ from growth winner", "[shape_search]") {
  Bins bins({{10, 0, 0}, {0, 10, 0}, {0, 0, 10}}, {0, 1, 1});
  TreeBuildingParams outer;
  outer.branchingPenalty = 11.0;
  CoordinateDescentParams cd;
  cd.maxIters = 0;
  for (double alpha : {0.0, 100.0}) {
    outer.minGainSplit = alpha;
    std::mt19937_64 rng(42);
    const auto result = searchShapeBranchAssignmentFromDiscretizer(
        bins, LearningCriterion::Gini, 2.0 / 3.0, 3, outer, cd, rng, {3}, 1,
        nullptr, nullptr, 0, false);
    REQUIRE(result.rawBest.found);
    REQUIRE(result.rawBest.chosenK == 3);
    REQUIRE_THAT(result.rawBest.childImpurity, WithinAbs(0.0, 1e-12));
    REQUIRE_THAT(result.rawBest.impurityDecrease, WithinAbs(20.0, 1e-12));
    if (alpha == 0.0) {
      REQUIRE(result.best.chosenK == 2);
      REQUIRE_THAT(result.best.impurityDecrease, WithinAbs(10.0, 1e-12));
    } else {
      REQUIRE_FALSE(result.best.found);
    }
  }
}

TEST_CASE("Compacted raw proxies must still have finite positive final gain", "[shape_search]") {
  std::mt19937_64 rng(0);
  std::vector<std::vector<double>> histograms;
  std::vector<std::vector<double>> parentCounts(1, std::vector<double>(2));
  for (size_t i = 0; i < 6; ++i) {
    const double weight = 1 + (rng() % 10000) / 10.0;
    histograms.push_back({weight, 2 * weight});
    parentCounts[0][0] += weight;
    parentCounts[0][1] += 2 * weight;
  }
  Bins bins(histograms, {0, 1, 0, 1, 0, 1}, false, false, 3);
  const auto result = searchShapeBranchAssignmentFromDiscretizer(
      bins, LearningCriterion::Gini, Criterion::gini(parentCounts), 3,
      TreeBuildingParams{}, CoordinateDescentParams{}, rng, {2}, 1,
      nullptr, nullptr, 0, false);
  // Identical class proportions expose roundoff between trial scoring and the
  // compacted snapshot. Eligibility must use the final score, without tolerance.
  const auto hasValidGain = [](const auto &candidate) {
    return !candidate.found || (std::isfinite(candidate.impurityDecrease) &&
        candidate.impurityDecrease > std::numeric_limits<double>::epsilon());
  };
  REQUIRE(std::all_of(result.byArity.begin(), result.byArity.end(), hasValidGain));
  REQUIRE(hasValidGain(result.rawBest));
}

TEST_CASE("Weighted k-means preserves fractional and tiny masses and ignores empty bins", "[shape_search]") {
  for (double scale : {1.0, 1e-16, 1e8}) {
    std::vector<std::vector<double>> hist = {{.1, 0}, {.72, .18}, {.55, .45}, {0, 1}};
    std::vector<double> weights = {.1, .9, 1, 1};
    for (auto &row : hist) for (auto &mass : row) mass *= scale;
    for (auto &weight : weights) weight *= scale;
    std::vector<size_t> labels;
    std::mt19937_64 rng(0);
    algorithms::seedBinAssignmentsKMeans(2, 4, 2, hist, {1, 1, 1, 1}, weights, rng, labels);
    REQUIRE(labels == std::vector<size_t>{0, 0, 0, 1});
    hist.insert(hist.begin(), {0, 0});
    weights.insert(weights.begin(), 0);
    rng.seed(0);
    algorithms::seedBinAssignmentsKMeans(2, 5, 2, hist, {0, 1, 1, 1, 1}, weights, rng, labels);
    REQUIRE(labels == std::vector<size_t>{0, 0, 0, 0, 1});
  }
  arma::mat points = {{0, 1}, {1, 0}, {.5, .5}};
  arma::vec weights = {0, 0, 0};
  std::vector<size_t> labels;
  std::mt19937_64 rng(0);
  algorithms::initAssignmentsWeightedKMeans(points, weights, 3, rng, labels);
  REQUIRE(labels == std::vector<size_t>{0, 0, 0});
}

TEST_CASE("Coordinate descent observes rejected trials as well as initial and final states", "[shape_search]") {
  Bins bins({{10, 0}, {0, 10}}, {0, 1});
  auto labels = bins.root;
  auto objective = makeBranchAssignment(LearningCriterion::Gini, labels, 2,
      bins.leafStats(), bins.leafNodeWeights(), bins.leafNumSamples(), {2});
  std::set<std::vector<size_t>> seen;
  std::mt19937_64 rng(42);
  coordinateDescent(2, *objective, rng, 1, 1, false,
      [&](BranchAssignment &state) { seen.insert(state.assignments); });
  REQUIRE(seen == std::set<std::vector<size_t>>{{0, 1}, {0, 0}, {1, 1}});
  REQUIRE(labels == bins.root);
}

TEST_CASE("Selection uses occupied-branch penalty and preserves the binary root bound", "[shape_search]") {
  for (auto criterion : {LearningCriterion::Gini, LearningCriterion::Entropy}) {
    Bins bins({{10, 0, 0}, {0, 10, 0}, {0, 0, 10}, {0, 0, 0}}, {0, 1, 1, 0}, true);
    const auto rootImpurity = impurity(bins, bins.root, 2, criterion);
    // Penalties now use total mass (30 here), rather than normalized impurity.
    const auto binary = search(bins, 4, 5, 60.0, 5, criterion);
    REQUIRE(binary.found);
    REQUIRE(binary.chosenK == 2);
    REQUIRE(binary.rootFeasible);
    REQUIRE(impurity(bins, binary.assignments, binary.chosenK, criterion) <= rootImpurity + 1e-12);
    const auto multiway = search(bins, 4, 5, 0.0, 5, criterion);
    REQUIRE(multiway.chosenK == 3);
    REQUIRE_THAT(impurity(bins, multiway.assignments, 3, criterion), WithinAbs(0.0, 1e-12));
    REQUIRE(binary.partitionSampleCounts.size() == binary.chosenK);
    REQUIRE(binary.partitionClassCounts.size() == binary.chosenK);
    REQUIRE(binary.partitionWeights.size() == binary.chosenK);
    REQUIRE(binary.assignments.back() < binary.chosenK);
  }
}

TEST_CASE("Infeasible numeric root falls back to a feasible bin boundary", "[shape_search]") {
  for (auto criterion : {LearningCriterion::Gini, LearningCriterion::Entropy}) {
    Bins bins({{0, 1}, {4, 0}, {3, 2}, {0, 0}}, {0, 1, 1, 0}, true, true);
    const auto result = search(bins, 3, 5, 0.0, 0, criterion);
    REQUIRE_FALSE(result.rootFeasible);
    REQUIRE(result.found);
    REQUIRE(result.partitionSampleCounts == std::vector<size_t>{5, 5});
    const double fallback = impurity(bins, {0, 0, 1, 0}, 2, criterion);
    REQUIRE(impurity(bins, result.assignments, 2, criterion) <= fallback + 1e-12);
    REQUIRE_FALSE(search(bins, 3, 6, 0.0, 5, criterion).found);
  }
}

TEST_CASE("Missing-bin snapshots consider feasible placements without changing the live search", "[shape_search]") {
  Bins bins({{4, 0}, {0, 6}, {0, 2}}, {0, 1, 0}, true);
  // The lowest-impurity missing placement makes the first branch too small.
  REQUIRE(impurity(bins, {0, 1, 1}, 2, LearningCriterion::Gini) == 0.0);
  const auto result = search(bins, 3, 5, 0.0, 5);
  REQUIRE(result.found);
  REQUIRE(result.partitionSampleCounts == std::vector<size_t>{6, 6});
  REQUIRE(result.assignments[0] == result.assignments[2]);
  REQUIRE(result.assignments[0] != result.assignments[1]);
}

TEST_CASE("A rejected coordinate trial can be the only feasible split", "[shape_search]") {
  // Root and k-means isolate one pure minority sample: impurity zero, but
  // infeasible. Moving either majority bin into it is feasible and is rejected
  // by the live impurity optimizer. It must nevertheless survive selection.
  Bins bins({{0, 1}, {5, 0}, {5, 0}}, {0, 1, 1});
  REQUIRE_FALSE(search(bins, 2, 5, 0.0, 0).found);
  const auto result = search(bins, 2, 5, 0.0, 1);
  REQUIRE(result.found);
  REQUIRE_FALSE(result.rootFeasible);
  auto counts = result.partitionSampleCounts;
  std::sort(counts.begin(), counts.end());
  REQUIRE(counts == std::vector<size_t>{5, 6});
  REQUIRE_THAT(impurity(bins, result.assignments, 2, LearningCriterion::Gini),
               WithinAbs(5.0 / 33.0, 1e-12));
}

TEST_CASE("Zero-weight samples still occupy a utilized branch", "[shape_search]") {
  Bins bins({{0, 0}, {0, 5}, {5, 0}}, {0, 1, 0});
  bins.leafNumSamples()[0] = 5;
  const auto result = search(bins, 3, 5, 0.0, 5);
  REQUIRE(result.found);
  REQUIRE(std::accumulate(result.partitionSampleCounts.begin(), result.partitionSampleCounts.end(), size_t{0}) == 15);
  for (size_t count : result.partitionSampleCounts) REQUIRE(count >= 5);
}

TEST_CASE("Regression retains rejected feasible trials and compacts occupied metadata", "[shape_search]") {
  Bins bins({{0, 1}, {5, 0}, {5, 0}}, {0, 1, 1});
  bins.leafStats() = {{{1, 1}}, {{0, 0}}, {{0, 0}}};
  const auto criterion = LearningCriterion::SquaredError;
  REQUIRE_FALSE(search(bins, 2, 5, 0.0, 0, criterion).found);
  const auto result = search(bins, 2, 5, 0.0, 1, criterion);
  REQUIRE(result.found);
  REQUIRE_FALSE(result.rootFeasible);
  auto counts = result.partitionSampleCounts;
  std::sort(counts.begin(), counts.end());
  REQUIRE(counts == std::vector<size_t>{5, 6});
  REQUIRE(result.partitionAggStats.size() == result.chosenK);
  REQUIRE(result.partitionWeights.size() == result.chosenK);
  REQUIRE_THAT(impurity(bins, result.assignments, 2, criterion),
               WithinAbs(5.0 / 66.0, 1e-12));

  bins.leafStats() = {{{0, 0}}, {{5, 5}}, {{10, 20}}};
  bins.leafNodeWeights() = {5, 5, 5};
  bins.leafNumSamples() = {5, 5, 5};
  const auto binary = search(bins, 3, 5, 30.0, 5, criterion);
  REQUIRE(binary.chosenK == 2);
  REQUIRE(binary.partitionAggStats.size() == 2);
  REQUIRE(binary.partitionSampleCounts.size() == 2);
  REQUIRE(impurity(bins, binary.assignments, 2, criterion) <=
          impurity(bins, bins.root, 2, criterion) + 1e-12);
}

TEST_CASE("Pair root membership merges the entire missing subtree into either branch", "[shape_search]") {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  arma::fmat X = {{-2, -2, 2, 2, nan, nan}, {-2, 2, -2, 2, -2, 2}};
  arma::Mat<size_t> y(1, 6);
  y.row(0) = arma::Row<size_t>{0, 0, 1, 1, 0, 1};
  FeatureInfo first, second;
  first.type = second.type = FeatureType::Continuous;
  first.indices = arma::uvec{0};
  second.indices = arma::uvec{1};
  arma::uvec features = {0, 1};
  PairClassificationDiscretizer disc(LearningCriterion::Gini, first, second);
  disc.Train(X, features, y, {2}, 1, 0.0, 2, 0);
  REQUIRE(disc.routingTree()[0].rawFeature == 0);
  REQUIRE_FALSE(disc.routingTree()[disc.routingTree()[0].missing].isLeaf);
  for (size_t missingBranch : {0, 1}) {
    const auto root = disc.rootBinAssignments(missingBranch);
    for (size_t bin = 0; bin < root.size(); ++bin)
      for (size_t sample : disc.inSampleDiscretizations()[bin])
        REQUIRE(root[bin] == (sample >= 4 ? missingBranch : (sample >= 2 ? 1 : 0)));
  }
  for (auto criterion : {LearningCriterion::SquaredError, LearningCriterion::AbsoluteError}) {
    PairRegressionDiscretizer regression(criterion, first, second);
    const arma::fmat targets = arma::conv_to<arma::fmat>::from(y);
    regression.Train(X, features, targets, 1, 0.0, 2, 0);
    REQUIRE(regression.routingTree()[0].rawFeature == 0);
    REQUIRE_FALSE(regression.routingTree()[regression.routingTree()[0].missing].isLeaf);
    for (size_t missingBranch : {0, 1}) {
      const auto root = regression.rootBinAssignments(missingBranch);
      REQUIRE(root.size() == regression.numLeaves());
      for (size_t bin = 0; bin < root.size(); ++bin)
        for (size_t sample : regression.inSampleDiscretizations()[bin])
          REQUIRE(root[bin] == (sample >= 4 ? missingBranch : (sample >= 2 ? 1 : 0)));
    }
  }
}
