#include "Discretizers/univariate/NumericFallbackDiscretizer.h"

#include "BranchAssignmentObjectives/BranchAssignmentFactory.h"
#include "Discretizers/univariate/UnivariateDiscretizer.h"
#include "algorithms/missing_values.h"

#include <limits>
#include <type_traits>

namespace {

class NumericFallbackDiscretizer final
    : public UnivariateDiscretizer<std::vector<double>> {
public:
  template <typename Target>
  NumericFallbackDiscretizer(const arma::fmat &X, size_t feature,
                            const arma::Mat<Target> &y,
                            const arma::Row<float> &weights,
                            const std::vector<size_t> &classesPerOutput) {
    this->feature = feature;
    const auto sorted = missing_values::sort_index_finite_first(X.row(feature));
    const size_t finiteCount = sorted.first_non_finite_index;
    inSampleDiscretizations_.emplace_back();
    for (size_t i = 0; i < finiteCount; ++i) {
      const size_t sample = sorted.order(i);
      if (i > 0) {
        const float previous = X(feature, sorted.order(i - 1));
        const float current = X(feature, sample);
        // Match the inner splitter's adjacent-value tolerance and midpoint.
        if (current > previous + static_cast<float>(1e-7)) {
          double threshold = static_cast<double>(previous) / 2.0 +
                             static_cast<double>(current) / 2.0;
          if (threshold == static_cast<double>(current) || !std::isfinite(threshold))
            threshold = previous;
          thresholds_.push_back(threshold);
          inSampleDiscretizations_.emplace_back();
        }
      }
      inSampleDiscretizations_.back().push_back(sample);
    }
    thresholds_.push_back(std::numeric_limits<double>::infinity());
    numLeaves_ = inSampleDiscretizations_.size();
    inSampleDiscretizations_.emplace_back();
    for (size_t i = finiteCount; i < sorted.order.n_elem; ++i)
      inSampleDiscretizations_.back().push_back(sorted.order(i));

    leafStats_.resize(numLeaves_ + 1);
    leafNodeWeights_.assign(numLeaves_ + 1, 0.0);
    leafNumSamples_.resize(numLeaves_ + 1);
    for (size_t b = 0; b <= numLeaves_; ++b) {
      leafStats_[b].resize(y.n_rows);
      for (size_t o = 0; o < y.n_rows; ++o)
        leafStats_[b][o].assign(
            classesPerOutput.empty() ? 2 : classesPerOutput[o], 0.0);
      leafNumSamples_[b] = inSampleDiscretizations_[b].size();
      for (size_t sample : inSampleDiscretizations_[b]) {
        const double w = weights(sample);
        leafNodeWeights_[b] += w;
        for (size_t o = 0; o < y.n_rows; ++o) {
          if constexpr (std::is_same_v<Target, size_t>) {
            leafStats_[b][o][y(o, sample)] += w;
          } else {
            const double value = y(o, sample);
            leafStats_[b][o][0] += w * value;
            leafStats_[b][o][1] += w * value * value;
          }
        }
      }
    }
    markTrained();
  }

  void keepThreshold(size_t cut) {
    const size_t finiteBins = numLeaves_;
    const bool numericSplit = cut > 0 && cut < finiteBins;
    const double threshold =
        numericSplit ? thresholds_[cut - 1] : thresholds_.back();
    auto samples = std::move(inSampleDiscretizations_);
    auto stats = std::move(leafStats_);
    auto weights = std::move(leafNodeWeights_);
    numLeaves_ = numericSplit ? 2 : 1;
    thresholds_ = numericSplit ? std::vector<double>{threshold, thresholds_.back()}
                              : std::vector<double>{threshold};
    inSampleDiscretizations_.assign(numLeaves_ + 1, {});
    leafStats_.assign(numLeaves_ + 1, stats.front());
    for (auto &bin : leafStats_)
      for (auto &output : bin)
        std::fill(output.begin(), output.end(), 0.0);
    leafNodeWeights_.assign(numLeaves_ + 1, 0.0);
    leafNumSamples_.assign(numLeaves_ + 1, 0);
    for (size_t b = 0; b <= finiteBins; ++b) {
      const size_t destination =
          b == finiteBins ? numLeaves_ : (numericSplit && b >= cut);
      auto &rows = inSampleDiscretizations_[destination];
      rows.insert(rows.end(), samples[b].begin(), samples[b].end());
      leafNumSamples_[destination] += samples[b].size();
      leafNodeWeights_[destination] += weights[b];
      for (size_t o = 0; o < stats[b].size(); ++o)
        for (size_t j = 0; j < stats[b][o].size(); ++j)
          leafStats_[destination][o][j] += stats[b][o][j];
    }
  }

  std::vector<size_t> rootBinAssignments(size_t missingBranch = 0) const override {
    return numLeaves_ == 2 ? std::vector<size_t>{0, 1, missingBranch}
                           : std::vector<size_t>{0, missingBranch};
  }
};

template <typename Target>
std::shared_ptr<InnerDiscretizer<std::vector<double>>> makeFallback(
    LearningCriterion criterion, const arma::fmat &X, size_t feature,
    const arma::Mat<Target> &y, const arma::Row<float> &weights,
    size_t minLeafSize, const std::vector<size_t> &classesPerOutput) {
  auto disc = std::make_shared<NumericFallbackDiscretizer>(
      X, feature, y, weights, classesPerOutput);
  const size_t finiteBins = disc->numLeaves();
  if (criterion != LearningCriterion::AbsoluteError) {
    const auto &stats = disc->leafStats();
    const auto &weightsByBin = disc->leafNodeWeights();
    const auto &counts = disc->leafNumSamples();
    auto leftStats = stats.front();
    for (auto &output : leftStats)
      std::fill(output.begin(), output.end(), 0.0);
    const auto addStats = [](auto &destination, const auto &source) {
      for (size_t o = 0; o < source.size(); ++o)
        for (size_t j = 0; j < source[o].size(); ++j)
          destination[o][j] += source[o][j];
    };
    // Accumulate each side independently: total-minus-prefix can erase small
    // remaining weights and moments when one bin dominates the total.
    std::vector suffixStats(finiteBins + 1, leftStats);
    std::vector<double> suffixWeights(finiteBins + 1, 0.0);
    std::vector<size_t> suffixCounts(finiteBins + 1, 0);
    for (size_t b = finiteBins; b > 0; --b) {
      suffixStats[b - 1] = suffixStats[b];
      addStats(suffixStats[b - 1], stats[b - 1]);
      suffixWeights[b - 1] = suffixWeights[b] + weightsByBin[b - 1];
      suffixCounts[b - 1] = suffixCounts[b] + counts[b - 1];
    }
    double leftWeight = 0.0;
    size_t leftCount = 0;
    size_t bestCut = 0;
    double bestImpurity = std::numeric_limits<double>::infinity();
    for (size_t cut = 0; cut <= finiteBins; ++cut) {
      std::vector compactStats{leftStats, suffixStats[cut], stats.back()};
      std::vector<double> compactWeights{
          leftWeight, suffixWeights[cut], weightsByBin.back()};
      std::vector<size_t> compactCounts{
          leftCount, suffixCounts[cut], counts.back()};
      for (size_t missingBranch = 0; missingBranch < 2; ++missingBranch) {
        std::vector<size_t> labels{0, 1, missingBranch};
        auto objective = makeBranchAssignment(
            criterion, labels, 2, compactStats, compactWeights, compactCounts,
            classesPerOutput, y.n_rows);
        if (!objective->partitionCountsMeetMinLeaf(minLeafSize))
          continue;
        const double impurity = objective->objective();
        if (std::isfinite(impurity) && impurity < bestImpurity) {
          bestImpurity = impurity;
          bestCut = cut;
        }
      }
      if (cut < finiteBins) {
        addStats(leftStats, stats[cut]);
        leftWeight += weightsByBin[cut];
        leftCount += counts[cut];
      }
    }
    disc->keepThreshold(bestCut);
    return disc;
  }
  std::vector<std::vector<double>> unusedStats;
  size_t bestCut = 0;
  double bestImpurity = std::numeric_limits<double>::infinity();
  // Include finite-vs-missing alone; then test every legal finite threshold.
  for (size_t cut = 0; cut <= finiteBins; ++cut) {
    std::vector maeYs(3, std::vector<std::vector<float>>(y.n_rows));
    std::vector<std::vector<float>> maeWeights(3);
    std::vector<double> compactWeights(3, 0.0);
    std::vector<size_t> compactCounts(3, 0);
    for (size_t b = 0; b <= finiteBins; ++b) {
      const size_t destination = b == finiteBins ? 2 : (b >= cut);
      for (size_t sample : disc->inSampleDiscretizations()[b]) {
        maeWeights[destination].push_back(weights(sample));
        compactWeights[destination] += weights(sample);
        ++compactCounts[destination];
        for (size_t o = 0; o < y.n_rows; ++o)
          maeYs[destination][o].push_back(y(o, sample));
      }
    }
    std::vector<size_t> labels{0, 1, 0};
    for (size_t missingBranch = 0; missingBranch < 2; ++missingBranch) {
      labels.back() = missingBranch;
      // Rebuild from bins: subtracting a dominant bin can erase the remaining
      // partition mass even though its sorted target samples remain intact.
      // ponytail: sorting three compact bins per trial costs O(n² log n);
      // use range-median queries if missing-data MAE profiling warrants it.
      auto objective = makeBranchAssignment(
          criterion, labels, 2, unusedStats,
          compactWeights, compactCounts, &maeYs, &maeWeights);
      if (!objective->partitionCountsMeetMinLeaf(minLeafSize))
        continue;
      const double impurity = objective->objective();
      if (std::isfinite(impurity) && impurity < bestImpurity) {
        bestImpurity = impurity;
        bestCut = cut;
      }
    }
  }
  // Re-score the compact router through ordinary search/retention, so penalties,
  // epsilon, missing assignments, and pair proxies share the existing contract.
  disc->keepThreshold(bestCut);
  return disc;
}

} // namespace

std::shared_ptr<InnerDiscretizer<std::vector<double>>>
makeNumericFallbackDiscretizer(
    LearningCriterion criterion, const arma::fmat &X, size_t feature,
    const arma::Mat<size_t> &y, const arma::Row<float> &weights,
    size_t minLeafSize, const std::vector<size_t> &classesPerOutput) {
  return makeFallback(criterion, X, feature, y, weights, minLeafSize,
                      classesPerOutput);
}

std::shared_ptr<InnerDiscretizer<std::vector<double>>>
makeNumericFallbackDiscretizer(
    LearningCriterion criterion, const arma::fmat &X, size_t feature,
    const arma::Mat<float> &y, const arma::Row<float> &weights, size_t minLeafSize) {
  return makeFallback(criterion, X, feature, y, weights, minLeafSize, {});
}
