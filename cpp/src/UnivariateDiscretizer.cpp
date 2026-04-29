
#include <armadillo>
#include <stdexcept>
#include <memory>
#include "frontiers.h"
#include <limits>
#include <map>
#include <algorithm>
#include <cmath>
#include <numeric>
#include "SplitCandidate.h"
#include "UnivariateDiscretizer.h"


constexpr double FEATURE_SPLIT_MIN_DIFF = 1e-7;

constexpr double eps = std::numeric_limits<double>::epsilon();

namespace {
double giniFromCounts(const std::vector<size_t> &counts, const size_t n) {
    if (n == 0) return 0.0;
    const double nDouble = static_cast<double>(n);
    const double sumP2 = std::accumulate(
        counts.begin(),
        counts.end(),
        0.0,
        [nDouble](const double acc, const size_t count) {
            const double p = static_cast<double>(count) / nDouble;
            return acc + p * p;
        }
    );
    return 1.0 - sumP2;
}

size_t argmaxIndex(const std::vector<size_t> &counts) {
    return static_cast<size_t>(std::distance(
        counts.begin(),
        std::max_element(counts.begin(), counts.end())
    ));
}
}  // namespace

UnivariateDiscretizer::~UnivariateDiscretizer() = default;

bool UnivariateDiscretizer::findBestSplit(const arma::fmat &X, SplitCandidate &split) {
    double bestProxyImprovement = -std::numeric_limits<double>::infinity();
    bool found = false;

    const size_t N = split.end - split.start + 1;
    if (N < 2 * minLeafSize)
        return false;
    arma::frowvec featureValues = X.row(feature);

    std::vector<size_t> leftStats(numClasses, 0);
    std::vector<size_t> rightStats = split.classCounts;
    for (size_t i = split.start + 1; i <= split.end; ++i) {
        const size_t movedLabel = sortedLabels(sortedOrder(i - 1));
        ++leftStats[movedLabel];
        --rightStats[movedLabel];

        const size_t Nl = i - split.start;
        const size_t Nr = split.end - i + 1;
        if (Nl < minLeafSize) continue;
        if (Nr < minLeafSize) break;

        const float currValue = static_cast<float>(featureValues(sortedOrder(i)));
        const float prevValue = static_cast<float>(featureValues(sortedOrder(i - 1)));
        if (currValue <= prevValue + static_cast<float>(FEATURE_SPLIT_MIN_DIFF))
            continue;
    
        
        const double leftScore = giniFromCounts(leftStats, static_cast<size_t>(Nl));
        const double rightScore = giniFromCounts(rightStats, static_cast<size_t>(Nr));

        const double proxyImprovement =
            -static_cast<double>(Nr) * rightScore - static_cast<double>(Nl) * leftScore;
        if (proxyImprovement > bestProxyImprovement) {
            found = true;
            bestProxyImprovement = proxyImprovement;
            const float leftValue = static_cast<float>(featureValues(sortedOrder(i - 1)));
            const float rightValue = static_cast<float>(featureValues(sortedOrder(i)));
            
            // Mirror sklearn threshold selection to avoid midpoint rounding onto rightValue.
            double threshold = static_cast<double>(leftValue) / 2.0 + static_cast<double>(rightValue) / 2.0;
            if (threshold == static_cast<double>(rightValue) ||
                !std::isfinite(threshold)) {
                threshold = static_cast<double>(leftValue);
            }
            split.threshold = threshold;

            split.leftStart = split.start;
            split.leftEnd = i - 1;
            split.leftScore = leftScore;
            split.leftPrediction = argmaxIndex(leftStats);
            split.leftClassCounts = leftStats;

            split.rightStart = i;
            split.rightEnd = split.end;
            split.rightScore = rightScore;
            split.rightPrediction = argmaxIndex(rightStats);
            split.rightClassCounts = rightStats;

            const double gain = split.score - (
                static_cast<double>(Nl) / static_cast<double>(N) * leftScore +
                static_cast<double>(Nr) / static_cast<double>(N) * rightScore
            );
            split.informationGain = static_cast<double>(N) / static_cast<double>(totalSamples) * gain;
        }
    }
    return found;
}

void UnivariateDiscretizer::finalizeTraining(std::map<std::tuple<size_t, size_t>, SplitCandidate> &leaves) {
    inSampleDiscretizations.clear();
    binPredictions.clear();
    std::vector<double> thresholds;

    for (auto &[_, leaf]: leaves) {
        inSampleDiscretizations.push_back(
            arma::conv_to<std::vector<size_t> >::from(sortedOrder.subvec(leaf.start, leaf.end)));
        binPredictions.push_back(leaf.prediction);
        thresholds.push_back(leaf.routingThreshold);
    }

    binMapFunction = [this, thresholds](const arma::fmat &X, arma::Row<size_t> &binLoc) {
        binLoc = arma::Row<size_t>(X.n_cols);
        arma::frowvec featureRow = X.row(feature);
        for (size_t col = 0; col < X.n_cols; ++col) {
            const auto it = std::ranges::lower_bound(thresholds, featureRow(col));
            binLoc(col) = std::distance(thresholds.begin(), it);
        }
    };
    numLeaves = leaves.size();
    sortedOrder.clear();
    sortedLabels.clear();
}

void UnivariateDiscretizer::Train(
    const arma::fmat &X,
    arma::uvec &features,
    const arma::frowvec &responses,
    size_t minLeafSize,
    double minGainSplit,
    size_t maxDepth,
    size_t maxLeafNodes
) {
    throw std::runtime_error("not implemented");
}

void UnivariateDiscretizer::Train(
    const arma::fmat &X,
    arma::uvec &features,
    const arma::Row<size_t> &labels,
    size_t numClasses,
    size_t minLeafSize,
    double minGainSplit,
    size_t maxDepth,
    size_t maxLeafNodes
) {
    if (features.n_elem != 1) throw std::invalid_argument("features must be size 1");
    if (minLeafSize == 0) throw std::invalid_argument("minLeafSize must be > 0");
    this->numClasses = numClasses;
    this->feature = features(0);
    this->minLeafSize = minLeafSize;
    this->minGainSplit = minGainSplit;
    this->maxDepth = maxDepth;
    this->maxLeafNodes = maxLeafNodes;

    const size_t N_total = X.n_cols;
    totalSamples = N_total;

    sortedOrder = arma::sort_index(X.row(feature));
    sortedLabels = labels;

    std::vector<size_t> rootClassCounts(numClasses, 0);
    for (size_t idx = 0; idx < N_total; ++idx)
        ++rootClassCounts[sortedLabels(sortedOrder(idx))];

    const auto frontier = [&]() -> std::unique_ptr<frontiers::IFrontier<SplitCandidate> > {
        if (maxLeafNodes == 0) return std::make_unique<frontiers::Stack<SplitCandidate> >();
        return std::make_unique<frontiers::MinHeap<SplitCandidate> >();
    }();

    std::map<std::tuple<size_t, size_t>, SplitCandidate> leaves;

    SplitCandidate rootSplit = {
        .height = 0,
        .start = 0,
        .end = N_total - 1,
        .score = giniFromCounts(rootClassCounts, N_total),
        .prediction = argmaxIndex(rootClassCounts),
        .routingThreshold = std::numeric_limits<double>::infinity(),
        .classCounts = rootClassCounts
    };


    if (rootSplit.score > eps && findBestSplit(X, rootSplit))
        frontier->push(rootSplit);

    leaves[std::make_tuple(rootSplit.start, rootSplit.end)] = rootSplit;

    
    while (frontier->size() > 0 && (maxLeafNodes == 0 || leaves.size() < maxLeafNodes)) {
        SplitCandidate split = frontier->peek();
        frontier->pop();

        if (split.score <= eps ||
            split.informationGain + eps < minGainSplit ||
            (maxDepth != 0 && split.height >= maxDepth))
            continue;

        SplitCandidate left = {
            .height = split.height + 1,
            .start = split.leftStart,
            .end = split.leftEnd,
            .score = split.leftScore,
            .prediction = split.leftPrediction,
            .routingThreshold = split.threshold,
            .classCounts = split.leftClassCounts
        };

        SplitCandidate right = {
            .height = split.height + 1,
            .start = split.rightStart,
            .end = split.rightEnd,
            .score = split.rightScore,
            .prediction = split.rightPrediction,
            .routingThreshold = split.routingThreshold,
            .classCounts = split.rightClassCounts
        };


        if (right.score > eps && findBestSplit(X, right)) frontier->push(right);
        if (left.score > eps && findBestSplit(X, left)) frontier->push(left);
        

        leaves.erase(std::make_tuple(split.start, split.end));
        leaves[std::make_tuple(left.start, left.end)] = left;
        leaves[std::make_tuple(right.start, right.end)] = right;
    }

    finalizeTraining(leaves);
}


void UnivariateDiscretizer::transform(const arma::fmat &X, arma::Row<size_t> &binLoc) {
    if (binMapFunction == nullptr)
        throw std::runtime_error("Cannot transform values without first training the discretizer");

    binMapFunction(X, binLoc);
}


const std::vector<std::vector<size_t> > &UnivariateDiscretizer::getInSampleDiscretizations() {
    if (binMapFunction == nullptr)
        throw std::runtime_error("Cannot transform values without first training the discretizer");

    return inSampleDiscretizations;
}

const std::vector<size_t> &UnivariateDiscretizer::getBinPredictions() {
    if (binMapFunction == nullptr)
        throw std::runtime_error("Cannot transform values without first training the discretizer");

    return binPredictions;
}
