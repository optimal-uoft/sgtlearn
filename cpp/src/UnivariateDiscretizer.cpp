#include "Discretizers.h"
#include <armadillo>
#include <stdexcept>
#include <memory>
#include "frontiers.h"
#include <limits>
#include <map>
#include <algorithm>
#include <cmath>
#include <numeric>

constexpr float FEATURE_SPLIT_MIN_DIFF = 1e-7f;

constexpr double eps = std::numeric_limits<double>::epsilon();
IDiscretizer::~IDiscretizer() = default;

UnivariateDiscretizer::~UnivariateDiscretizer() = default;

bool UnivariateDiscretizer::findBestSplit(SplitCandidate &split) {
    double bestInformationGain = -std::numeric_limits<double>::infinity();
    bool found = false;

    const int N = split.end - split.start + 1;
    if (N < 2 * minLeafSize)
        return false;
    arma::frowvec featureValues = X.row(feature);

    for (int i = split.start + minLeafSize; i <= split.end; i++) {
        int Nl = i - split.start, Nr = split.end - i + 1;
        if (Nl < minLeafSize) continue;
        if (Nr < minLeafSize) break;

        const float currValue = static_cast<float>(featureValues(sortedOrder(i)));
        const float prevValue = static_cast<float>(featureValues(sortedOrder(i - 1)));
        if (currValue <= prevValue + static_cast<float>(FEATURE_SPLIT_MIN_DIFF))
            continue;
    
        
        arma::Row leftStats = prefix.row(i - 1);
        if (split.start > 0)
            leftStats -= prefix.row(split.start - 1);

        const double leftScore = 1.0 - std::accumulate(
                                     leftStats.begin(),
                                     leftStats.end(),
                                     0.0,
                                     [Nl](const double acc, const size_t count) {
                                         const double p = static_cast<double>(count) / static_cast<double>(Nl);
                                         return acc + p * p;
                                     }
                                 );

        arma::Row rightStats = prefix.row(split.end) - prefix.row(i - 1);
        const double rightScore = 1.0 - std::accumulate(
                                      rightStats.begin(),
                                      rightStats.end(),
                                      0.0,
                                      [Nr](const double acc, const size_t count) {
                                          const double p = static_cast<double>(count) / static_cast<double>(Nr);
                                          return acc + p * p;
                                      }
                                  );

        if (const double gain = split.score - (
                static_cast<double>(Nl) / static_cast<double>(N) * leftScore +
                static_cast<double>(Nr) / static_cast<double>(N) * rightScore
                                );
            static_cast<double>(N) / static_cast<double>(totalSamples) * gain > bestInformationGain) {
            found = true;
            const double weightedGain = static_cast<double>(N) / static_cast<double>(totalSamples) * gain;
            bestInformationGain = weightedGain;
            split.informationGain = weightedGain;
            split.threshold = (featureValues(sortedOrder(i)) + featureValues(sortedOrder(i - 1))) / 2.0;

            split.leftStart = split.start;
            split.leftEnd = i - 1;
            split.leftScore = leftScore;
            split.leftPrediction = leftStats.index_max();

            split.rightStart = i;
            split.rightEnd = split.end;
            split.rightScore = rightScore;
            split.rightPrediction = rightStats.index_max();
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
    this->X = X;
    if (features.n_elem != 1) throw std::invalid_argument("features must be size 1");
    if (minLeafSize == 0) throw std::invalid_argument("minLeafSize must be > 0");
    this->feature = features(0);
    this->minLeafSize = minLeafSize;

    const size_t N_total = X.n_cols;
    totalSamples = N_total;

    sortedOrder = arma::sort_index(X.row(feature));

    arma::Mat<size_t> prefix_mat(N_total, numClasses, arma::fill::zeros);
    for (size_t c = 0; c < numClasses; ++c)
        prefix_mat.col(c) = arma::cumsum(arma::conv_to<arma::Col<size_t> >::from(labels(sortedOrder) == c));
    this->prefix = prefix_mat;

    const auto frontier = [&]() -> std::unique_ptr<frontiers::IFrontier<SplitCandidate> > {
        if (maxLeafNodes == 0) return std::make_unique<frontiers::Stack<SplitCandidate> >();
        return std::make_unique<frontiers::MinHeap<SplitCandidate> >();
    }();

    std::map<std::tuple<size_t, size_t>, SplitCandidate> leaves;

    SplitCandidate rootSplit = {
        .height = 0,
        .start = 0,
        .end = N_total - 1,
        .score = 1.0 - std::accumulate(
                     prefix.row(N_total - 1).begin(),
                     prefix.row(N_total - 1).end(),
                     0.0,
                     [N_total](const double acc, const size_t c) {
                         const double p = static_cast<double>(c) / static_cast<double>(N_total);
                         return acc + p * p;
                     }),
        .prediction = prefix.row(N_total - 1).index_max(),
        .routingThreshold = std::numeric_limits<double>::infinity()
    };


    if (rootSplit.score > eps && findBestSplit(rootSplit))
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
            .routingThreshold = split.threshold
        };

        SplitCandidate right = {
            .height = split.height + 1,
            .start = split.rightStart,
            .end = split.rightEnd,
            .score = split.rightScore,
            .prediction = split.rightPrediction,
            .routingThreshold = split.routingThreshold
        };


        if (right.score > eps && findBestSplit(right)) frontier->push(right);
        if (left.score > eps && findBestSplit(left)) frontier->push(left);
        

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
