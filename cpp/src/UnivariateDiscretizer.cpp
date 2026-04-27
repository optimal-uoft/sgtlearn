#include "Discretizers.h"
#include <armadillo>
#include <stdexcept>
#include <memory>
#include "frontiers.h"
#include <limits>
#include <map>
#include <algorithm>
#include <cmath>

const double FEATURE_SPLIT_MIN_DIFF = 1e-7;

bool UnivariateDiscretizer::findBestSplit(SplitCandidate &split) {
    double bestInformationGain = std::numeric_limits<double>::infinity();;
    bool found = false;

    int N = split.end - split.start + 1;
    for (int i = split.start; i < split.end + 1; i++) {
        int Nl = i - split.start;
        int Nr = split.end - i + 1;

        if (Nl < minLeafSize)
            continue;

        if (Nr < minLeafSize)
            break;

        if (X(sortedOrder(i), feature) - X(sortedOrder(i - 1), feature) < FEATURE_SPLIT_MIN_DIFF)
            continue;

        double leftScore = 1 - std::accumulate(
                               prefix.row(i - 1).begin(),
                               prefix.row(i - 1).end(),
                               0.0,
                               [Nl](double acc, int classCount) {
                                   return acc + std::pow(classCount / Nl, 2);
                               }
                           );
        double rightScore = 1 - std::accumulate(
                                prefix.row(i).begin(),
                                prefix.row(i).end(),
                                0.0,
                                [Nr](double acc, int classCount) {
                                    return acc + std::pow(classCount / Nr, 2);
                                }
                            );

        double informationGain = split.score - (Nl * leftScore + Nr * rightScore) / N;

        if (informationGain < bestInformationGain) {
            found = true;
            bestInformationGain = informationGain;
            split.threshold = (X(sortedOrder(i), feature) + X(sortedOrder(i - 1), feature)) / 2;

            split.leftStart = split.start;
            split.leftEnd = i - 1;

            split.rightStart = i;
            split.rightEnd = split.end;
        }
    }

    return found;
}

double UnivariateDiscretizer::Train(
    const arma::mat &X,
    arma::uvec &features,
    const arma::Row<size_t> &labels,
    size_t numClasses,
    size_t minLeafSize,
    double minGainSplit,
    size_t maxDepth,
    size_t maxLeafNodes
) {
    if (features.n_elem != 1)
        throw std::invalid_argument("features must be of size 1");
    feature = features(0);

    if (minLeafSize == 0)
        throw new std::invalid_argument("minLeafSize must be greater than 0");


    sortedOrder = arma::sort_index(X.cols(features));

    const auto frontier = [&]() -> std::unique_ptr<frontiers::IFrontier<SplitCandidate> > {
        if (maxLeafNodes == 0)
            return std::make_unique<frontiers::Stack<SplitCandidate> >();
        return std::make_unique<frontiers::MinHeap<SplitCandidate> >();
    }();

    // TODO: memory optimization, bring the total statistic sums down to each children. When you set the threshold in
    // findBestSplit at that moment you know the totals of each new leaf node which become new leaf right accumulators when their best
    // splits get computed O(d*n) -> O(d)
    // setup prefix
    arma::Mat<size_t> prefix(sortedOrder.n_elem, numClasses, arma::fill::zeros);
    for (size_t c = 0; c < numClasses; ++c)
        prefix.col(c) = arma::cumsum(arma::conv_to<arma::Col<size_t> >::from(labels(sortedOrder) == c));
    this->prefix = prefix;

    // initialize frontier
    size_t N = prefix.size();
    std::map<std::tuple<size_t, size_t>, SplitCandidate *> leaves;
    auto rootSplit = SplitCandidate{
        .height = 0,
        .score = 1 - std::accumulate(
                     prefix.row(N - 1).begin(),
                     prefix.row(N - 1).end(),
                     0.0,
                     [N](double acc, int classCount) {
                         return acc + std::pow(classCount / N, 2);
                     }
                 ),
        .start = 0,
        .end = N - 1,
        .routingThreshold = std::numeric_limits<double>::infinity()
    };

    leaves[std::make_tuple(rootSplit.start, rootSplit.end)] = &rootSplit;

    if (!findBestSplit(rootSplit)) {
        //instantiate bin of one lol
        return rootSplit.score;
    }

    frontier->push(rootSplit);

    while (frontier->size() > 0 && (maxLeafNodes != 0 || leaves.size() < maxLeafNodes)) {
        SplitCandidate split = frontier->peek();
        frontier->pop();
        // region split meets constraint requirements

        if (split.informationGain < minGainSplit) {
            if (maxLeafNodes == 0)
                continue;
            break;
        }

        if (split.height + 1 > maxDepth)
            continue;

        // endregion

        // region commit split in leaves tracking

        auto left = SplitCandidate{
            .height = split.height + 1,
            .score = split.leftScore,
            .start = split.leftStart,
            .end = split.leftEnd,
            .routingThreshold = split.threshold,
        };

        auto right = SplitCandidate{
            .height = split.height + 1,
            .score = split.rightScore,
            .start = split.rightStart,
            .end = split.rightEnd,
            .routingThreshold = split.routingThreshold
        };

        if (leaves.erase(std::make_tuple(split.start, split.end)) == 0)
            throw std::runtime_error("Could not find candidate split interval in sorted map");

        leaves[std::make_tuple(left.start, left.end)] = &left;
        leaves[std::make_tuple(right.start, right.end)] = &right;

        // endregion

        // region compute the next best split for the two children we just commited
        if (findBestSplit(left)) {
            frontier->push(left);
        }

        if (findBestSplit(right))
            frontier->push(right);
        // endregion
    }

    // region construct any necessary discretization mapping and statistics for branching assignments
    inSampleDiscretizations = std::vector<std::vector<size_t> >(leaves.size());
    std::vector<double> thresholds(leaves.size(), 0);

    int i = 0;
    for (const auto &[_, leaf]: leaves) {
        thresholds[i] = leaf->routingThreshold;
        inSampleDiscretizations[i] = arma::conv_to<std::vector<size_t> >::from(
            sortedOrder.subvec(leaf->start, leaf->end));
        i++;
    }

    binMapFunction = [this, thresholds = std::move(thresholds)](const arma::mat &X, arma::Row<size_t> &binLoc) {
        binLoc = arma::Row<size_t>(X.n_rows);
        for (size_t row = 0; row < X.n_rows; ++row) {
            double target = X(row, feature);
            binLoc(row) = std::distance(thresholds.begin(), std::ranges::lower_bound(thresholds, target));
        }
    };

    // endregion
}


void UnivariateDiscretizer::transform(const arma::mat &X, arma::Row<size_t> &binLoc) {
    if (binMapFunction == nullptr)
        throw std::runtime_error("Cannot transform values without first training the discretizer");

    binMapFunction(X, binLoc);
}


std::vector<std::vector<size_t> > &UnivariateDiscretizer::getInSampleDiscretizations() const {
    if (binMapFunction == nullptr)
        throw std::runtime_error("Cannot transform values without first training the discretizer");

    return inSampleDiscretizations;
}
