#pragma once
#include <vector>
#include <tuple>
#include <armadillo>
#include <functional>


//TODO: split this into one for regression and one
// for classification using template generics
class IDiscretizer {
public:
    virtual ~IDiscretizer() = 0;

    //for regression
    virtual void Train(
        const arma::fmat &X,
        arma::uvec &features,
        const arma::frowvec &responses,
        size_t minLeafSize = 1,
        double minGainSplit = 1e-7,
        size_t maxDepth = 0,
        size_t maxLeafNodes = 0
    ) = 0;

    // for classification
    virtual void Train(
        const arma::fmat &X,
        arma::uvec &features,
        const arma::Row<size_t> &labels,
        size_t numClasses,
        size_t minLeafSize = 1,
        double minGainSplit = 1e-7,
        size_t maxDepth = 0,
        size_t maxLeafNodes = 0
    ) = 0;

    virtual void transform(const arma::fmat &X, arma::Row<size_t> &binLoc) = 0;

    virtual const std::vector<std::vector<size_t> > &getInSampleDiscretizations() = 0;

    virtual const std::vector<size_t> &getBinPredictions() = 0;
};


struct SplitCandidate {
    double informationGain;
    size_t height;
    size_t start;
    size_t end;
    double score;
    size_t prediction;
    double routingThreshold;

    double threshold;
    size_t leftStart;
    size_t leftEnd;
    size_t leftPrediction;
    double leftScore;
    size_t rightStart;
    size_t rightEnd;
    size_t rightPrediction;
    double rightScore;

    bool operator==(const SplitCandidate &other) const {
        return std::tie(informationGain, height) == std::tie(other.informationGain, other.height);
    };

    std::weak_ordering operator<=>(const SplitCandidate &other) const {
        // Match sklearn best-first builder: prioritize only impurity improvement.
        return std::compare_weak_order_fallback(-informationGain, -other.informationGain);
    }
};

class UnivariateDiscretizer : public IDiscretizer {
    arma::uvec sortedOrder;
    arma::fmat X;
    arma::Mat<size_t> prefix;
    size_t feature;
    size_t totalSamples{0};

    std::vector<std::vector<size_t> > inSampleDiscretizations;
    std::vector<size_t> binPredictions;
    std::function<void(const arma::fmat &X, arma::Row<size_t> &binLoc)> binMapFunction = nullptr;


    bool findBestSplit(SplitCandidate &split);

    void finalizeTraining(std::map<std::tuple<size_t, size_t>, SplitCandidate> &leaves);

public:
    size_t numClasses;
    size_t minLeafSize;
    double minGainSplit;
    size_t maxDepth;
    size_t maxLeafNodes;

    size_t depth{0};
    size_t numLeaves{0};
    size_t numNodes{0};

    ~UnivariateDiscretizer() override;

    void Train(
        const arma::fmat &X,
        arma::uvec &features,
        const arma::frowvec &responses,
        size_t minLeafSize = 1,
        double minGainSplit = 1e-7,
        size_t maxDepth = 0,
        size_t maxLeafNodes = 0
    ) override;

    void Train(
        const arma::fmat &X,
        arma::uvec &features,
        const arma::Row<size_t> &labels,
        size_t numClasses,
        size_t minLeafSize = 1,
        double minGainSplit = 1e-7,
        size_t maxDepth = 0,
        size_t maxLeafNodes = 0
    ) override;

    void transform(const arma::fmat &X, arma::Row<size_t> &binLoc) override;

    const std::vector<std::vector<size_t> > &getInSampleDiscretizations() override;

    const std::vector<size_t> &getBinPredictions() override;
};
