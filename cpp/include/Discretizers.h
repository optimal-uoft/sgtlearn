#pragma once
#include <vector>
#include <tuple>
#include <armadillo>
#include <functional>

class IDiscretizer {
public:
    virtual ~IDiscretizer() = default;


    //for regression
    virtual double Train(
        const arma::mat &X,
        arma::uvec &features,
        const arma::Row<double> &responses,
        size_t minLeafSize = 1,
        double minGainSplit = 1e-7,
        size_t maxDepth = 0,
        size_t maxLeafNodes = 0
    );

    // for classification
    virtual double Train(
        const arma::mat &X,
        arma::uvec &features,
        const arma::Row<size_t> &labels,
        size_t numClasses,
        size_t minLeafSize = 1,
        double minGainSplit = 1e-7,
        size_t maxDepth = 0,
        size_t maxLeafNodes = 0
    );

    virtual void transform(const arma::mat &X, arma::Row<size_t> &binLoc);

    virtual std::vector<std::vector<size_t>> &getInSampleDiscretizations() const;
};


struct SplitCandidate {
    double informationGain;
    size_t height;
    size_t start;
    size_t end;
    double score;
    double routingThreshold;

    double threshold;
    size_t leftStart;
    size_t leftEnd;
    double leftScore;
    size_t rightStart;
    size_t rightEnd;
    double rightScore;

    bool operator==(const SplitCandidate &other) const {
        return std::tie(informationGain, height) == std::tie(other.informationGain, other.height);
    };

    std::weak_ordering operator<=>(const SplitCandidate &other) const {
        if (auto cmp = std::compare_weak_order_fallback(-informationGain, -other.informationGain); cmp != 0) {
            return cmp;
        }
        return height <=> other.height;
    }
};

class UnivariateDiscretizer : public IDiscretizer {
    arma::uvec sortedOrder;
    const arma::vec X;
    const arma::Row<double> responses;
    const arma::Row<size_t> labels;
    arma::Mat<size_t> prefix;
    size_t feature;
    std::vector<std::vector<size_t>> &inSampleDiscretizations;
    std::function<void(const arma::mat &X, arma::Row<size_t> &binLoc)> binMapFunction = nullptr;

    bool findBestSplit(SplitCandidate &split);

public:
    size_t numClasses;
    size_t minLeafSize;
    double minGainSplit;
    size_t maxDepth;
    size_t maxLeafNodes;

    size_t depth{0};
    size_t numLeaves{0};
    size_t numNodes{0};

    UnivariateDiscretizer();

    double Train(
        const arma::mat &X,
        arma::uvec &features,
        const arma::Row<double> &responses,
        size_t minLeafSize = 1,
        double minGainSplit = 1e-7,
        size_t maxDepth = 0,
        size_t maxLeafNodes = 0
    ) override;

    double Train(
        const arma::mat &X,
        arma::uvec &features,
        const arma::Row<size_t> &labels,
        size_t numClasses,
        size_t minLeafSize = 1,
        double minGainSplit = 1e-7,
        size_t maxDepth = 0,
        size_t maxLeafNodes = 0
    ) override;

    void transform(const arma::mat &X, arma::Row<size_t> &binLoc) override;

    std::vector<std::vector<size_t>> &getInSampleDiscretizations() const override;
};
