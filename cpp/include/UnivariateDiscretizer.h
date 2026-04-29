#pragma once
#include <vector>
#include <tuple>
#include <armadillo>
#include <functional>
#include "IDiscretizer.h"
#include "SplitCandidate.h"


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
