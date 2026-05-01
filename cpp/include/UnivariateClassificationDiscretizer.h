#pragma once
#include <vector>
#include <armadillo>
#include <functional>


class UnivariateClassificationDiscretizer {
    std::vector<std::vector<size_t> > inSampleDiscretizations;
    std::vector<size_t> binPredictions;
    std::function<void(const arma::fmat &X, arma::Row<size_t> &binLoc)> binMapFunction = nullptr;

public:
    size_t numClasses;
    size_t minLeafSize;
    double minGainSplit;
    size_t maxDepth;
    size_t maxLeafNodes;

    size_t depth{0};
    size_t numLeaves{0};
    size_t numNodes{0};

    ~UnivariateClassificationDiscretizer();

    void Train(
        const arma::fmat &X,
        arma::uvec &features,
        const arma::Row<size_t> &y,
        size_t numClasses,
        size_t minLeafSize = 1,
        double minGainSplit = 1e-7,
        size_t maxDepth = 0,
        size_t maxLeafNodes = 0
    );

    void transform(const arma::fmat &X, arma::Row<size_t> &binLoc);

    const std::vector<std::vector<size_t> > &getInSampleDiscretizations();

    const std::vector<size_t> &getBinPredictions();
};
