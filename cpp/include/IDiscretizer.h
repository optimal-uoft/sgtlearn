#pragma once
#include <vector>
#include <tuple>
#include <armadillo>
#include <functional>


//TODO: split this into one for regression and one
//for classification and regression using template generics
class IDiscretizer {
public:
    virtual ~IDiscretizer() = default;

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