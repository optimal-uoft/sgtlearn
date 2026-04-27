#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>
#include "Discretizers.h"
#include <mlpack/methods/decision_tree/decision_tree.hpp>


void classificationPredict(UnivariateDiscretizer &ud, arma::mat &X, arma::Row<size_t> &predictions) {
    throw std::runtime_error("not implemented");
}

TEST_CASE("UnivariateDiscretizer vs mlpack fidelity check", "[parameters]") {
    // define testing grid
    auto _N = GENERATE(1000, 5000, 10000);
    auto _numClasses = GENERATE(2, 3);
    auto _minimumLeafSize = GENERATE(1, 10);
    auto _minimumGainSplit = GENERATE(1E-07, 0.5);
    auto _maximumDepth = GENERATE(0, 4);
    auto pair = std::make_tuple(_N, _numClasses, _minimumLeafSize, _minimumGainSplit, _maximumDepth);

    SECTION("Testing param combinations") {
        int N = std::get<0>(pair);
        int numClasses = std::get<1>(pair);
        size_t minimumLeafSize = std::get<2>(pair);
        double minimumGainSplit = std::get<3>(pair);
        size_t maximumDepth = std::get<4>(pair);

        arma::mat dataset(1, N, arma::fill::randu);
        arma::Row<size_t> labels =
                arma::randi<arma::Row<size_t> >(N, arma::distr_param(0, numClasses - 1));

        mlpack::DecisionTree mlPackTree;
        double mlPackEntropy = mlPackTree.Train(
            dataset,
            labels,
            numClasses,
            minimumLeafSize,
            minimumGainSplit,
            maximumDepth
        );
        arma::Row<size_t> mlPackPreds;
        mlPackTree.Classify(dataset, mlPackPreds);

        UnivariateDiscretizer ud;
        arma::uvec features = {0};
        double udEntropy = ud.Train(
            dataset,
            features,
            labels,
            numClasses,
            minimumLeafSize,
            minimumGainSplit,
            maximumDepth
        );
        arma::Row<size_t> udPreds;
        classificationPredict(ud, dataset, udPreds);


        REQUIRE(mlPackEntropy >= udEntropy);
        REQUIRE(arma::all(mlPackPreds == udPreds));
        // todo: check for depth and number of nodes are the same
    }
}
