#pragma once
#include <vector>
#include <tuple>
#include <compare>

struct SplitCandidate {
    double informationGain;
    size_t height;
    size_t start;
    size_t end;
    double score;
    size_t prediction;
    double routingThreshold;
    std::vector<size_t> classCounts;

    double threshold;
    size_t leftStart;
    size_t leftEnd;
    size_t leftPrediction;
    double leftScore;
    size_t rightStart;
    size_t rightEnd;
    size_t rightPrediction;
    double rightScore;
    std::vector<size_t> leftClassCounts;
    std::vector<size_t> rightClassCounts;

    bool operator==(const SplitCandidate &other) const {
        return std::tie(informationGain) == std::tie(other.informationGain);
    };

    std::weak_ordering operator<=>(const SplitCandidate &other) const {
        // Match sklearn best-first builder: prioritize only impurity improvement.
        return std::compare_weak_order_fallback(-informationGain, -other.informationGain);
    }
};