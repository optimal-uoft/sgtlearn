#pragma once
#include "frontiers.h"
#include <functional>
#include <limits>

template <typename T> class TreeBuilder {
public:
  size_t minLeafSize;
  double minGainSplit;
  size_t maxDepth;
  size_t maxLeafNodes;
  double eps = std::numeric_limits<double>::epsilon();

  TreeBuilder(size_t minLeafSize, double minGainSplit, size_t maxDepth,
              size_t maxLeafNodes)
      : minLeafSize(minLeafSize), minGainSplit(minGainSplit),
        maxDepth(maxDepth), maxLeafNodes(maxLeafNodes) {}

  void buildTree(T &root, std::function<bool(T &, size_t)> findBestSplit,
                 std::function<std::vector<T>(T &)> makeChildren,
                 std::function<void(T &, std::vector<T> &)> commitSplit

  );
};

#include "algorithms/TreeBuilder.tpp"