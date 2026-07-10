#pragma once

/**
 * @file TreeBuilder.h
 * @brief Generic best-first or depth-first tree expansion over a node type ``T`` with pluggable split callbacks.
 */

#include <cstddef>
#include "frontiers.h"
#include <functional>
#include <limits>

/**
 * Expands a decision tree using a stack (``maxLeafNodes == 0``) or a best-first heap.
 *
 * @tparam T node type providing ``score``, ``informationGain``, ``height``, and ``operator<=>`` for the heap frontier.
 */
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

  /**
   * @param findBestSplit mutates a node in place; returns false if it becomes a leaf.
   * @param makeChildren builds child nodes after a committed split.
   * @param commitSplit persists parent + children into the trainer's storage.
   */
  void buildTree(T &root, std::function<bool(T &, size_t)> findBestSplit,
                 std::function<std::vector<T>(T &)> makeChildren,
                 std::function<void(T &, std::vector<T> &)> commitSplit);
};

#include "algorithms/TreeBuilder.tpp"