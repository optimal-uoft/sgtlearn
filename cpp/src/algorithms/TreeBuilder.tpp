/**
 * @file TreeBuilder.tpp
 * @brief Template implementation of ``TreeBuilder::buildTree``.
 */

template <typename T>
void TreeBuilder<T>::buildTree(
    T &root, std::function<bool(T &, size_t)> findBestSplit,
    std::function<std::vector<T>(T &)> makeChildren,
    std::function<void(T &, std::vector<T> &)> commitSplit) {
  std::unique_ptr<frontiers::IFrontier<T>> frontier;
  if (maxLeafNodes == 0)
    frontier = std::make_unique<frontiers::Stack<T>>();
  else
    frontier = std::make_unique<frontiers::Heap<T>>();

  size_t numLeaves = 0;

  if (root.score > eps && findBestSplit(root, minLeafSize)) {
    frontier->push(root);
    numLeaves++;
  }

  while (frontier->size() > 0 &&
         (maxLeafNodes == 0 || numLeaves < maxLeafNodes)) {
    auto split = frontier->peek();
    frontier->pop();

    if (split.score <= eps)
      continue;

    if (split.informationGain + eps < minGainSplit)
      continue;

    if (maxDepth != 0 && split.height >= maxDepth)
      continue;

    auto children = makeChildren(split);

    commitSplit(split, children);
    numLeaves += children.size() - 1;

    for (auto child : children) {
      if (child.score > eps && (maxDepth == 0 || child.height < maxDepth) &&
          findBestSplit(child, minLeafSize))
        frontier->push(child);
    }
  }
}
