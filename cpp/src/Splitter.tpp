template <typename T>
const std::vector<T> &Splitter<T>::getStats(const SplitCandidate &split) {
  auto startIt = splitStats.find(split.start);
  if (startIt == splitStats.end())
    throw std::runtime_error(
        "stats for the given split partition does not exist");
  auto endIt = startIt->second.find(split.end);
  if (endIt == startIt->second.end())
    throw std::runtime_error(
        "stats for the given split partition does not exist");

  return endIt->second;
}

template <typename T> std::vector<T> Splitter<T>::makeEmptyStats() {
  return std::vector<T>(statsSize, 0);
}

template <typename T>
std::array<SplitCandidate, 2>
Splitter<T>::makeChildren(const SplitCandidate &parent) {
  auto statsStartIt = childrenSplitStats.find(parent.start);
  if (statsStartIt == childrenSplitStats.end())
    throw std::runtime_error(
        "Cannot make children for a class that has not found a best split");
  auto statsEndIt = statsStartIt->second.find(parent.end);
  if (statsEndIt == statsStartIt->second.end())
    throw std::runtime_error(
        "Cannot make children for a class that has not found a best split");

  auto &stats = statsEndIt->second;

  SplitCandidate left{
      .height = parent.height + 1,
      .start = parent.leftStart,
      .end = parent.leftEnd,
      .score = parent.leftScore,
      .routingThreshold = parent.threshold,
  };
  SplitCandidate right{
      .height = parent.height + 1,
      .start = parent.rightStart,
      .end = parent.rightEnd,
      .score = parent.rightScore,
      .routingThreshold = parent.routingThreshold,
  };

  splitStats[left.start][left.end] = std::move(stats[0]);
  splitStats[right.start][right.end] = std::move(stats[1]);
  return {left, right};
}

template <typename T>
bool Splitter<T>::findBestSplit(SplitCandidate &split, const arma::frowvec &X,
                                const arma::Row<T> &y, size_t minLeafSize) {

  std::vector<T> leftStats = makeEmptyStats();
  std::vector<T> rightStats(getStats(split));

  const size_t N = split.end - split.start + 1;
  if (N < 2 * minLeafSize)
    return false;

  std::vector<T> foundLeftStats;
  std::vector<T> foundRightStats;
  double bestProxyImprovement = -std::numeric_limits<double>::infinity();
  bool found = false;

  for (size_t i = split.start + 1; i <= split.end; ++i) {
    moveSample(rightStats, leftStats, y(i - 1));

    const size_t Nl = i - split.start;
    const size_t Nr = split.end - i + 1;
    if (Nl < minLeafSize)
      continue;
    if (Nr < minLeafSize)
      break;

    const float currValue = X(i);
    const float prevValue = X(i - 1);
    if (currValue <= prevValue + static_cast<float>(1e-7))
      continue;

    const double leftScore = score(leftStats, Nl);
    const double rightScore = score(rightStats, Nr);
    const double proxyImprovement = -static_cast<double>(Nr) * rightScore -
                                    static_cast<double>(Nl) * leftScore;
    if (proxyImprovement > bestProxyImprovement) {
      found = true;
      bestProxyImprovement = proxyImprovement;

      double threshold = static_cast<double>(prevValue) / 2.0 +
                         static_cast<double>(currValue) / 2.0;
      if (threshold == static_cast<double>(currValue) ||
          !std::isfinite(threshold)) {
        threshold = static_cast<double>(prevValue);
      }
      split.threshold = threshold;

      split.leftStart = split.start;
      split.leftEnd = i - 1;
      split.leftScore = leftScore;
      foundLeftStats = leftStats;

      split.rightStart = i;
      split.rightEnd = split.end;
      split.rightScore = rightScore;
      foundRightStats = rightStats;

      const double gain =
          split.score -
          (static_cast<double>(Nl) / static_cast<double>(N) * leftScore +
           static_cast<double>(Nr) / static_cast<double>(N) * rightScore);
      split.informationGain =
          static_cast<double>(N) / static_cast<double>(X.n_cols) * gain;
    }
  }
  if (found)
    childrenSplitStats[split.start][split.end] = {foundLeftStats,
                                                   foundRightStats};

  return found;
}
