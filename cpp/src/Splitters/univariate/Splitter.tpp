/**
 * @file Splitter.tpp
 * @brief Template implementation of generic ``Splitter`` helpers and ``findBestSplit``.
 */

template <typename StatsT, typename PredictT>
void Splitter<StatsT, PredictT>::buildWeightPrefix() {
  weightPrefix.assign(sampleWeights.n_elem + 1, 0.0);
  for (size_t i = 0; i < sampleWeights.n_elem; ++i)
    weightPrefix[i + 1] = weightPrefix[i] + static_cast<double>(sampleWeights(i));
}

template <typename StatsT, typename PredictT>
double Splitter<StatsT, PredictT>::intervalWeight(size_t l, size_t r) const {
  if (r < l)
    return 0.0;
  return weightPrefix[r + 1] - weightPrefix[l];
}

template <typename StatsT, typename PredictT>
size_t Splitter<StatsT, PredictT>::intervalNumSamples(size_t l, size_t r) {
  if (r < l)
    return 0;
  return r - l + 1;
}

template <typename StatsT, typename PredictT>
void Splitter<StatsT, PredictT>::fillIntervalMeta(UnivariateSplitCandidate &split) const {
  split.numSamples = intervalNumSamples(split.start, split.end);
  split.nodeWeight = intervalWeight(split.start, split.end);
}

template <typename StatsT, typename PredictT>
const std::vector<StatsT> &
Splitter<StatsT, PredictT>::getStats(const UnivariateSplitCandidate &split) {
  if (!splitStats.contains(split.start) ||
      !splitStats[split.start].contains(split.end))
    throw std::runtime_error(
        "stats for the given split partition does not exist");

  return splitStats[split.start][split.end];
}

template <typename StatsT, typename PredictT>
std::vector<StatsT> Splitter<StatsT, PredictT>::makeEmptyStats() {
  return std::vector<StatsT>(statsSize, StatsT{0});
}

template <typename StatsT, typename PredictT>
std::vector<UnivariateSplitCandidate>
Splitter<StatsT, PredictT>::makeChildren(const UnivariateSplitCandidate &parent) {

  if (!childrenSplitStats.contains(parent.start) ||
      !childrenSplitStats[parent.start].contains(parent.end))
    throw std::runtime_error(
        "Cannot make children for a class that has not found a best split");
  auto &stats = childrenSplitStats[parent.start][parent.end];

  UnivariateSplitCandidate left{
      .height = parent.height + 1,
      .start = parent.leftStart,
      .end = parent.leftEnd,
      .numSamples = parent.leftNumSamples,
      .nodeWeight = parent.leftWeight,
      .score = parent.leftScore,
      .routingThreshold = parent.threshold,
  };
  UnivariateSplitCandidate right{
      .height = parent.height + 1,
      .start = parent.rightStart,
      .end = parent.rightEnd,
      .numSamples = parent.rightNumSamples,
      .nodeWeight = parent.rightWeight,
      .score = parent.rightScore,
      .routingThreshold = parent.routingThreshold,
  };

  splitStats[left.start][left.end] = std::move(stats[0]);
  splitStats[right.start][right.end] = std::move(stats[1]);
  return {right, left};
}

template <typename StatsT, typename PredictT>
bool Splitter<StatsT, PredictT>::findBestSplit(UnivariateSplitCandidate &split,
                                               size_t minLeafSize) {

  std::vector<StatsT> leftStats = makeEmptyStats();
  std::vector<StatsT> rightStats(getStats(split));

  const size_t N = split.numSamples;
  const double W = split.nodeWeight;
  if (N < 2 * minLeafSize || W <= 0.0)
    return false;

  std::vector<StatsT> foundLeftStats;
  std::vector<StatsT> foundRightStats;
  double bestProxyImprovement = -std::numeric_limits<double>::infinity();
  bool found = false;

  for (size_t i = split.start + 1; i <= split.end; ++i) {
    const float currValue = X(i);
    if (!missing_values::is_finite(currValue))
      break;

    moveSample(rightStats, leftStats, i - 1);

    const size_t Nl = i - split.start;
    const size_t Nr = split.end - i + 1;
    if (Nl < minLeafSize)
      continue;
    if (Nr < minLeafSize)
      break;

    const float prevValue = X(i - 1);
    if (!missing_values::is_finite(prevValue))
      throw std::runtime_error("NaN value passed to findBestSplit");
    if (currValue <= prevValue + static_cast<float>(1e-7))
      continue;

    const double Wl = intervalWeight(split.start, i - 1);
    const double Wr = intervalWeight(i, split.end);
    const double leftScore = score(leftStats, split.start, i - 1);
    const double rightScore = score(rightStats, i, split.end);
    const double proxyImprovement = -Wr * rightScore - Wl * leftScore;
    if (proxyImprovement <= bestProxyImprovement)
      continue;
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
    split.leftNumSamples = Nl;
    split.leftWeight = Wl;
    split.leftScore = leftScore;
    foundLeftStats = leftStats;

    split.rightStart = i;
    split.rightEnd = split.end;
    split.rightNumSamples = Nr;
    split.rightWeight = Wr;
    split.rightScore = rightScore;
    foundRightStats = rightStats;

    const double gain =
        split.score - (Wl / W * leftScore + Wr / W * rightScore);
    const double weightScale =
        totalSampleWeight() > 0.0 ? W / totalSampleWeight()
                                  : static_cast<double>(N) /
                                        static_cast<double>(X.n_cols);
    split.informationGain = weightScale * gain;
  }
  if (found)
    childrenSplitStats[split.start][split.end] = {foundLeftStats,
                                                  foundRightStats};

  return found;
}
