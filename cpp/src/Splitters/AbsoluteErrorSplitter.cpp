/**
 * @file AbsoluteErrorSplitter.cpp
 * @brief MAE splitter using wavelet-tree median queries on contiguous sample ranges.
 */

 #include "AbsoluteErrorSplitter.h"

 #include <limits>
 
 QuantileStats AbsoluteErrorSplitter::getMedianQuantileStats(size_t l,
                                                             size_t r) {
   return waveletTree.quantileStatsForMedian(static_cast<int>(l),
                                             static_cast<int>(r));
 }
 
 AbsoluteErrorSplitter::AbsoluteErrorSplitter(arma::frowvec &X,
                                              arma::Mat<float> &y)
     : Splitter(X, y, 0), waveletTree(arma::Row<float>(y.row(0))) {}
 
 SplitCandidate AbsoluteErrorSplitter::makeRoot() {
   return {.height = 0,
           .start = 0,
           .end = y.n_cols - 1,
           .score = score(makeEmptyStats(), 0, y.n_cols - 1),
           .routingThreshold = std::numeric_limits<double>::infinity()};
 }
 
 float AbsoluteErrorSplitter::predict(const SplitCandidate &split) {
   return getMedianQuantileStats(split.start, split.end).median_val;
 }
 
 double AbsoluteErrorSplitter::score(const std::vector<float> &stats, size_t l,
                                     size_t r) {
   return getMedianQuantileStats(l, r).mae();
 }
 
 double AbsoluteErrorSplitter::score(const SplitCandidate &split) {
   return score(makeEmptyStats(), split.start, split.end);
 }
 
 const std::vector<float> &
 AbsoluteErrorSplitter::getStats(const SplitCandidate &split) {
   return empty_stats_cache_;
 }