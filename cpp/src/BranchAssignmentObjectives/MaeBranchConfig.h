#pragma once

/**
 * @file MaeBranchConfig.h
 * @brief Runtime toggles for AbsoluteError branch assignment (benchmarking / experiments).
 *
 * Environment variables (read on each call):
 * - ``SGTLEARN_MAE_BACKEND``: ``merge`` (default), ``bst``, or ``sort``
 * - ``SGTLEARN_MAE_CD``: ``1`` / ``true`` enables coordinate descent for
 *   ``absolute_error`` (default off for sklearn CART parity)
 */

#include <cstdlib>
#include <cstring>

namespace mae_branch_config {

enum class Backend { Merge, Bst, Sort };

inline Backend backend() {
  const char *v = std::getenv("SGTLEARN_MAE_BACKEND");
  if (v == nullptr)
    return Backend::Merge;
  if (std::strcmp(v, "sort") == 0 || std::strcmp(v, "Sort") == 0)
    return Backend::Sort;
  if (std::strcmp(v, "bst") == 0 || std::strcmp(v, "Bst") == 0)
    return Backend::Bst;
  // Explicit merge, unknown values, or empty → default merge.
  return Backend::Merge;
}

inline bool coordinateDescentEnabled() {
  const char *v = std::getenv("SGTLEARN_MAE_CD");
  return v != nullptr &&
         (std::strcmp(v, "1") == 0 || std::strcmp(v, "true") == 0 ||
          std::strcmp(v, "TRUE") == 0 || std::strcmp(v, "yes") == 0);
}

} // namespace mae_branch_config
