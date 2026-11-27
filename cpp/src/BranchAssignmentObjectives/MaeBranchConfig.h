#pragma once

/**
 * @file MaeBranchConfig.h
 * @brief Runtime toggle for AbsoluteError coordinate descent.
 *
 * Environment variable (read on each call):
 * - ``SGTLEARN_MAE_CD``: ``1`` / ``true`` / ``TRUE`` / ``yes`` enables
 *   coordinate descent for ``absolute_error`` (default off for sklearn CART parity)
 */

#include <cstdlib>
#include <cstring>

namespace mae_branch_config {

inline bool coordinateDescentEnabled() {
  const char *v = std::getenv("SGTLEARN_MAE_CD");
  return v != nullptr &&
         (std::strcmp(v, "1") == 0 || std::strcmp(v, "true") == 0 ||
          std::strcmp(v, "TRUE") == 0 || std::strcmp(v, "yes") == 0);
}

} // namespace mae_branch_config
