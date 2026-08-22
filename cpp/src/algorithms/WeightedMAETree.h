#pragma once

/**
 * @file WeightedMAETree.h
 * @brief Augmented AVL multiset for dynamic weighted median and pinball MAE.
 *
 * Keys are merged by value: each distinct ``y`` stores total weight at that
 * value. Inserts/erases of a batch of size ``K`` are ``O(K log N)``. Median and
 * MAE queries are ``O(log N)`` via subtree weight / ``Σw·y`` aggregates.
 */

#include <cstddef>
#include <utility>
#include <vector>

/**
 * Self-balancing BST ordered by ``y``, maintaining per-subtree
 * ``(Σw, Σw·y)`` so weighted median and MAE match ``Criterion::absoluteError``
 * under batch membership updates.
 */
class WeightedMAETree {
public:
  WeightedMAETree() = default;
  ~WeightedMAETree() { clear(); }

  WeightedMAETree(const WeightedMAETree &) = delete;
  WeightedMAETree &operator=(const WeightedMAETree &) = delete;

  WeightedMAETree(WeightedMAETree &&other) noexcept
      : root_(other.root_), totalWeight_(other.totalWeight_),
        totalWy_(other.totalWy_) {
    other.root_ = nullptr;
    other.totalWeight_ = 0.0;
    other.totalWy_ = 0.0;
  }

  WeightedMAETree &operator=(WeightedMAETree &&other) noexcept {
    if (this != &other) {
      clear();
      root_ = other.root_;
      totalWeight_ = other.totalWeight_;
      totalWy_ = other.totalWy_;
      other.root_ = nullptr;
      other.totalWeight_ = 0.0;
      other.totalWy_ = 0.0;
    }
    return *this;
  }

  void clear();

  /** Insert ``(y[i], w[i])`` for all ``i``; ``O(K log N)``. */
  void insert_batch(const std::vector<float> &ys,
                    const std::vector<float> &ws);

  /** Erase the same multiset of pairs previously inserted; ``O(K log N)``. */
  void remove_batch(const std::vector<float> &ys,
                    const std::vector<float> &ws);

  void insert(double y, double w);
  void erase(double y, double w);

  double totalWeight() const { return totalWeight_; }

  /** Weighted median (sklearn half-tie average). ``O(log N)``. */
  double median() const;

  /** Mean absolute error about ``median()``. ``O(log N)``. */
  double mae() const;

  /** ``(median, mae)`` in one walk + aggregate query. */
  std::pair<double, double> medianAndMae() const;

private:
  struct Node {
    double key = 0.0;
    double weight = 0.0;
    double subWeight = 0.0;
    double subWy = 0.0;
    int height = 1;
    Node *left = nullptr;
    Node *right = nullptr;
  };

  Node *root_ = nullptr;
  double totalWeight_ = 0.0;
  double totalWy_ = 0.0;

  static int heightOf(const Node *n) { return n ? n->height : 0; }
  static double weightOf(const Node *n) { return n ? n->subWeight : 0.0; }
  static double wyOf(const Node *n) { return n ? n->subWy : 0.0; }

  static void pull(Node *n);
  static Node *rotateLeft(Node *x);
  static Node *rotateRight(Node *y);
  static Node *balance(Node *n);

  Node *insertNode(Node *n, double key, double w);
  Node *eraseNode(Node *n, double key, double w);
  static Node *minNode(Node *n);
  static Node *eraseMin(Node *n);
  static void destroy(Node *n);

  /** Weight / ``Σw·y`` over keys strictly ``< key``. */
  static void aggregatesLessThan(const Node *n, double key, double &wOut,
                                 double &wyOut);

  /** Largest key strictly less than ``key``, or false if none. */
  static bool predecessor(const Node *n, double key, double &out);
};
