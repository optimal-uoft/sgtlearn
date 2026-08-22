/**
 * @file WeightedMAETree.cpp
 * @brief Augmented AVL implementation for dynamic weighted median / MAE.
 */

#include "algorithms/WeightedMAETree.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace {

constexpr double kWeightEps = 1e-15;
constexpr double kHalfTieEps = 1e-12;

} // namespace

void WeightedMAETree::pull(Node *n) {
  if (!n)
    return;
  n->height = 1 + std::max(heightOf(n->left), heightOf(n->right));
  n->subWeight = n->weight + weightOf(n->left) + weightOf(n->right);
  n->subWy = n->key * n->weight + wyOf(n->left) + wyOf(n->right);
}

WeightedMAETree::Node *WeightedMAETree::rotateLeft(Node *x) {
  Node *y = x->right;
  x->right = y->left;
  y->left = x;
  pull(x);
  pull(y);
  return y;
}

WeightedMAETree::Node *WeightedMAETree::rotateRight(Node *y) {
  Node *x = y->left;
  y->left = x->right;
  x->right = y;
  pull(y);
  pull(x);
  return x;
}

WeightedMAETree::Node *WeightedMAETree::balance(Node *n) {
  pull(n);
  const int bf = heightOf(n->left) - heightOf(n->right);
  if (bf > 1) {
    if (heightOf(n->left->right) > heightOf(n->left->left))
      n->left = rotateLeft(n->left);
    return rotateRight(n);
  }
  if (bf < -1) {
    if (heightOf(n->right->left) > heightOf(n->right->right))
      n->right = rotateRight(n->right);
    return rotateLeft(n);
  }
  return n;
}

void WeightedMAETree::destroy(Node *n) {
  if (!n)
    return;
  destroy(n->left);
  destroy(n->right);
  delete n;
}

void WeightedMAETree::clear() {
  destroy(root_);
  root_ = nullptr;
  totalWeight_ = 0.0;
  totalWy_ = 0.0;
}

WeightedMAETree::Node *WeightedMAETree::insertNode(Node *n, double key,
                                                   double w) {
  if (!n) {
    Node *created = new Node();
    created->key = key;
    created->weight = w;
    pull(created);
    return created;
  }
  if (key < n->key)
    n->left = insertNode(n->left, key, w);
  else if (key > n->key)
    n->right = insertNode(n->right, key, w);
  else
    n->weight += w;
  return balance(n);
}

WeightedMAETree::Node *WeightedMAETree::minNode(Node *n) {
  while (n && n->left)
    n = n->left;
  return n;
}

WeightedMAETree::Node *WeightedMAETree::eraseMin(Node *n) {
  if (!n->left)
    return n->right;
  n->left = eraseMin(n->left);
  return balance(n);
}

WeightedMAETree::Node *WeightedMAETree::eraseNode(Node *n, double key,
                                                  double w) {
  if (!n)
    throw std::runtime_error("WeightedMAETree::erase: key not found");

  if (key < n->key)
    n->left = eraseNode(n->left, key, w);
  else if (key > n->key)
    n->right = eraseNode(n->right, key, w);
  else {
    n->weight -= w;
    if (n->weight < -kWeightEps)
      throw std::runtime_error("WeightedMAETree::erase: weight underflow");
    if (n->weight <= kWeightEps) {
      Node *left = n->left;
      Node *right = n->right;
      delete n;
      if (!right)
        return left;
      if (!left)
        return right;
      Node *m = minNode(right);
      m->right = eraseMin(right);
      m->left = left;
      return balance(m);
    }
  }
  return balance(n);
}

void WeightedMAETree::insert(double y, double w) {
  if (w < 0.0)
    throw std::invalid_argument("WeightedMAETree::insert: negative weight");
  if (w <= kWeightEps)
    return;
  root_ = insertNode(root_, y, w);
  totalWeight_ += w;
  totalWy_ += y * w;
}

void WeightedMAETree::erase(double y, double w) {
  if (w < 0.0)
    throw std::invalid_argument("WeightedMAETree::erase: negative weight");
  if (w <= kWeightEps)
    return;
  root_ = eraseNode(root_, y, w);
  totalWeight_ -= w;
  totalWy_ -= y * w;
  if (totalWeight_ < 0.0 && totalWeight_ > -kWeightEps)
    totalWeight_ = 0.0;
  if (std::fabs(totalWy_) < kWeightEps)
    totalWy_ = 0.0;
}

void WeightedMAETree::insert_batch(const std::vector<float> &ys,
                                   const std::vector<float> &ws) {
  if (ys.size() != ws.size())
    throw std::invalid_argument(
        "WeightedMAETree::insert_batch: ys/ws size mismatch");
  for (size_t i = 0; i < ys.size(); ++i)
    insert(static_cast<double>(ys[i]), static_cast<double>(ws[i]));
}

void WeightedMAETree::remove_batch(const std::vector<float> &ys,
                                   const std::vector<float> &ws) {
  if (ys.size() != ws.size())
    throw std::invalid_argument(
        "WeightedMAETree::remove_batch: ys/ws size mismatch");
  for (size_t i = 0; i < ys.size(); ++i)
    erase(static_cast<double>(ys[i]), static_cast<double>(ws[i]));
}

void WeightedMAETree::aggregatesLessThan(const Node *n, double key, double &wOut,
                                         double &wyOut) {
  while (n) {
    if (key <= n->key) {
      n = n->left;
    } else {
      wOut += weightOf(n->left) + n->weight;
      wyOut += wyOf(n->left) + n->key * n->weight;
      n = n->right;
    }
  }
}

bool WeightedMAETree::predecessor(const Node *n, double key, double &out) {
  bool found = false;
  while (n) {
    if (n->key < key) {
      out = n->key;
      found = true;
      n = n->right;
    } else {
      n = n->left;
    }
  }
  return found;
}

double WeightedMAETree::median() const { return medianAndMae().first; }

double WeightedMAETree::mae() const { return medianAndMae().second; }

std::pair<double, double> WeightedMAETree::medianAndMae() const {
  if (!root_ || totalWeight_ <= 0.0)
    return {0.0, 0.0};

  const double half = 0.5 * totalWeight_;
  double wLeft = 0.0;
  double wyLeft = 0.0;
  const Node *n = root_;
  const Node *medianNode = nullptr;

  while (n) {
    const double leftW = weightOf(n->left);
    if (wLeft + leftW > half) {
      n = n->left;
      continue;
    }
    if (wLeft + leftW + n->weight > half) {
      wLeft += leftW;
      wyLeft += wyOf(n->left);
      medianNode = n;
      break;
    }
    wLeft += leftW + n->weight;
    wyLeft += wyOf(n->left) + n->key * n->weight;
    n = n->right;
  }

  if (!medianNode) {
    // Degenerate: land on rightmost key (mirrors Criterion fallback).
    n = root_;
    while (n->right)
      n = n->right;
    medianNode = n;
    wLeft = totalWeight_ - n->weight;
    wyLeft = totalWy_ - n->key * n->weight;
  }

  double m = medianNode->key;
  if (std::fabs(wLeft - half) <= kHalfTieEps) {
    double pred = 0.0;
    if (predecessor(root_, medianNode->key, pred))
      m = 0.5 * (pred + medianNode->key);
  }

  // Pinball about m with value split (y < m vs y > m); ties at m contribute 0.
  double wLt = 0.0;
  double wyLt = 0.0;
  aggregatesLessThan(root_, m, wLt, wyLt);

  double wEq = 0.0;
  double wyEq = 0.0;
  if (std::fabs(m - medianNode->key) <= kHalfTieEps) {
    wEq = medianNode->weight;
    wyEq = medianNode->key * medianNode->weight;
  } else {
    // Half-tie average is not an inserted key; scan for an exact match anyway.
    const Node *eq = root_;
    while (eq) {
      if (m < eq->key)
        eq = eq->left;
      else if (m > eq->key)
        eq = eq->right;
      else {
        wEq = eq->weight;
        wyEq = eq->key * eq->weight;
        break;
      }
    }
  }

  const double wGt = totalWeight_ - wLt - wEq;
  const double wyGt = totalWy_ - wyLt - wyEq;
  const double pinball = (m * wLt - wyLt) + (wyGt - m * wGt);
  return {m, pinball / totalWeight_};
}
