#pragma once
#include <concepts>
#include <queue>
#include <stack>
#include <vector> // Required for priority_queue's default container

namespace frontiers {
template <typename T> class IFrontier {
public:
  virtual ~IFrontier() = default;
  virtual const T &peek() const = 0;
  virtual void pop() = 0;
  virtual void push(const T &x) = 0;
  virtual size_t size() const = 0; // Added const

  virtual bool empty() const { return size() == 0; } // Helpful helper
};

template <typename T> class Stack : public IFrontier<T> {
  std::stack<T> s;

public:
  const T &peek() const override { return s.top(); }
  void pop() override { s.pop(); }
  void push(const T &x) override { s.push(x); }
  size_t size() const override { return s.size(); }
};

template <std::three_way_comparable T> class Heap : public IFrontier<T> {
  std::priority_queue<T> q;

public:
  const T &peek() const override { return q.top(); }
  void pop() override { q.pop(); }
  void push(const T &x) override { q.push(x); }
  size_t size() const override { return q.size(); }
};
} // namespace frontiers
