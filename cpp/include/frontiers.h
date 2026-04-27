#pragma once
#include <concepts>
#include <stack>
#include <queue>


namespace frontiers {
    template<typename T>
    class IFrontier {
    public:
        virtual ~IFrontier() = default;

        virtual const T &peek() const = 0;

        virtual void pop() = 0;

        virtual void push(const T &x) = 0;

        virtual size_t size() = 0;
    };

    template<typename T>
    class Stack : public IFrontier<T> {
        std::stack<T> s;

    public:
        Stack() = default;

        const T &peek() const override { return s.top(); }

        void pop() override { s.pop(); }

        void push(const T &x) override { s.push(x); }

        size_t size() override { return s.size(); }
    };

    template<std::totally_ordered T>
    class MinHeap : public IFrontier<T> {
        std::priority_queue<T, std::vector<T>, std::greater<T> > q;

    public:
        MinHeap() = default;

        const T &peek() const override { return q.top(); }

        void pop() override { q.pop(); }

        void push(const T &x) override { q.push(x); }

        size_t size() override { return q.size(); }
    };
}
