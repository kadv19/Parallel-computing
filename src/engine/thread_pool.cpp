#include "engine/thread_pool.hpp"

#include <stdexcept>

ThreadPool::ThreadPool(size_t num_threads) {
    if (num_threads == 0) {
        throw std::runtime_error("ThreadPool requires at least 1 thread");
    }
    workers_.reserve(num_threads);
    for (size_t i = 0; i < num_threads; ++i) {
        workers_.emplace_back([this] { worker_loop(); });
    }
}

ThreadPool::~ThreadPool() {
    shutdown();
}

void ThreadPool::enqueue(std::function<void()> task) {
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        if (stop_) {
            throw std::runtime_error("enqueue on stopped ThreadPool");
        }
        tasks_.push(std::move(task));
    }
    condition_.notify_one();
}

void ThreadPool::wait() {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    completed_.wait(lock, [this] { return tasks_.empty() && active_tasks_ == 0; });
}

void ThreadPool::shutdown() {
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        if (stop_) return;
        stop_ = true;
    }
    condition_.notify_all();
    for (auto& w : workers_) {
        if (w.joinable()) w.join();
    }
    workers_.clear();
}

void ThreadPool::worker_loop() {
    while (true) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            condition_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
            if (stop_ && tasks_.empty()) {
                return;
            }
            task = std::move(tasks_.front());
            tasks_.pop();
            ++active_tasks_;
        }

        task();

        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            --active_tasks_;
            if (tasks_.empty() && active_tasks_ == 0) {
                completed_.notify_all();
            }
        }
    }
}
