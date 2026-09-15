#pragma once

#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

/// Minimal fixed-size thread pool.
/// Uses std::thread, mutex, condition_variable, and a task queue.
/// No busy waiting, no detached threads, clean shutdown.
class ThreadPool {
public:
    explicit ThreadPool(size_t num_threads);
    ~ThreadPool();

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    /// Submit a job for execution.
    void enqueue(std::function<void()> task);

    /// Wait until all enqueued tasks have completed.
    void wait();

    /// Shut down the pool. No further tasks should be enqueued after this.
    void shutdown();

private:
    void worker_loop();

    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;

    std::mutex queue_mutex_;
    std::condition_variable condition_;
    std::condition_variable completed_;
    bool stop_{false};
    size_t active_tasks_{0};
};
