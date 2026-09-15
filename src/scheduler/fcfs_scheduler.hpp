#pragma once

#include "scheduler/ischeduler.hpp"

/// FCFS — First-Come, First-Served
/// Static queue-order assignment. For W workers and N tasks, preserve task order
/// and distribute contiguous blocks: worker 0 gets first block, worker 1 gets
/// next, etc. Assignment is computed BEFORE execution (static), unlike Dynamic.
/// Example N=6 W=3: A->[0,1] B->[2,3] C->[4,5]
class FcfsScheduler : public IScheduler {
public:
    std::string name() const override { return "FCFS"; }
    void execute(std::vector<double>& array,
                 const std::vector<Task>& tasks,
                 size_t worker_count,
                 double a, double b, double c,
                 int repeat_count) override;
};
