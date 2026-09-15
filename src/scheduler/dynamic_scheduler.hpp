#pragma once

#include "scheduler/ischeduler.hpp"

/// Dynamic — shared queue where workers pull next task when idle.
/// Uses mutex + shared atomic index. Differs from FCFS (static assignment)
/// by allowing load balancing when task costs vary.
class DynamicScheduler : public IScheduler {
public:
    std::string name() const override { return "Dynamic"; }
    void execute(std::vector<double>& array,
                 const std::vector<Task>& tasks,
                 size_t worker_count,
                 double a, double b, double c,
                 int repeat_count) override;
};
