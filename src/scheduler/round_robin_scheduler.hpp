#pragma once

#include "scheduler/ischeduler.hpp"

/// Round Robin — simplified academic cyclic assignment (NOT OS preemptive).
/// Task i -> worker (i % W). No time quantum, no preemption, no context switching.
/// A worker simply executes its assigned subset.
class RoundRobinScheduler : public IScheduler {
public:
    std::string name() const override { return "RoundRobin"; }
    void execute(std::vector<double>& array,
                 const std::vector<Task>& tasks,
                 size_t worker_count,
                 double a, double b, double c,
                 int repeat_count) override;
};
