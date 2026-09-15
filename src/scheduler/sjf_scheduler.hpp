#pragma once

#include "scheduler/ischeduler.hpp"

/// SJF — Shortest Job First
/// Uses a-priori estimated cost: estimated_cost = element_count * repeat_count.
/// If needed, could include access-pattern penalty, but phase 3 keeps it simple.
/// Sorts tasks by ascending estimated cost, then statically partitions like FCFS.
/// Does NOT use measured future execution time (would leak prediction target).
class SjfScheduler : public IScheduler {
public:
    std::string name() const override { return "SJF"; }
    void execute(std::vector<double>& array,
                 const std::vector<Task>& tasks,
                 size_t worker_count,
                 double a, double b, double c,
                 int repeat_count) override;
};
