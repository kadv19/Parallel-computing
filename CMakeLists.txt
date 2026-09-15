cmake_minimum_required(VERSION 3.14)
project(ParallelImageScheduler VERSION 0.1.0 LANGUAGES CXX)

# Work around stale GIT_EXEC_PATH in some environments (e.g. custom git install
# that no longer exists). Without this, FetchContent git clone fails.
if(DEFINED ENV{GIT_EXEC_PATH} AND NOT EXISTS "$ENV{GIT_EXEC_PATH}/git-submodule")
  unset(ENV{GIT_EXEC_PATH})
endif()

# ---------------------------------------------------------------------------
# Global settings
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Warnings — must build with no warnings on default flags
add_compile_options(-Wall -Wextra -Wpedantic)

# Threading support
find_package(Threads REQUIRED)

# ---------------------------------------------------------------------------
# Libraries
# ---------------------------------------------------------------------------
add_library(config_loader
    src/config/config_loader.cpp
)
target_include_directories(config_loader PUBLIC src/config)
target_include_directories(config_loader PUBLIC src)

add_library(array_generator
    src/data/array_generator.cpp
)
target_include_directories(array_generator PUBLIC src)

add_library(task_generator
    src/core/task_generator.cpp
)
target_include_directories(task_generator PUBLIC src)

add_library(access_pattern
    src/core/access_pattern.cpp
)
target_include_directories(access_pattern PUBLIC src)

add_library(numeric_kernel
    src/kernel/numeric_kernel.cpp
)
target_include_directories(numeric_kernel PUBLIC src)
target_link_libraries(numeric_kernel PUBLIC access_pattern task_generator)

add_library(thread_pool
    src/engine/thread_pool.cpp
)
target_include_directories(thread_pool PUBLIC src)
target_link_libraries(thread_pool PUBLIC Threads::Threads)

add_library(schedulers
    src/scheduler/fcfs_scheduler.cpp
    src/scheduler/sjf_scheduler.cpp
    src/scheduler/round_robin_scheduler.cpp
    src/scheduler/dynamic_scheduler.cpp
    src/scheduler/ai_scheduler.cpp
)
target_include_directories(schedulers PUBLIC src)
target_link_libraries(schedulers PUBLIC thread_pool numeric_kernel task_generator ml_model metrics)

add_library(metrics
    src/benchmark/metrics.cpp
)
target_include_directories(metrics PUBLIC src)
target_link_libraries(metrics PUBLIC task_generator)

add_library(ml_model
    src/ml/model.cpp
)
target_include_directories(ml_model PUBLIC src)
target_link_libraries(ml_model PUBLIC metrics task_generator)

# ---------------------------------------------------------------------------
# Main executable
# ---------------------------------------------------------------------------
add_executable(main
    src/main.cpp
)
target_link_libraries(main PRIVATE config_loader array_generator task_generator access_pattern numeric_kernel thread_pool schedulers metrics ml_model)
target_include_directories(main PRIVATE src)

# Benchmark executable
add_executable(benchmark
    src/benchmark/benchmark.cpp
)
target_link_libraries(benchmark PRIVATE config_loader array_generator task_generator access_pattern numeric_kernel thread_pool schedulers metrics ml_model)
target_include_directories(benchmark PRIVATE src)

add_executable(task_benchmark
    src/benchmark/task_benchmark.cpp
)
target_link_libraries(task_benchmark PRIVATE array_generator task_generator access_pattern numeric_kernel metrics ml_model)
target_include_directories(task_benchmark PRIVATE src)

# ---------------------------------------------------------------------------
# Testing — Catch2 via FetchContent
# ---------------------------------------------------------------------------
include(FetchContent)

FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.7.1
)
FetchContent_MakeAvailable(Catch2)

enable_testing()

add_executable(tests
    tests/cpp/test_config_loader.cpp
    tests/cpp/test_task_generator.cpp
    tests/cpp/test_access_pattern.cpp
    tests/cpp/test_numeric_kernel.cpp
    tests/cpp/test_schedulers.cpp
    tests/cpp/test_benchmark.cpp
    tests/cpp/test_ai_scheduler.cpp
)
target_link_libraries(tests PRIVATE config_loader array_generator task_generator access_pattern numeric_kernel thread_pool schedulers metrics ml_model Catch2::Catch2WithMain)
target_include_directories(tests PRIVATE src)

include(Catch)
catch_discover_tests(tests WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
