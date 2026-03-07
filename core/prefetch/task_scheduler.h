// Copyright (c) EfficientMoE.
// SPDX-License-Identifier: Apache-2.0

// EfficientMoE Team

#pragma once

#include <deque>
#include <iostream>
#include <list>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include "base/noncopyable.h"
#include "common/pytorch.h"
#include "model/model_topology.h"

#define SKIP_TO_NEXT_ITERATION                                \
  std::this_thread::sleep_for(std::chrono::microseconds(10)); \
  continue;

#define NUM_PRIORITY 20UL

struct Task {
  bool on_demand = false;
  NodePtr node;
  std::vector<NodePtr> remove_nodes;
  std::uint32_t priority;
  std::uint64_t request_id;
  torch::Device src_device = DISK_DEVICE;
  torch::Device dst_device = DISK_DEVICE;
  cudaStream_t stream = nullptr;

  bool remove_layer = false;

  std::string DebugString() {
    std::stringstream ss;
    ss << "Task: node: " << node->str() << ", on_demand: " << on_demand
       << ", priority: " << priority << "[" << src_device.str() << "->"
       << dst_device.str() << "]";
    return ss.str();
  }
};
typedef std::shared_ptr<Task> TaskPtr;

class ArcherTaskPool : public base::noncopyable {
 public:
  void StartExec(const std::uint64_t& request_id, const NodePtr& node);
  void FetchExec(const std::uint64_t& request_id, const NodePtr& node);
  void StopExec(const std::uint64_t& request_id, const NodePtr& node);
  void EnqueueTask(const TaskPtr& task);

  void ClearQueue() {
    std::lock_guard<std::mutex> lock(unified_mutex_);
    for (std::uint32_t priority = 1; priority < NUM_PRIORITY; priority++) {
      unified_queue_[priority].clear();
    }
  }

  bool RemoveCachedSparseNode(const NodePtr& node, int device_id = -1);
  bool RemoveCachedDenseNode(const NodePtr& node);
  // void RemoveCachedNode(const NodePtr& node);

  void ReplaceCacheCandidates(const NodePtrList& candidates) {
    std::lock_guard<std::mutex> lock(unified_mutex_);
    {
      std::lock_guard<std::mutex> lock(this->candidates_mutex_);
      candidates_.clear();
      for (auto& node : candidates) {
        candidates_.insert(node);
      }
    }

    for (std::uint32_t priority = 1; priority < NUM_PRIORITY; priority++) {
      unified_queue_[priority].clear();
    }
  }

  DELETE_COPY_AND_ASSIGN(ArcherTaskPool);
  STATIC_GET_INSTANCE(ArcherTaskPool);

  ArcherTaskPool();
  ~ArcherTaskPool() {
    std::cout << "ArcherTaskPool destructor" << std::endl;

    // Clear all pending tasks so threads don't start new CUDA operations.
    ClearQueue();

    // Signal all threads to exit.
    main_thread_stop_flag_.store(true);

    // Join threads with a timeout — if a thread is stuck in a CUDA call
    // (e.g. on unified memory systems during process teardown), detach it
    // rather than hanging the process indefinitely.
    for (auto& thread_list : exec_threads_) {
      for (auto& thread : thread_list) {
        if (!thread.joinable()) continue;

        // Use a helper thread to perform the join so we can impose a timeout.
        std::atomic<bool> join_done{false};
        std::thread joiner([&thread, &join_done]() {
          thread.join();
          join_done.store(true);
        });
        joiner.detach();

        auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::seconds(3);
        while (!join_done.load() &&
               std::chrono::steady_clock::now() < deadline) {
          std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        if (!join_done.load()) {
          // Thread is stuck — let the OS clean up on process exit.
          std::cerr << "ArcherTaskPool: thread did not exit in time, detaching"
                    << std::endl;
        }
      }
    }
  }

 private:
  void GPUThreadFunc(int gpu_id, int thread_id);

  void SetNodeDevice(const TaskPtr& task);

  std::string DebugString(const std::vector<std::deque<TaskPtr>>& queue);

 private:
  std::vector<std::deque<TaskPtr>> unified_queue_;  // For ordered prefetch
  std::vector<std::vector<std::uint32_t>> gpu_min_priority_;
  std::unordered_map<std::uint64_t, TaskPtr> exec_queue_;
  std::mutex exec_mutex_;
  std::mutex unified_mutex_;
  std::mutex candidates_mutex_;

  std::vector<std::list<std::thread>> exec_threads_;

  std::unordered_set<NodePtr> candidates_;

  std::atomic<bool> main_thread_stop_flag_;
};

extern std::unique_ptr<ArcherTaskPool> kTaskPool;
