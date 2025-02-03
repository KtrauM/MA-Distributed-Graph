#pragma once

#include "distribution_strategy.hpp"
#include <algorithm>
#include <cassert>
#include <kamping/collectives/alltoall.hpp>
#include <kamping/collectives/gather.hpp>
#include <kamping/communicator.hpp>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace distributed {

template <typename T> struct ArrayUpdate {
  size_t global_index;
  T value;
};

template <typename T> class DistributedArray {
public:
  DistributedArray(std::unique_ptr<DistributionStrategy> strategy, kamping::Communicator<> const &comm)
      : _strategy(std::move(strategy)), _rank(comm.rank()), _num_ranks(comm.size()), _local_data(strategy->local_size(_rank), T{}) {}

  void set(size_t global_index, T value) {
    int owner = _strategy->owner(global_index);
    // data is local
    if (owner == _rank) {
      std::cout << "Local data encountered for global index: " << global_index << ", value: " << value << ", rank: " << _rank << ", owner: " << owner << '\n'; 
      size_t local_index = _strategy->to_local_index(_rank, global_index);
      _local_data[local_index] = value;
      std::cout << "Local index: " << local_index << ", data: " << _local_data[local_index] << '\n';
      return;
    }
    // data is non-local
    _outgoing_data[owner].push_back({global_index, value});
    std::cout << "Outgoing data from " << _rank << '\n';
    for (const auto &[owner, updates]: _outgoing_data) {
      for (const auto &update: updates) {
        std::cout << "Owner: " << owner << ", global_index: " << update.global_index << ", value: " << update.value << '\n';
      }
    }
  }

  T get(size_t global_index) const {
    if (_rank != _strategy->owner(global_index)) {
      throw std::out_of_range("Accessed global_index is not present in local data.");
    }
    size_t local_index = _strategy->to_local_index(_rank, global_index);
    return _local_data[local_index];
  }

  void exchange() {
    std::vector<int> send_counts(_num_ranks, 0);
    std::vector<ArrayUpdate<T>> send_buffer;
    std::vector<ArrayUpdate<T>> recv_buffer;

    for (int rank = 0; rank < _num_ranks; ++rank) {
        const auto& updates = _outgoing_data[rank];
        send_counts[rank] = updates.size();
        send_buffer.insert(send_buffer.end(), updates.begin(), updates.end());
    }

    std::cout << "Rank: " << _rank << "send_counts";

    _comm.alltoallv(kamping::send_buf(send_buffer), kamping::send_counts(send_counts), kamping::recv_buf<kamping::BufferResizePolicy::grow_only>(recv_buffer));
    _comm.barrier();
    std::cout << "Passed alltoallv on PE: " << _rank << "\n"; 

    for (const auto &update : recv_buffer) {
      std::cout << "Update on rank:" << _rank << ", global_index: " << update.global_index << ", value: " << update.value << '\n';
      size_t local_index = _strategy->to_local_index(_rank, update.global_index);
      std::cout << "local_index: " << local_index << '\n';
      _local_data[local_index] = update.value;
    }

    _outgoing_data.clear();
  }

  std::vector<T> gather(int root) const {
    std::vector<size_t> indices, all_indices;
    std::vector<T> data, all_data, global_array;

    for (size_t local_index = 0; local_index < _local_data.size(); ++local_index) {
      indices.push_back(_strategy->to_global_index(_rank, local_index));
      data.push_back(_local_data[local_index]);
    }

    // can also calculate recv_counts to eliminate overhead here
    // as all workers know the strategy and can calculate how many elements are in each worker
    _comm.gatherv(kamping::send_buf(indices), kamping::recv_buf<kamping::BufferResizePolicy::grow_only>(all_indices));
    _comm.gatherv(kamping::send_buf(data), kamping::recv_buf<kamping::BufferResizePolicy::grow_only>(all_data));

    global_array.resize(all_data.size());
    for (size_t i = 0; i < all_indices.size(); ++i) {
      size_t global_index = all_indices[i];
      global_array[global_index] = all_data[i];
    }

    return global_array;
  }

  void print_local() const {
    std::cout<< "PE: " << _rank << ": ";
    for (auto const& x: _local_data) {
      std::cout << x << ' ';
    }
    std::cout << '\n';
  }

private:
  int _rank;
  int _num_ranks;
  std::vector<T> _local_data;
  std::unordered_map<int, std::vector<ArrayUpdate<T>>> _outgoing_data;
  std::unique_ptr<DistributionStrategy> _strategy;
  kamping::Communicator<> _comm;
};

} // namespace distributed