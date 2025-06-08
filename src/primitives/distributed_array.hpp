#pragma once

#include "distribution_strategy.hpp"
#include <algorithm>
#include <kamping/collectives/alltoall.hpp>
#include <kamping/collectives/gather.hpp>
#include <kamping/communicator.hpp>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <vector>
#include <kamping/measurements/timer.hpp>

namespace distributed {

enum class OperationType { IDENTITY, MIN, MAX };

template <typename T> std::function<T(T, T)> get_operation(OperationType operation_type) {
  switch (operation_type) {
  case OperationType::IDENTITY:
    return [](T x, T y) { return y; };
  case OperationType::MIN:
    return [](T x, T y) { return std::min(x, y); };
  case OperationType::MAX:
    return [](T x, T y) { return std::max(x, y); };
  default:
    return [](T x, T y) { return y; };
  }
}

template <typename T> struct ArrayUpdate {
  size_t global_index;
  T value;
  OperationType operation_type;

  ArrayUpdate() : global_index(0), value{}, operation_type(OperationType::IDENTITY) {}
  ArrayUpdate(size_t global_index, T value) : global_index(global_index), value(value), operation_type(OperationType::IDENTITY) {}
  ArrayUpdate(size_t global_index, T value, OperationType operation_type) : global_index(global_index), value(value), operation_type(operation_type) {}
};

template <typename T> class DistributedArray {
public:
  DistributedArray(std::shared_ptr<DistributionStrategy> strategy, kamping::Communicator<> const &comm)
      : _strategy(std::move(strategy)), _comm(comm), _local_data(strategy->local_size(comm.rank()), T{}) {}

  DistributedArray(std::shared_ptr<DistributionStrategy> strategy, kamping::Communicator<> const &comm, T initialization_value)
      : _strategy(std::move(strategy)), _comm(comm), _local_data(strategy->local_size(comm.rank()), initialization_value) {}

  void initialize_local(std::vector<T> elements, int rank) {
    // TODO: there is a bug here
    if (_local_data.size() != elements.size()) {
      // std::cout << "PE " << _comm.rank() << "/" << _comm.size() << ": local_data.size(): " << _local_data.size() << ", elements.size(): " << elements.size() << std::endl;
      throw std::logic_error("Mismatch between expected _local_data and input data size");
    }
    _local_data = std::move(elements);
  }

  void set(size_t global_index, T value) {
    int owner = _strategy->owner(global_index);
    // Data is local
    if (owner == _comm.rank()) {
      size_t local_index = _strategy->to_local_index(_comm.rank(), global_index);
      _local_data[local_index] = value;
      return;
    }
    // Data is non-local
    _outgoing_data[owner].push_back({global_index, value});
  }

  void set(size_t global_index, T value, OperationType operation_type) {
    int owner = _strategy->owner(global_index);
    // Data is local
    if (owner == _comm.rank()) {
      size_t local_index = _strategy->to_local_index(_comm.rank(), global_index);
      std::function<T(T, T)> operation = get_operation<T>(operation_type);
      // std::cout << "_local data vs update avlue " << _local_data[local_index] << " " << value << "\n";
      _local_data[local_index] = operation(_local_data[local_index], value);
      return;
    }
    // Data is non-local
    std::cout << "Set delegated to exchange as the index is not local, update value: " << value << "\n";
    _outgoing_data[owner].push_back({global_index, value, operation_type});
  }

  T get(size_t global_index) const {
    if (_comm.rank() != _strategy->owner(global_index)) {
      throw std::out_of_range("Accessed global_index " + std::to_string(global_index) + " is not present in local data of PE " + std::to_string(_comm.rank()) + ".");
    }
    size_t local_index = _strategy->to_local_index(_comm.rank(), global_index);
    return _local_data[local_index];
  }

  void exchange(kamping::Communicator<> &sub_comm) {
    std::vector<int> send_counts(sub_comm.size(), 0);
    std::vector<ArrayUpdate<T>> send_buffer;
    std::vector<ArrayUpdate<T>> recv_buffer;

    uint32_t send_buffer_size = 0;
    for (int rank = 0; rank < sub_comm.size(); ++rank) {
        const auto& updates = _outgoing_data[rank];
        if (updates.size() > 0) {
            std::cout << "PE " << _comm.rank() << " sending " << updates.size() << " updates to PE " << rank << "\n";
        }
        send_counts[rank] = updates.size();
        send_buffer_size += send_counts[rank];
    }
    
    send_buffer.reserve(send_buffer_size);
    for (int rank = 0; rank < sub_comm.size(); ++rank) {
      const auto& updates = _outgoing_data[rank];
      send_buffer.insert(send_buffer.end(), updates.begin(), updates.end());
    }

    std::cout << "PE " << _comm.rank() << " sending " << send_buffer.size() << " updates to all PEs\n";
    kamping::measurements::timer().start("distance_alltoallv");
    sub_comm.alltoallv(kamping::send_buf(send_buffer), kamping::send_counts(send_counts), kamping::recv_buf<kamping::BufferResizePolicy::grow_only>(recv_buffer));
    kamping::measurements::timer().stop_and_add();

    for (const auto &update : recv_buffer) {
      size_t local_index = _strategy->to_local_index(_comm.rank(), update.global_index);
      std::function<T(T, T)> operation = get_operation<T>(update.operation_type);
      // std::cout << "exchange _local data vs update avlue " << _local_data[local_index] << " " << update.value << "\n";
      _local_data[local_index] = operation(_local_data[local_index], update.value);
    }

    _outgoing_data.clear();
  }

  std::vector<T> gather(int root) const {
    std::vector<size_t> indices, all_indices;
    std::vector<T> data, all_data, global_array;

    for (size_t local_index = 0; local_index < _local_data.size(); ++local_index) {
      indices.push_back(_strategy->to_global_index(_comm.rank(), local_index));
      data.push_back(_local_data[local_index]);
    }

    // Can also calculate recv_counts to eliminate overhead here as all workers know the strategy,
    // so we can calculate how many elements are in each worker
    _comm.gatherv(kamping::send_buf(indices), kamping::recv_buf<kamping::BufferResizePolicy::grow_only>(all_indices));
    _comm.gatherv(kamping::send_buf(data), kamping::recv_buf<kamping::BufferResizePolicy::grow_only>(all_data));

    global_array.resize(all_data.size());
    for (size_t i = 0; i < all_indices.size(); ++i) {
      size_t global_index = all_indices[i];
      global_array[global_index] = all_data[i];
    }

    return global_array;
  }

  std::vector<T> &local_data() { return _local_data; }

  void print_local() const {
    std::cout<< "PE: " << _comm.rank() << ": ";
    for (auto const& x: _local_data) {
      std::cout << x << ' ';
    }
    std::cout << '\n';
  }

  void print_local_vertex() const {
    std::cout<< "PE: " << _comm.rank() << ": ";
    for (auto const& x: _local_data) {
      std::cout << x.edge_start_index << ',' << x.edge_end_index << ' ';
    }
    std::cout << '\n';
  }

private:
  std::vector<T> _local_data;
  std::unordered_map<int, std::vector<ArrayUpdate<T>>> _outgoing_data;
  std::shared_ptr<DistributionStrategy> _strategy;
  kamping::Communicator<> _comm;
};

} // namespace distributed