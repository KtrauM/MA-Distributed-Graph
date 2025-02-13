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

template <typename T> class DistributedSet {
public:
  DistributedSet(std::shared_ptr<DistributionStrategy> strategy, kamping::Communicator<> const &comm)
      : _strategy(std::move(strategy)), _comm(comm), _rank(comm.rank()), _num_ranks(comm.size()), _local_data(strategy->local_size(_rank), T{}) {}

  void insert(T element) {
    int owner = _strategy->owner(element);
    if (owner == _rank) {
      auto pos = std::lower_bound(_local_data.begin(), _local_data.end(), element);
      _local_data.insert(pos, element);
      return;
    }
    _outgoing_data[owner].push_back(element);
  }

  void insert(std::vector<T> elements) {
    _local_data.reserve(_local_data.size() + elements.size());
    for (const auto &element : elements) {
      insert(element);
    }
  }

  void insert_immediate(T element) {
    int owner = _strategy->owner(element);
    if (owner == _rank) {
      insert(element);
      return;
    }
    // TODO: point to point using Isend but when to Irecv?
    throw std::logic_error("insert_immediate is not implemented yet!");
  }

  void deduplicate() {
    auto last = std::unique(_local_data.begin(), _local_data.end());
    _local_data.erase(last, _local_data.end());
  }

  void exchange() {
    std::vector<int> send_counts(_num_ranks, 0);
    std::vector<T> send_buffer;
    std::vector<T> recv_buffer;

    for (int rank = 0; rank < _num_ranks; ++rank) {
      const auto &updates = _outgoing_data[rank];
      send_counts[rank] = updates.size();
      send_buffer.insert(send_buffer.end(), updates.begin(), updates.end());
    }

    _comm.alltoallv(kamping::send_buf(send_buffer), kamping::send_counts(send_counts),
                    kamping::recv_buf<kamping::BufferResizePolicy::grow_only>(recv_buffer));

    std::vector<T> merged(_local_data.size() + recv_buffer.size());
    std::sort(recv_buffer.begin(), recv_buffer.end());
    std::merge(_local_data.begin(), _local_data.end(), recv_buffer.begin(), recv_buffer.end(), merged.begin());
    _local_data = std::move(merged);

    _outgoing_data.clear();
  }

  const std::vector<T> &local_data() const { return _local_data; }

private:
  int _rank;
  int _num_ranks;
  std::vector<T> _local_data; // always sorted
  std::unordered_map<int, std::vector<T>> _outgoing_data;
  std::shared_ptr<DistributionStrategy> _strategy;
  kamping::Communicator<> _comm;
};

} // namespace distributed