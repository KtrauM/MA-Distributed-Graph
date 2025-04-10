#pragma once

#include <algorithm>
#include <cassert>
#include <kamping/collectives/alltoall.hpp>
#include <kamping/collectives/gather.hpp>
#include <kamping/communicator.hpp>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <vector>
#include <functional>

namespace distributed {

template <typename T> class DistributedSet {
public:
  DistributedSet(kamping::Communicator<> const &comm)
      : _comm(comm), _rank(comm.rank()), _num_ranks(comm.size()) {}

  void insert(T element) {
    _local_data.push_back(element);
  }
  
  void insert(std::vector<T> elements) {
    _local_data.insert(_local_data.end(), elements.begin(), elements.end());
  }
  
  void insert(const distributed::DistributedSet<T> &other_set) {
    insert(other_set.local_data());
  }
  
  void remove(T element) {
    auto pos = std::find(_local_data.begin(), _local_data.end(), element);
    if (pos != _local_data.end()) {
      _local_data.erase(pos);
    }
  }

  bool contains(T element) {
    auto pos = std::find(_local_data.begin(), _local_data.end(), element);
    return pos != _local_data.end();
  }

  void deduplicate() {
    std::sort(_local_data.begin(), _local_data.end());
    auto last = std::unique(_local_data.begin(), _local_data.end());
    _local_data.erase(last, _local_data.end());
  }


  void filter(std::function<bool(const T&)> func) {
    auto last = std::remove_if(_local_data.begin(), _local_data.end(), func);
    _local_data.erase(last, _local_data.end());
  }

  void redistribute(std::function<int(const T&)> mapping) {
    std::unordered_map<int, std::vector<T>> outgoing_data;
    for (const T &element: _local_data) {
      int target = mapping(element);
      outgoing_data[target].push_back(element);
    }

    std::vector<int> send_counts(_num_ranks, 0);
    std::vector<T> send_buffer;

    for (int rank = 0; rank < _num_ranks; ++rank) {
      send_counts[rank] = outgoing_data[rank].size();
      send_buffer.insert(send_buffer.end(), outgoing_data[rank].begin(), outgoing_data[rank].end());
    }

    _local_data.clear();
    std::cout << "Distributed set alltoallv, comm size " << _comm.size() << " send buffer size " << send_buffer.size() << " receive buffer local data size: " << _local_data.size() << "\n"; 
    if (_comm.size() != 1) {
      _comm.alltoallv(kamping::send_buf(send_buffer), kamping::send_counts(send_counts), kamping::recv_buf<kamping::BufferResizePolicy::resize_to_fit>(_local_data));
    }
  }
 
  const std::vector<T> &local_data() const { return _local_data; }

  const void updateCommunicator(kamping::Communicator<> &new_comm) {
    _comm = new_comm;
  }

protected:
  int _rank;
  int _num_ranks;
  std::vector<T> _local_data;
  kamping::Communicator<> _comm;
};


// Invariant: _local_data is always sorted.
template <typename T> class SortedDistributedSet : public DistributedSet<T> {
  public:
    using DistributedSet<T>::DistributedSet;
    using DistributedSet<T>::_local_data;

    void insert(T element) {
      auto pos = std::lower_bound(_local_data.begin(), _local_data.end(), element);
      _local_data.insert(pos, element);
    }

    void insert(std::vector<T> elements) {
      std::sort(elements.begin(), elements.end());
      for (const T &element: elements) {
        insert(element);
      }
    }

    void insert(const distributed::DistributedSet<T> &other_set) {
      insert(other_set.local_data());
    }

    void remove(T element) {
      auto pos = std::lower_bound(_local_data.begin(), _local_data.end(), element);
      if (pos != _local_data.end() && *pos == element) {
        _local_data.erase(pos);
      }
    }

    bool contains(T element) {
      auto pos = std::lower_bound(_local_data.begin(), _local_data.end(), element);
      return pos != _local_data.end() && *pos == element;
    }
  
    void deduplicate() {
      auto last = std::unique(_local_data.begin(), _local_data.end());
      _local_data.erase(last, _local_data.end());
    }

    const std::vector<T> &local_data() const { return _local_data; }
};
} // namespace distributed