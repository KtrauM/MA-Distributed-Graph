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
#include <kamping/measurements/timer.hpp>

namespace distributed {

template <typename T> class DistributedSet {
public:
  DistributedSet(kamping::Communicator<> const &comm)
      : _comm(comm) {}


  uint32_t max_send_buffer_size = 0;
  
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
    std::vector<int> send_counts(_comm.size(), 0);
    for (const T &element: _local_data) {
      int target = mapping(element);
      ++send_counts[target];
    }
    send_counts[_comm.rank()] = 0; // don't send to self

    std::vector<size_t> offsets(_comm.size(), 0);
    uint32_t send_buffer_size = 0;
    for (int rank = 0; rank < _comm.size(); ++rank) {
      offsets[rank] = send_buffer_size;
      send_buffer_size += send_counts[rank];
    }

    std::vector<T> send_buffer(send_buffer_size);
    for (const T &element: _local_data) {
      int target = mapping(element);
      if (target != _comm.rank()) {
        send_buffer[offsets[target]] = std::move(element);
        ++offsets[target];
      }
    }

    max_send_buffer_size = std::max(max_send_buffer_size, send_buffer_size);

    std::vector<T> recv_buffer;
    kamping::measurements::timer().start("frontier_exchange_alltoall");
    _comm.alltoallv(kamping::send_buf(send_buffer), kamping::send_counts(send_counts), kamping::recv_buf<kamping::BufferResizePolicy::resize_to_fit>(recv_buffer));
    kamping::measurements::timer().stop_and_add();

    filter([this,mapping](const T& element) { return mapping(element) != _comm.rank(); });
    _local_data.reserve(_local_data.size() + recv_buffer.size());
    _local_data.insert(_local_data.end(), std::make_move_iterator(recv_buffer.begin()), std::make_move_iterator(recv_buffer.end()));
    // print local_data
    // std::cout << "Local data: ";
    // for (int i = 0; i < _local_data.size(); ++i) {
    //   std::cout << _local_data[i] << " ";
    // }
    // std::cout << "\n";
  }

// void redistribute(std::function<int(const T&)> mapping) {
//     std::unordered_map<int, std::vector<T>> outgoing_data;
//     for (const T &element: _local_data) {
//       int target = mapping(element);
//       outgoing_data[target].push_back(element);
//     }
//     // std::cout << "Comm size " << _comm.size() << " " << ", rank: " << _comm.rank() << ", outgoing data:\n";
//     // for (int rank = 0; rank < _comm.size(); rank++) {
//     //   std::cout << "Target PE rank: " << rank << "\n";
//     //   for (auto x: outgoing_data[rank]) {
//     //     std::cout << x << " ";
//     //   }

//     // }
//     // std::cout << "\n";

//     std::vector<int> send_counts(_comm.size(), 0);
//     std::vector<T> send_buffer;
//     uint32_t send_buffer_size = 0;
//     for (int rank = 0; rank < _comm.size(); ++rank) {
//       send_counts[rank] = outgoing_data[rank].size();
//       send_buffer_size += send_counts[rank];
//     }

//     send_buffer.reserve(send_buffer_size);
//     for (int rank = 0; rank < _comm.size(); ++rank) {
//       send_buffer.insert(send_buffer.end(), outgoing_data[rank].begin(), outgoing_data[rank].end());
//     }

//     //print self count
//     std::cout << "Self send count: " << send_counts[_comm.rank()] << "\n";
//     //print send_counts
//     std::cout << "Send counts: ";
//     for (int rank = 0; rank < _comm.size(); ++rank) {
//       std::cout << send_counts[rank] << " ";
//     }
//     std::cout << "\n";
//     // print send_buffer
//     std::cout << "Send buffer: ";
//     for (int i = 0; i < send_buffer_size; ++i) {
//       std::cout << send_buffer[i] << " ";
//     }
//     std::cout << "\n";

//     _local_data.clear();
//     kamping::measurements::timer().start("frontier_exchange_alltoall");
//     _comm.alltoallv(kamping::send_buf(send_buffer), kamping::send_counts(send_counts), kamping::recv_buf<kamping::BufferResizePolicy::resize_to_fit>(_local_data));
//     kamping::measurements::timer().stop_and_add();
//     // print local_data
//     std::cout << "Local data: ";
//     for (int i = 0; i < _local_data.size(); ++i) {
//       std::cout << _local_data[i] << " ";
//     }
//     std::cout << "\n";
//   }
 
  const std::vector<T> &local_data() const { return _local_data; }

  const void updateCommunicator(kamping::Communicator<> &new_comm) {
    _comm = new_comm;
  }

protected:
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