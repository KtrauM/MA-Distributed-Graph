#pragma once

#include <kamping/collectives/allreduce.hpp>
#include <set>
#include "../../primitives/distributed_array.hpp"
#include "../../primitives/distributed_set.hpp"
#include "../common/graph.hpp"
// #include <kamping/measurements/timer.hpp>

class SetBasedDistributedFrontier {
public:
  SetBasedDistributedFrontier(kamping::Communicator<> comm, std::vector<VertexId> initial_frontier)
      : _frontier(distributed::DistributedSet<VertexId>(comm)) {
    if (comm.rank() == 0) {
      _frontier.insert(initial_frontier);
    }
  }

  void add(VertexId vertex) { _frontier.insert(vertex); }

  void exchange(std::function<int(const VertexId)> mapping) {
    _frontier.redistribute(mapping);
  }

  void deduplicate() {
    _frontier.deduplicate();
  }
  
  void filter(std::function<bool(const VertexId)> pred) {
    _frontier.filter(pred); 
  }

  const std::vector<VertexId> &local_frontier() {
    return _frontier.local_data();
  }

  const std::string toString() const {
    std::string str = "Local data: ";
    for (const auto &x: _frontier.local_data()) {
      str += std::to_string(x);
    }
    return str;
  }

  const void updateCommunicator(kamping::Communicator<> &new_comm) {
    _frontier.updateCommunicator(new_comm);
  }

private:
  distributed::DistributedSet<VertexId> _frontier;
};


class DistributedBFS {
public:
  DistributedBFS(std::shared_ptr<DistributedCSRGraph> graph, kamping::Communicator<> const &comm, std::vector<VertexId> source_vertices)
      : _graph(std::move(graph)), _initial_rank(comm.rank()), _comm(comm), _frontier(SetBasedDistributedFrontier(_comm, source_vertices)), _distances(_graph->vertex_dist, comm, std::numeric_limits<uint64_t>::max()) {}

  void run() {
    // kamping::measurements::timer().start("bfs_total");
    // std::cout << "Frontier print " << _frontier.toString() << " on PE " << _comm.rank() << "\n";
    auto current_frontier = _frontier.local_frontier(); // contains ids of vertices
    // std::cout << "Frontier print after accessing local_frontier" << _frontier.toString() << " on PE " << _comm.rank() << "\n";
    bool local_active = !current_frontier.empty();
    bool global_active = true;
    // kamping::measurements::timer().start("bfs_while_loop");

    while (global_active) {
      // kamping::measurements::timer().synchronize_and_start("bfs_iteration");
      // std::cout << "Current frontier size " << current_frontier.size() << "\n";
      for (VertexId vertex : current_frontier) {
        // std::cout << "Setting vertex " << vertex << " distance to " << current_distance << "\n";
        _distances.set(vertex, current_distance, distributed::OperationType::MIN);
        if (_graph->vertex_dist->owner(vertex) != _initial_rank) {
          throw std::logic_error("Vertex " + std::to_string(vertex) + " belongs to another worker: " +
                                 std::to_string(_graph->vertex_dist->owner(vertex)) + ", but was found in worker " + std::to_string(_initial_rank));
        }

        auto [edge_index_start, edge_index_end] = _graph->vertex_array.get(vertex);
        for (EdgeId i = edge_index_start; i < edge_index_end; ++i) {
          VertexId neighbor = _graph->edge_array.get(i);
          _frontier.add(neighbor);
        }
      }
      // kamping::measurements::timer().stop();
      // kamping::measurements::timer().synchronize_and_start("bfs_exchange");
      _frontier.exchange([this](const size_t vertex_id) { return _graph->vertex_dist->owner(vertex_id); });
      // std::cout << "Distance exchange\n";
      _distances.exchange(_comm);
      // kamping::measurements::timer().stop();

      // kamping::measurements::timer().synchronize_and_start("bfs_local_frontier");
      _frontier.deduplicate();
      // prune visited vertices

      _frontier.filter([this](const VertexId vertex) { return _distances.get(vertex) != std::numeric_limits<uint64_t>::max(); });
      current_frontier = _frontier.local_frontier();
      local_active = !current_frontier.empty();
      // kamping::measurements::timer().stop();

      // kamping::measurements::timer().synchronize_and_start("bfs_allreduce");
      global_active = _comm.allreduce_single(kamping::send_buf(local_active), kamping::op(kamping::ops::logical_or<>{}));
      // kamping::measurements::timer().stop();

      // kamping::measurements::timer().synchronize_and_start("bfs_increment_distance");
      ++current_distance;
      // kamping::measurements::timer().stop();
    }
    // kamping::measurements::timer().stop();
    // kamping::measurements::timer().stop();
  }

  const distributed::DistributedArray<uint64_t>& distances() const { return _distances; }

  SetBasedDistributedFrontier &frontier() {
    return _frontier;
  }

  const void updateCommunicator(kamping::Communicator<> &new_comm) {
    _comm = new_comm;
    _frontier.updateCommunicator(new_comm);
  }

private:
  int _initial_rank;
  kamping::Communicator<> _comm;
  SetBasedDistributedFrontier _frontier;
  std::shared_ptr<DistributedCSRGraph> _graph;
  distributed::DistributedArray<uint64_t> _distances;
  uint64_t current_distance = 0;
};