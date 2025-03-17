#pragma once

#include <kamping/collectives/allreduce.hpp>

#include "../../primitives/distributed_array.hpp"
#include "../../primitives/distributed_set.hpp"
#include "../common/graph.hpp"

class SetBasedDistributedFrontier {
public:
  SetBasedDistributedFrontier(kamping::Communicator<> comm, std::vector<VertexId> initial_frontier)
      : _local_frontier(distributed::DistributedSet<VertexId>(comm)), _visited(distributed::DistributedSet<VertexId>(comm)) {
    if (comm.rank() == 0) {
      _local_frontier.insert(initial_frontier);
    }
  }

  void add(VertexId vertex) { _local_frontier.insert(vertex); }

  void exchange(std::function<int(const VertexId)> mapping) {
    _local_frontier.redistribute(mapping);
  }

  const std::vector<VertexId> &local_frontier() {
    _local_frontier.deduplicate();
    _local_frontier.filter([this](const VertexId &vertex) { return _visited.contains(vertex); });
    _visited.insert(_local_frontier);
    return _local_frontier.local_data();
  }

  const std::vector<VertexId> &visited() const {
    return _visited.local_data();
  }

private:
  distributed::DistributedSet<VertexId> _local_frontier;
  distributed::DistributedSet<VertexId> _visited;
};


class DistributedBFS {
public:
  DistributedBFS(std::shared_ptr<DistributedCSRGraph> graph, kamping::Communicator<> const &comm, std::vector<VertexId> source_vertices)
      : _graph(std::move(graph)), _comm(comm), _frontier(SetBasedDistributedFrontier(_comm, source_vertices)), _distances(_graph->vertex_dist, comm) {}

  void run() {
    auto current_frontier = _frontier.local_frontier(); // contains ids of vertices
    bool local_active = !current_frontier.empty();
    bool global_active = true;
    while (global_active) {
      for (VertexId vertex : current_frontier) {
        std::function<uint64_t (uint64_t, uint64_t)> minOp = [](uint64_t x, uint64_t y) { return std::min(x, y); };
        _distances.set(vertex, current_distance, minOp);
        if (_graph->vertex_dist->owner(vertex) != _comm.rank()) {
          throw std::logic_error("Vertex " + std::to_string(vertex) + " belongs to another worker: " +
                                 std::to_string(_graph->vertex_dist->owner(vertex)) + ", but was found in worker " + std::to_string(_comm.rank()));
        }

        auto [edge_index_start, edge_index_end] = _graph->vertex_array.get(vertex);
        for (EdgeId i = edge_index_start; i < edge_index_end; ++i) {
          VertexId neighbor = _graph->edge_array.get(i);
          _frontier.add(neighbor);
        }
      }

      _frontier.exchange([this](const size_t vertex_id) { return _graph->vertex_dist->owner(vertex_id); });
      current_frontier = _frontier.local_frontier();

      local_active = !current_frontier.empty();
      global_active = _comm.allreduce_single(kamping::send_buf(local_active), kamping::op(kamping::ops::logical_or<>{}));
      ++current_distance;
    }
  }

  const distributed::DistributedArray<uint64_t>& distances() const { return _distances; }

  SetBasedDistributedFrontier &frontier() {
    return _frontier;
  }

private:
  kamping::Communicator<> _comm;
  SetBasedDistributedFrontier _frontier;
  std::shared_ptr<DistributedCSRGraph> _graph;
  distributed::DistributedArray<uint64_t> _distances;
  uint64_t current_distance = 0;
};