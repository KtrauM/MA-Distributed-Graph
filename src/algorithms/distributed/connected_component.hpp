#pragma once

#include "../common/graph.hpp"
#include "bfs.hpp"

class BFSBasedDistributedConnectedComponent {
public:
  BFSBasedDistributedConnectedComponent(std::shared_ptr<DistributedCSRGraph> graph, kamping::Communicator<> const &comm)
      : _graph(std::move(graph)), _comm(comm), _bfs_runner(graph, _comm, std::vector<VertexId>()) {}

  uint32_t run() {
    int rank = _comm.rank();
    size_t vertex_start = _graph->vertex_dist->to_global_index(rank, 0);
    size_t vertex_end = vertex_start + _graph->vertex_dist->local_size(rank);
    uint32_t num_components = 0;
    for (VertexId vertex = vertex_start; vertex < vertex_end; ++vertex) {
        SetBasedDistributedFrontier &frontier = _bfs_runner.frontier();
        const std::vector<VertexId> &visited = frontier.visited();
        if (std::find(visited.begin(), visited.end(), vertex) != visited.end()) {
            continue;
        }
        frontier.add(vertex);
        _bfs_runner.run();
        num_components++;
    }
    _comm.allreduce_single(kamping::send_buf(num_components), kamping::op(kamping::ops::plus<uint32_t>{}));
    return num_components;
  }

private:
  kamping::Communicator<> _comm;
  DistributedBFS _bfs_runner;
  std::shared_ptr<DistributedCSRGraph> _graph;
};

// class LabelPropagationBasedDistributedConnectedComponent {
// public:
//   LabelPropagationBasedDistributedConnectedComponent(std::shared_ptr<DistributedCSRGraph> graph, kamping::Communicator<> const &comm)
//       : _graph(std::move(graph)), _comm(comm) {}

//   uint32_t run() {

//   }
// };