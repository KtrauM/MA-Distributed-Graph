#pragma once

#include "../common/graph.hpp"
#include "bfs.hpp"

class BFSBasedDistributedConnectedComponent {
public:
  BFSBasedDistributedConnectedComponent(std::shared_ptr<DistributedCSRGraph> graph, kamping::Communicator<> const &comm)
      : _graph(std::move(graph)), _comm(comm), _bfs_runner(graph, _comm, std::vector<VertexId>()) {}

  uint32_t run() {
    std::cout << "CC started\n";
    int rank = _comm.rank();
    size_t vertex_start = _graph->vertex_dist->to_global_index(rank, 0);
    size_t vertex_end = vertex_start + _graph->vertex_dist->local_size(rank);

    // Block distribution is assumed, the max iteration count is the size of the vertex set on rank 0.
    assert(typeid(*_graph->vertex_dist) == typeid(BlockDistribution) && "vertex_dist must be BlockDistribution");
    size_t max_iteration_count = _graph->vertex_dist->local_size(0);

    uint32_t num_components = 0;
    int local_bfs_active = 1;
    for (VertexId vertex = vertex_start; vertex < vertex_start + max_iteration_count; ++vertex) {
        SetBasedDistributedFrontier &frontier = _bfs_runner.frontier();
        const auto &distances = _bfs_runner.distances();
        if (vertex >= vertex_end) {
          local_bfs_active = 0;
        }
        kamping::Communicator<> sub_comm_bfs = _comm.split(local_bfs_active);
        std::cout << "sub_comm_bfs has size " << sub_comm_bfs.size() << " on PE " << _comm.rank() << ", new rank is " << sub_comm_bfs.rank() << "\n";
        _bfs_runner.updateCommunicator(sub_comm_bfs);

        std::cout << "Checking whether vertex " << vertex << " is visited on PE " << _comm.rank() << "\n";
        if (vertex >= vertex_end || distances.get(vertex) != std::numeric_limits<uint64_t>::max()) {
            continue;
        }

        std::cout << "Adding vertex " << vertex << "to frontier: " << frontier.toString() << "\n";
        frontier.add(vertex);
        std::cout << "BFS started on PE " << _comm.rank() << "\n";
        _bfs_runner.run();
        std::cout << "BFS finished on PE" << _comm.rank() << "\n";
        num_components++;
    }
    std::cout << "REACHED BARRIER ON PE " << _comm.rank() << ", comm size: " << _comm.size() << "\n";
    // _comm.barrier();
    std::cout << "Allreduce starte on PE " << _comm.rank() << "\n";
    _comm.allreduce_single(kamping::send_buf(num_components), kamping::op(kamping::ops::plus<uint32_t>{}));
    std::cout << "Allreduce finished\n";
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