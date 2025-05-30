#pragma once

#include "../common/graph.hpp"
#include "bfs.hpp"
#include <kamping/measurements/timer.hpp>
#include "min_label_propagation.hpp"

class BFSBasedDistributedConnectedComponentWithSubCommOptimization {
public:
  BFSBasedDistributedConnectedComponentWithSubCommOptimization(std::shared_ptr<DistributedCSRGraph> graph, kamping::Communicator<> const &comm)
      : _graph(std::move(graph)), _comm(comm), _bfs_runner(graph, _comm, std::vector<VertexId>()) {}

  uint32_t run() {
    if (_comm.rank() == 0) {
      std::cout << "CC started\n";
    }
    int initial_rank = _comm.rank();
    size_t vertex_start = _graph->vertex_dist->to_global_index(initial_rank, 0);
    size_t vertex_end = vertex_start + _graph->vertex_dist->local_size(initial_rank);

    // Block distribution is assumed, the max iteration count is the size of the vertex set on rank 0.
    assert(typeid(*_graph->vertex_dist) == typeid(BlockDistribution) && "vertex_dist must be BlockDistribution");
    size_t max_iteration_count = _graph->vertex_dist->local_size(0);


    uint32_t num_components = 0;
    int local_bfs_active = 1;
    for (VertexId vertex = vertex_start; vertex < vertex_start + max_iteration_count; ++vertex) {
        // All vertices belonging to this PE have been already processed,
        // thus we can reduce the number of PEs communicating.
        // However splitting the comm is not free, as it requires
        // an allgather step to determine the groups.
        // https://www.mpich.org/static/docs/v3.3/www3/MPI_Comm_split.html 
        if (vertex >= vertex_end) {
          local_bfs_active = 0;
        }
        kamping::Communicator<> sub_comm_bfs = _comm.split(local_bfs_active);
        _bfs_runner.updateCommunicator(sub_comm_bfs);


        SetBasedDistributedFrontier &frontier = _bfs_runner.frontier();
        const auto &distances = _bfs_runner.distances();
        if (vertex < vertex_end && distances.get(vertex) == std::numeric_limits<uint64_t>::max()) {
          frontier.add(vertex);
          num_components++;
        }

        _bfs_runner.run();
        std::cout << "Num components on rank " << _comm.rank() << " updated to " << num_components << "\n";
    }
    std::cout << "Num components on rank " << _comm.rank() << " is " << num_components << "\n";
    uint32_t total_components = _comm.allreduce_single(kamping::send_buf(num_components), kamping::op(kamping::ops::plus<uint32_t>{}));
    std::cout << "Total components: " << total_components << "\n";

    return total_components;
  }

private:
  kamping::Communicator<> _comm;
  DistributedBFS _bfs_runner;
  std::shared_ptr<DistributedCSRGraph> _graph;
};

class BFSBasedDistributedConnectedComponent {
public:
  BFSBasedDistributedConnectedComponent(std::shared_ptr<DistributedCSRGraph> graph, kamping::Communicator<> const &comm)
      : _graph(std::move(graph)), _comm(comm), _bfs_runner(graph, _comm, std::vector<VertexId>()) {}

  uint32_t run() {
    if (_comm.rank() == 0) {
      std::cout << "CC started\n";
    }
    int initial_rank = _comm.rank();
    size_t vertex_start = _graph->vertex_dist->to_global_index(initial_rank, 0);
    size_t vertex_end = vertex_start + _graph->vertex_dist->local_size(initial_rank);

    // Block distribution is assumed, the max iteration count is the size of the vertex set on rank 0.
    assert(typeid(*_graph->vertex_dist) == typeid(BlockDistribution) && "vertex_dist must be BlockDistribution");
    size_t max_iteration_count = _graph->vertex_dist->local_size(0);

    uint32_t num_components = 0;
    for (VertexId vertex = vertex_start; vertex < vertex_start + max_iteration_count; ++vertex) {
        SetBasedDistributedFrontier &frontier = _bfs_runner.frontier();
        const auto &distances = _bfs_runner.distances();
        if (vertex < vertex_end && distances.get(vertex) == std::numeric_limits<uint64_t>::max()) {
          frontier.add(vertex);
          num_components++;
        }
        kamping::measurements::timer().synchronize_and_start("bfs_run_iteration");
        _bfs_runner.run();
        kamping::measurements::timer().stop_and_add();
    }

    uint32_t total_components = _comm.allreduce_single(kamping::send_buf(num_components), kamping::op(kamping::ops::plus<uint32_t>{}));
    
    return total_components;
  }

  uint32_t max_num_iterations() const { return _bfs_runner.max_num_iterations; }
  uint32_t max_send_buffer_size() const { return _bfs_runner.max_send_buffer_size; }

private:
  kamping::Communicator<> _comm;
  DistributedBFS _bfs_runner;
  std::shared_ptr<DistributedCSRGraph> _graph;
};

class LabelPropagationBasedDistributedConnectedComponent {
public:
  LabelPropagationBasedDistributedConnectedComponent(std::shared_ptr<DistributedCSRGraph> graph, kamping::Communicator<> const &comm)
      : _graph(std::move(graph)), _comm(comm), _label_propagation_runner(comm, graph), _unique_labels(comm) {}

  uint32_t run() {
    _label_propagation_runner.run();
    std::vector<Label> &local_labels = _label_propagation_runner.labels().local_data();
    for (VertexId vertex = 0; vertex < local_labels.size(); ++vertex) {
      _unique_labels.insert(local_labels[vertex]);
    }
    _unique_labels.deduplicate();
    _unique_labels.redistribute([this](const Label label) { return _graph->vertex_dist->owner(label); });
    _unique_labels.deduplicate();
    uint32_t num_components = _comm.allreduce_single(kamping::send_buf((uint32_t)_unique_labels.local_data().size()), kamping::op(kamping::ops::plus<uint32_t>{}));
    return num_components;
  }

  private:
  MinLabelPropagation _label_propagation_runner;
  kamping::Communicator<> _comm;
  std::shared_ptr<DistributedCSRGraph> _graph;
  distributed::DistributedSet<Label> _unique_labels;
};