#include <kamping/collectives/allreduce.hpp>
#include <unordered_map>
#include <unordered_set>

#include "distributed_array.hpp"
#include "distributed_set.hpp"

template <typename T> class TraditionalDistributedFrontier {
public:
  TraditionalDistributedFrontier(std::shared_ptr<DistributionStrategy> strategy, kamping::Communicator<> comm, std::vector<T> initial_frontier)
      : _strategy(std::move(strategy)), _comm(comm) {
    add_multiple(initial_frontier);
  }

  void add(T element) {
    int owner = _strategy->owner(element);
    if (owner == _comm.rank() && _visited.find(element) == _visited.end()) {
      _next_frontier.insert(element);
      return;
    }
    _outgoing_data[owner].push_back(element);
  }

  void add_multiple(std::vector<T> elements) {
    for (const auto &element : elements) {
      add(element);
    }
  }

  void exchange() {
    std::vector<int> send_counts(_comm.size(), 0);
    std::vector<T> send_buffer;
    std::vector<T> recv_buffer;

    for (int rank = 0; rank < _comm.size(); ++rank) {
      send_counts[rank] = _outgoing_data[rank].size();
      send_buffer.insert(send_buffer.end(), _outgoing_data[rank].begin(), _outgoing_data[rank].end());
    }

    _comm.alltoallv(kamping::send_buf(send_buffer), kamping::send_counts(send_counts),
                    kamping::recv_buf<kamping::BufferResizePolicy::grow_only>(recv_buffer));

    _local_frontier.clear();

    _local_frontier.insert(_next_frontier.begin(), _next_frontier.end());
    _next_frontier.clear();

    _local_frontier.insert(recv_buffer.begin(), recv_buffer.end());
    _outgoing_data.clear();

    for (const auto &element : _visited) {
      _local_frontier.erase(element);
    }

    _current_step += 1;
    for (auto const &x : _local_frontier) {
      _visited.insert(x);
      _distances[x] = _current_step;
    }
  }

  const std::unordered_set<T> &local_frontier() const { return _local_frontier; }
  const std::unordered_map<T, uint64_t> &distances() const { return _distances; }

private:
  std::shared_ptr<DistributionStrategy> _strategy;
  kamping::Communicator<> _comm;
  std::unordered_set<T> _local_frontier;
  std::unordered_set<T> _next_frontier;
  std::unordered_map<int, std::vector<T>> _outgoing_data;
  std::unordered_set<T> _visited;
  std::unordered_map<T, uint64_t> _distances;
  uint64_t _current_step = 0;
};

// Wrapper class for DistributedSet
template <typename T> class SetBasedDistributedFrontier {
public:
  SetBasedDistributedFrontier(std::shared_ptr<DistributionStrategy> strategy, kamping::Communicator<> comm, std::vector<T> initial_frontier)
      : _distributed_set(distributed::DistributedSet<size_t>(strategy, comm)) {
    _distributed_set.insert(initial_frontier);
  }

  void add(T element) { _distributed_set.insert(element); }

  void exchange() { _distributed_set.exchange(); }

  const std::vector<T> &local_frontier() {
    _distributed_set.deduplicate();
    return _distributed_set.local_data();
  }

private:
  distributed::DistributedSet<T> _distributed_set;
};

struct VertexEdgeMapping {
  size_t edge_start_index; // inclusive
  size_t edge_end_index;   // exclusive
  VertexEdgeMapping() : edge_start_index(0), edge_end_index(0) {}
  VertexEdgeMapping(size_t start, size_t end) : edge_start_index(start), edge_end_index(end) {}
};

struct DistributedCSRGraph {
  distributed::DistributedArray<VertexEdgeMapping> vertex_array;
  distributed::DistributedArray<size_t> edge_array;
  std::shared_ptr<DistributionStrategy> vertex_dist;
  DistributedCSRGraph(distributed::DistributedArray<VertexEdgeMapping> vertex_arr, distributed::DistributedArray<size_t> edge_arr,
                      std::shared_ptr<DistributionStrategy> dist)
      : vertex_array(std::move(vertex_arr)), edge_array(std::move(edge_arr)), vertex_dist(dist) {}
};

class DistributedBFS {
public:
  DistributedBFS(std::unique_ptr<DistributedCSRGraph> graph, kamping::Communicator<> const &comm, std::vector<size_t> source_vertices)
      : _graph(std::move(graph)), _comm(comm), _frontier(SetBasedDistributedFrontier<size_t>(graph->vertex_dist, _comm, source_vertices)) {}

  void run() {
    auto current_frontier = _frontier.local_frontier(); // contains ids of vertices
    bool local_active = !current_frontier.empty();
    bool global_active = true;

    while (global_active) {
      std::vector<size_t> next_frontier_local;
      std::unordered_map<int, std::vector<size_t>> next_frontier_outgoing;

      for (size_t vertex : current_frontier) {
        if (_graph->vertex_dist->owner(vertex) != _comm.rank()) {
          throw std::logic_error("Vertex " + std::to_string(vertex) + " belongs to another worker: " +
                                 std::to_string(_graph->vertex_dist->owner(vertex)) + ", but was found in worker " + std::to_string(_comm.rank()));
        }

        auto [edge_index_start, edge_index_end] = _graph->vertex_array.get(vertex);

        for (size_t i = edge_index_start; i < edge_index_end; ++i) {
          size_t neighbor = _graph->edge_array.get(i);
          _frontier.add(neighbor);
        }
      }
      _frontier.exchange();
      current_frontier = _frontier.local_frontier();

      std::cout << "Current frontier in PE " << _comm.rank() << '\n';
      for (auto const &x : current_frontier) {
        std::cout << x << ", ";
      }
      std::cout << '\n';

      local_active = !current_frontier.empty();
      global_active = _comm.allreduce_single(kamping::send_buf(local_active), kamping::op(kamping::ops::logical_or<>{}));
      std::cout << "Global active " << global_active << '\n';
    }
  }

  // std::unordered_map<size_t, uint64_t> getDistances() { return _frontier.distances(); }

private:
  kamping::Communicator<> _comm;
  std::unordered_set<size_t> _visited;
  SetBasedDistributedFrontier<size_t> _frontier;
  std::unique_ptr<DistributedCSRGraph> _graph;
};