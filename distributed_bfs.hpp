#include <kamping/collectives/allreduce.hpp>
#include <unordered_set>
#include <unordered_map>

#include "distributed_array.hpp"

template <typename T> class DistributedFrontier {
public:
  DistributedFrontier(std::unique_ptr<DistributionStrategy> strategy, kamping::Communicator<> comm) : _strategy(std::move(strategy)), _comm(comm) {}

  void add(T element) {
    int owner = _strategy->owner(element);
    if (owner == _comm.rank() && _visited.find(element) == _visited.end()) {
      _next_frontier.push_back(element);
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
    
    _local_frontier.insert(_local_frontier.end(), _next_frontier.begin(), _next_frontier.end());
    _next_frontier.clear();

    _local_frontier.insert(_local_frontier.end(), recv_buffer.begin(), recv_buffer.end());
    _outgoing_data.clear();
  }

  const std::vector<T> &local_frontier() const { return _local_frontier; }

private:
  std::unique_ptr<DistributionStrategy> _strategy;
  kamping::Communicator<> _comm;
  std::vector<T> _local_frontier;
  std::vector<T> _next_frontier;
  std::unordered_map<int, std::vector<T>> _outgoing_data;
  std::unordered_set<size_t> _visited;
};

struct DistributedCSRGraph {
  // TODO: Not sure if std::pair is a valid choice here
  // TODO: otherwise how can we handle the vertices at the end of chunk boundary, as vertex_array[v + 1] is non-local
  distributed::DistributedArray<std::pair<size_t, size_t>> vertex_array; // {vertex_array[v], vertex_array[v + 1]}
  distributed::DistributedArray<size_t> edge_array;
  std::shared_ptr<DistributionStrategy> vertex_dist;
};

class DistributedBFS {
public:
  DistributedBFS(std::unique_ptr<DistributedCSRGraph> graph, kamping::Communicator<> const &comm) : _graph(std::move(graph)), _comm(comm) {}

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
        // vertex array:
        // 0   1   2    3
        // 0,1 1,3 3,6 6,8
        auto [edge_index_start, edge_index_end] = _graph->vertex_array.get(vertex);

        for (size_t i = edge_index_start; i < edge_index_end; ++i) {
          size_t neighbor = _graph->edge_array.get(i);
          _frontier.add(neighbor);
        }

        _frontier.exchange();
        current_frontier = _frontier.local_frontier();
        local_active = !current_frontier.empty();
        global_active = _comm.allreduce_single(kamping::send_buf(local_active), kamping::op(kamping::ops::logical_or<>{}));
      }
    }
  }

private:
  kamping::Communicator<> _comm;
  std::unordered_set<size_t> _visited;
  DistributedFrontier<size_t> _frontier;
  std::unique_ptr<DistributedCSRGraph> _graph;
};