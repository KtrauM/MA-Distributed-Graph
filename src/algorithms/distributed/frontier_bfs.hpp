#pragma once

template <typename T> class TraditionalDistributedFrontier {
    public:
      TraditionalDistributedFrontier(std::shared_ptr<DistributionStrategy> strategy, kamping::Communicator<> comm)
          : _strategy(std::move(strategy)), _comm(comm) {}

      TraditionalDistributedFrontier(std::shared_ptr<DistributionStrategy> strategy, kamping::Communicator<> comm, std::vector<VertexId> initial_frontier)
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
    
      bool exchange() {
        // kamping::measurements::timer().synchronize_and_start("whole_exchange");
        // kamping::measurements::timer().synchronize_and_start("buffer_alloc");
        std::vector<int> send_counts(_comm.size(), 0);
        std::vector<T> send_buffer;
        std::vector<T> recv_buffer;
        // kamping::measurements::timer().stop_and_add();
        
        // kamping::measurements::timer().synchronize_and_start("construct_send_buffer");
        for (int rank = 0; rank < _comm.size(); ++rank) {
          send_counts[rank] = _outgoing_data[rank].size();
          send_buffer.insert(send_buffer.end(), _outgoing_data[rank].begin(), _outgoing_data[rank].end());
        }
        // kamping::measurements::timer().stop_and_add();

        // kamping::measurements::timer().synchronize_and_start("exchange_alltoallv");
        _comm.alltoallv(kamping::send_buf(send_buffer), kamping::send_counts(send_counts),
                        kamping::recv_buf<kamping::BufferResizePolicy::grow_only>(recv_buffer));
        // kamping::measurements::timer().stop_and_add();
        // kamping::measurements::timer().synchronize_and_start("local_copying");
        _local_frontier.clear();
        
        ++_current_step;
        _local_frontier.reserve(_next_frontier.size() + recv_buffer.size());
        for (const auto& v : _next_frontier) {
          if (_visited.find(v) == _visited.end()) {
            _local_frontier.insert(v);
            _visited.insert(v);
            _distances[v] = _current_step;
          }
        }
        _next_frontier.clear();

        for (const auto& v : recv_buffer) {
          if (_visited.find(v) == _visited.end()) {
            _local_frontier.insert(v);
            _visited.insert(v);
            _distances[v] = _current_step;
          }
        }
        _outgoing_data.clear();
        // kamping::measurements::timer().stop_and_add();
        // kamping::measurements::timer().synchronize_and_start("allreduce_single");
        bool all_empty = _comm.allreduce_single(kamping::send_buf(_local_frontier.empty()), kamping::op(kamping::ops::logical_and<>{}));
        // kamping::measurements::timer().stop_and_add();
        // kamping::measurements::timer().stop_and_add();
        return all_empty;
      }

      bool visited(const T &element) {
        return _local_frontier.find(element) != _local_frontier.end();
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

class TraditionalDistributedBFS {
  public:
  TraditionalDistributedBFS(std::shared_ptr<DistributedCSRGraph> graph, kamping::Communicator<> comm, std::vector<VertexId> initial_frontier)
          : _graph(std::move(graph)), _comm(comm), _frontier(graph->vertex_dist, comm) {}

  void run() {
    bool done = false;
    while (!done) {
      // kamping::measurements::timer().synchronize_and_start("traverse_neighbors_of_local_frontier");
      for (const auto &vertex: _frontier.local_frontier()) {
        const auto &[edge_index_start, edge_index_end] = _graph->vertex_array.get(vertex);
        for (EdgeId edge_idx = edge_index_start; edge_idx < edge_index_end; ++edge_idx) {
          VertexId neighbor = _graph->edge_array.get(edge_idx);
          _frontier.add(neighbor);
        }
      }
      // kamping::measurements::timer().stop_and_add();
      // kamping::measurements::timer().synchronize_and_start("exchange_wrapper");
      done = _frontier.exchange();
      // kamping::measurements::timer().stop_and_add();
    }
  }

  TraditionalDistributedFrontier<VertexId> &frontier() { return _frontier; }

  private:
  TraditionalDistributedFrontier<VertexId> _frontier;
  std::shared_ptr<DistributedCSRGraph> _graph;
  kamping::Communicator<> _comm;
};