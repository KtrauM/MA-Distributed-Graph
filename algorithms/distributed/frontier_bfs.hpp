#pragma once

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