#pragma once

#include "../common/bfs.hpp"
#include <vector>
#include <unordered_set>

class SequentialBFS : public BFS {
    public:
    void run() override {
        std::vector<VertexId> current_frontier = std::copy(_start_nodes);
        
        while (!current_frontier.empty()) {
            std::vector<VertexId> next_frontier;
            for (VertexId vertex_id: current_frontier) {
                if (_visited.find(vertex_id) != _visited.end()) {
                    continue;
                }
                _visited.insert(vertex_id);

                for (EdgeId edge_id = _graph.vertex_array[vertex_id]; edge_id < _graph.vertex_array[vertex_id + 1]; ++edge_id) {
                    VertexId next_vertex = _graph.edge_array[edge_id];
                    if (_visited.find(next_vertex) == _visited.end()) {
                        next_frontier.push_back(next_vertex);
                    }
                }
            }
            current_frontier = std::move(next_frontier);
        }
    }

    void set_start_nodes(std::vector<VertexId> start_nodes) override {
        _start_nodes = std::move(start_nodes);
    }

    private:
    std::vector<VertexId> _start_nodes;
    std::unordered_set<VertexId> _visited;
    CSRGraph _graph;
};
