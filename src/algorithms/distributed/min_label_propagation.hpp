#pragma once
#include "../common/graph.hpp"

typedef uint64_t Label;

class MinLabelPropagation {
public:
  void run() {
    int rank = _comm.rank();
    size_t vertex_start = _graph.vertex_dist->to_global_index(rank, 0);
    size_t vertex_end = vertex_start + _graph.vertex_dist->local_size(rank); 
    std::function<Label (Label, Label)> minOp = [](Label x, Label y) { return std::min(x, y); };
    bool converged = false;

    while(!converged) {
      std::vector<Label> prev_labels = _labels.local_data();
      for (VertexId vertex = vertex_start; vertex < vertex_end; ++vertex) {
        Label own_label = _labels.get(vertex);
        auto [edge_start, edge_end] = _graph.vertex_array.get(vertex);
        for (EdgeId edge = edge_start; edge < edge_end; ++edge) {
          VertexId neighbor = _graph.edge_array.get(edge);
          _labels.set(neighbor, own_label, minOp);
        }
      }
      _labels.exchange();
      converged = true;
      for (VertexId vertex = vertex_start; vertex < vertex_end; ++vertex) {
        Label prev_label = prev_labels[vertex - vertex_start];
        Label own_label = _labels.get(vertex);
        if (prev_label != own_label) {
          converged = false;
          break;
        }
      }
      converged = _comm.allreduce_single(kamping::send_buf(converged), kamping::op(kamping::ops::logical_and<>{}));
    }
  }

private:
  kamping::Communicator<> _comm;
  DistributedCSRGraph _graph;
  distributed::DistributedArray<Label> _labels;
};