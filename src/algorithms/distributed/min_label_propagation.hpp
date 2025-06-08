// #pragma once
// #include "../common/graph.hpp"
// #include <limits>
// #include "../../primitives/distributed_array.hpp"

// typedef uint64_t Label;

// class MinLabelPropagation {
// public:
//   MinLabelPropagation(kamping::Communicator<> const &comm, std::shared_ptr<DistributedCSRGraph> graph)
//       : _comm(comm), _graph(std::move(graph)), _labels(_graph->vertex_dist, comm) {}


//   void run() {
//     int rank = _comm.rank();
//     size_t vertex_start = _graph->vertex_dist->to_global_index(rank, 0);
//     size_t vertex_end = vertex_start + _graph->vertex_dist->local_size(rank); 
//     bool converged = false;

//     // Initialize labels to the vertex id
//     for (VertexId vertex = vertex_start; vertex < vertex_end; ++vertex) {
//       _labels.set(vertex, vertex, distributed::OperationType::IDENTITY);
//     }

//     while(!converged) {
//       std::vector<Label> prev_labels = _labels.local_data(); // copy labels before the iteration
//       for (VertexId vertex = vertex_start; vertex < vertex_end; ++vertex) {
//         Label own_label = _labels.get(vertex);
//         auto [edge_start, edge_end] = _graph->vertex_array.get(vertex);
//         for (EdgeId edge = edge_start; edge < edge_end; ++edge) {
//           VertexId neighbor = _graph->edge_array.get(edge);
//           _labels.set(neighbor, own_label, distributed::OperationType::MIN);
//         }
//       }
//       kamping::measurements::timer().start("frontier_exchange");
//       _labels.exchange(_comm);
//       kamping::measurements::timer().stop_and_add();
//       converged = true;
//       for (VertexId vertex = vertex_start; vertex < vertex_end; ++vertex) {
//         Label prev_label = prev_labels[vertex - vertex_start];
//         Label own_label = _labels.get(vertex);
//         if (prev_label != own_label) {
//           converged = false;
//           break;
//         }
//       }
//       converged = _comm.allreduce_single(kamping::send_buf(converged), kamping::op(kamping::ops::logical_and<>{}));
//     }
//   }

//   distributed::DistributedArray<Label> &labels() {
//     return _labels;
//   }

// private:
//   kamping::Communicator<> _comm;
//   std::shared_ptr<DistributedCSRGraph> _graph;
//   distributed::DistributedArray<Label> _labels;
// };