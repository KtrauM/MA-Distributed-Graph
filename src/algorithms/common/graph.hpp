#pragma once

#include "../../primitives/distributed_array.hpp"
#include <stdint.h>
#include <vector>

typedef uint64_t VertexId;
typedef uint64_t EdgeId;

struct VertexEdgeMapping {
  EdgeId edge_start_index; // inclusive
  EdgeId edge_end_index;   // exclusive
  VertexEdgeMapping() : edge_start_index(0), edge_end_index(0) {}
  VertexEdgeMapping(size_t start, size_t end) : edge_start_index(start), edge_end_index(end) {}
};

struct CSRGraph {
  std::vector<EdgeId> vertex_array;
  std::vector<VertexId> edge_array;

  CSRGraph(std::vector<EdgeId> vertex_array, std::vector<VertexId> edge_array)
      : vertex_array(std::move(vertex_array)), edge_array(std::move(edge_array)) {}
};

struct DistributedCSRGraph {
  distributed::DistributedArray<VertexEdgeMapping> vertex_array;
  distributed::DistributedArray<VertexId> edge_array;
  std::shared_ptr<DistributionStrategy> vertex_dist;
  DistributedCSRGraph(distributed::DistributedArray<VertexEdgeMapping> vertex_array, distributed::DistributedArray<size_t> edge_array,
                      std::shared_ptr<DistributionStrategy> dist)
      : vertex_array(std::move(vertex_array)), edge_array(std::move(edge_array)), vertex_dist(dist) {}
};


