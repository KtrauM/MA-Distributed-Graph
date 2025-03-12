#pragma once

#include "algorithms/common/graph.hpp"
#include <vector>

class BFS {
public:
  virtual void run() = 0;
  virtual void set_start_nodes(std::vector<VertexId> start_nodes) = 0;
};

