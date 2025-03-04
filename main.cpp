#include "distributed_array.hpp"
#include "distributed_bfs.hpp"
#include "distribution_strategy.hpp"
#include <kagen.h>
#include <kamping/collectives/allgather.hpp>
#include <kamping/collectives/allreduce.hpp>
#include <kamping/collectives/alltoall.hpp>
#include <kamping/collectives/reduce.hpp>
#include <kamping/communicator.hpp>
#include <kamping/measurements/printer.hpp>
#include <kamping/measurements/timer.hpp>
#include <memory>
#include <vector>

#include "CLI11.hpp"
#include <iostream>
#include <mpi.h>

void PrintGraphAttributes(const kagen::Graph &graph) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get the rank of the process

  std::cout << "[PE " << rank << "] Vertex Range: (" << graph.vertex_range.first << ", " << graph.vertex_range.second << ")\n";
  std::cout << "[PE " << rank << "] Graph Representation: " << static_cast<int>(graph.representation) << "\n";
  MPI_Barrier(MPI_COMM_WORLD);

  std::cout << "[PE " << rank << "] Edges (Edgelist):\n";
  for (const auto &edge : graph.edges) {
    std::cout << "[PE " << rank << "] (" << edge.first << ", " << edge.second << ")\n";
  }
  MPI_Barrier(MPI_COMM_WORLD);

  std::cout << "[PE " << rank << "] XadjArray:\n";
  for (const auto &val : graph.xadj) {
    std::cout << "[PE " << rank << "] " << val << " ";
  }
  std::cout << "\n";
  MPI_Barrier(MPI_COMM_WORLD);

  std::cout << "[PE " << rank << "] AdjncyArray:\n";
  for (const auto &val : graph.adjncy) {
    std::cout << "[PE " << rank << "] " << val << " ";
  }
  std::cout << "\n";
  MPI_Barrier(MPI_COMM_WORLD);

  std::cout << "[PE " << rank << "] Vertex Weights:\n";
  for (const auto &weight : graph.vertex_weights) {
    std::cout << "[PE " << rank << "] " << weight << " ";
  }
  std::cout << "\n";
  MPI_Barrier(MPI_COMM_WORLD);

  std::cout << "[PE " << rank << "] Edge Weights:\n";
  for (const auto &weight : graph.edge_weights) {
    std::cout << "[PE " << rank << "] " << weight << " ";
  }
  std::cout << "\n";
  MPI_Barrier(MPI_COMM_WORLD);

  std::cout << "[PE " << rank << "] Coordinates (2D):\n";
  for (const auto &coord : graph.coordinates.first) {
    std::cout << "[PE " << rank << "] (" << std::get<0>(coord) << ", " << std::get<1>(coord) << ")\n";
  }
  MPI_Barrier(MPI_COMM_WORLD);

  std::cout << "[PE " << rank << "] Coordinates (3D):\n";
  for (const auto &coord : graph.coordinates.second) {
    std::cout << "[PE " << rank << "] (" << std::get<0>(coord) << ", " << std::get<1>(coord) << ", " << std::get<2>(coord) << ")\n";
  }
  MPI_Barrier(MPI_COMM_WORLD);

  std::cout << "[PE " << rank << "] Number of Local Vertices: " << graph.NumberOfLocalVertices() << "\n";
  MPI_Barrier(MPI_COMM_WORLD);
  std::cout << "[PE " << rank << "] Number of Global Vertices: " << graph.NumberOfGlobalVertices() << "\n";
  MPI_Barrier(MPI_COMM_WORLD);
  std::cout << "[PE " << rank << "] Number of Local Edges: " << graph.NumberOfLocalEdges() << "\n";
  MPI_Barrier(MPI_COMM_WORLD);
  std::cout << "[PE " << rank << "] Number of Global Edges: " << graph.NumberOfGlobalEdges() << "\n";
}

void SetupCommandLineArguments(CLI::App &app, std::string &generator_options, std::vector<size_t> &bfs_start_vertices, bool &print_graph) {
  app.add_option("-g,--generator", generator_options, "Graph generator options string (e.g., 'grid2d;grid_x=32;grid_y=12')")->required();

  app.add_option("-s,--start", bfs_start_vertices, "BFS starting vertices (comma-separated list)")
      ->delimiter(',')
      ->default_val(std::vector<size_t>{0});

  app.add_flag("-p,--print-graph", print_graph, "Print detailed graph attributes");
}

int main(int argc, char **argv) {
  kamping::Environment env(argc, argv);
  auto comm = kamping::comm_world();

  // Setup CLI args
  CLI::App app{"Distributed BFS Benchmark"};
  std::string generator_options;
  std::vector<size_t> bfs_start_vertices;
  bool print_graph = false;
  SetupCommandLineArguments(app, generator_options, bfs_start_vertices, print_graph);
  CLI11_PARSE(app, argc, argv);

  // BFS example
  kagen::KaGen gen(MPI_COMM_WORLD);
  gen.UseCSRRepresentation();

  kamping::measurements::timer().synchronize_and_start("kagen_gen");
  auto kagen_graph = gen.GenerateFromOptionString(generator_options);
  if (print_graph) {
    PrintGraphAttributes(kagen_graph);
  }
  kamping::measurements::timer().stop();

  // Setup edge array
  kamping::measurements::timer().synchronize_and_start("build_edge_array");
  std::vector<size_t> edge_dist(comm.size() + 1);
  std::vector<size_t> adjncy_sizes(comm.size());
  comm.allgather(kamping::send_buf(kagen_graph.adjncy.size()), kamping::recv_buf(adjncy_sizes));

  for (size_t i = 1; i < edge_dist.size(); ++i) {
    edge_dist[i] = edge_dist[i - 1] + adjncy_sizes[i - 1];
  }

  BlockDistribution edge_strategy = BlockDistribution(edge_dist);
  distributed::DistributedArray<size_t> edge_array = distributed::DistributedArray<size_t>(std::make_unique<BlockDistribution>(edge_strategy), comm);
  std::vector<size_t> recasted_edge_ids(kagen_graph.adjncy.size());
  for (size_t i = 0; i < recasted_edge_ids.size(); ++i) {
    recasted_edge_ids[i] = (size_t)kagen_graph.adjncy[i];
  }

  edge_array.initialize_local(recasted_edge_ids, comm.rank());
  kamping::measurements::timer().stop();

  // Setup vertex array
  kamping::measurements::timer().synchronize_and_start("build_vertex_array");
  std::vector<size_t> vertex_dist(comm.size() + 1);
  comm.allgather(kamping::send_buf(kagen_graph.vertex_range.first), kamping::recv_buf(vertex_dist));
  vertex_dist[vertex_dist.size() - 1] = kagen_graph.NumberOfGlobalVertices();

  BlockDistribution vertex_strategy = BlockDistribution(vertex_dist);
  distributed::DistributedArray<VertexEdgeMapping> vertex_array =
      distributed::DistributedArray<VertexEdgeMapping>(std::make_unique<BlockDistribution>(vertex_strategy), comm);

  std::vector<VertexEdgeMapping> local_vertex_array;
  for (size_t i = 0; i < kagen_graph.xadj.size() - 1; ++i) {
    local_vertex_array.emplace_back(edge_dist[comm.rank()] + kagen_graph.xadj[i], edge_dist[comm.rank()] + kagen_graph.xadj[i + 1]);
  }
  vertex_array.initialize_local(local_vertex_array, comm.rank());
  kamping::measurements::timer().stop();

  // Build graph
  kamping::measurements::timer().synchronize_and_start("build_csr_graph");
  DistributedCSRGraph graph =
      DistributedCSRGraph(std::move(vertex_array), std::move(edge_array), std::make_shared<BlockDistribution>(vertex_strategy));
  DistributedBFS bfs = DistributedBFS(std::make_unique<DistributedCSRGraph>(std::move(graph)), comm, bfs_start_vertices);
  kamping::measurements::timer().stop();
  
  // Run bfs
  kamping::measurements::timer().synchronize_and_start("run_bfs");
  bfs.run();
  kamping::measurements::timer().stop();
  kamping::measurements::timer().aggregate_and_print(kamping::measurements::SimpleJsonPrinter<>{std::cout});
  return 0;
}