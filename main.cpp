#include <memory>
#include <vector>
#include <kagen.h>
#include <kamping/communicator.hpp>
#include <kamping/collectives/allreduce.hpp>
#include <kamping/collectives/alltoall.hpp>
#include <kamping/collectives/allgather.hpp>
#include <kamping/collectives/reduce.hpp>
#include "distribution_strategy.hpp"
#include "distributed_array.hpp"
#include "distributed_bfs.hpp"

 #include <mpi.h>
#include <iostream>

void PrintGraphAttributes(const kagen::Graph& graph) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get the rank of the process

    std::cout << "[PE " << rank << "] Vertex Range: (" << graph.vertex_range.first << ", " << graph.vertex_range.second << ")\n";
    std::cout << "[PE " << rank << "] Graph Representation: " << static_cast<int>(graph.representation) << "\n";
    MPI_Barrier(MPI_COMM_WORLD);

    std::cout << "[PE " << rank << "] Edges (Edgelist):\n";
    for (const auto& edge : graph.edges) {
        std::cout << "[PE " << rank << "] (" << edge.first << ", " << edge.second << ")\n";
    }
    MPI_Barrier(MPI_COMM_WORLD);

    std::cout << "[PE " << rank << "] XadjArray:\n";
    for (const auto& val : graph.xadj) {
        std::cout << "[PE " << rank << "] " << val << " ";
    }
    std::cout << "\n";
    MPI_Barrier(MPI_COMM_WORLD);

    std::cout << "[PE " << rank << "] AdjncyArray:\n";
    for (const auto& val : graph.adjncy) {
        std::cout << "[PE " << rank << "] " << val << " ";
    }
    std::cout << "\n";
    MPI_Barrier(MPI_COMM_WORLD);

    std::cout << "[PE " << rank << "] Vertex Weights:\n";
    for (const auto& weight : graph.vertex_weights) {
        std::cout << "[PE " << rank << "] " << weight << " ";
    }
    std::cout << "\n";
    MPI_Barrier(MPI_COMM_WORLD);

    std::cout << "[PE " << rank << "] Edge Weights:\n";
    for (const auto& weight : graph.edge_weights) {
        std::cout << "[PE " << rank << "] " << weight << " ";
    }
    std::cout << "\n";
    MPI_Barrier(MPI_COMM_WORLD);

    std::cout << "[PE " << rank << "] Coordinates (2D):\n";
    for (const auto& coord : graph.coordinates.first) {
        std::cout << "[PE " << rank << "] (" << std::get<0>(coord) << ", " << std::get<1>(coord) << ")\n";
    }
    MPI_Barrier(MPI_COMM_WORLD);

    std::cout << "[PE " << rank << "] Coordinates (3D):\n";
    for (const auto& coord : graph.coordinates.second) {
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


int main(int argc, char** argv) {
    kamping::Environment env(argc, argv);
    auto comm = kamping::comm_world();
    constexpr size_t total_elements = 125;
    int num_ranks = comm.size();
    // Block distribution example
    std::vector<size_t> block_dist = {0, 33, 66, 100, total_elements};
    BlockDistribution block_strat = BlockDistribution(block_dist);
    distributed::DistributedArray<int> block_arr(std::make_unique<BlockDistribution>(block_strat), comm);

    // Set values
    if (comm.rank() == 0) {
        std::cout << "Block distribution example\n";
        block_arr.set(0, 42);      // Sent to rank 0
        block_arr.set(32, 42);
        block_arr.set(33, 50);     // Sent to rank 1
        block_arr.set(65, 50);
        block_arr.set(66, 86);     // Sent to rank 2
        block_arr.set(99, 86);
        block_arr.set(100, 99);    // Sent to rank 3
        block_arr.set(124, 99);
        block_arr.set(67, 87);     // Sent to rank 2
        block_arr.set(98, 87);
        block_arr.set(34, 51);     // Sent to rank 1
        block_arr.set(64, 51);
        block_arr.set(101, 100);   // Sent to rank 3
        block_arr.set(123, 100);
    }
    // Exchange updates
    block_arr.exchange();
    block_arr.print_local();

    // Gather results
    auto global_block = block_arr.gather(0);
    if (comm.rank() == 0) {
        std::cout << "Global array: " << global_block.size() << "\n";
        for (auto const x: global_block) {
            std::cout << x << ' ';
        }
        std::cout << '\n';
    }

    // Round-robin distribution example
    RoundRobinDistribution cyclic_strat = RoundRobinDistribution(total_elements, num_ranks);
    distributed::DistributedArray<int> cyclic_arr(std::make_unique<RoundRobinDistribution>(cyclic_strat), comm);

    // Set values
    if (comm.rank() == 0) {
        std::cout << "\nRound-robin distribution example\n";
        for (int32_t i = 0; i < total_elements; ++i) {
            cyclic_arr.set(i, i % num_ranks + 42);
        }
    }

    // Exchange updates
    cyclic_arr.exchange();
    cyclic_arr.print_local();

    // Gather results
    auto global_block_cyclic = cyclic_arr.gather(0);
    if (comm.rank() == 0) {
        std::cout << "Global array: " << global_block_cyclic.size() << "\n";
        for (auto const x: global_block_cyclic) {
            std::cout << x << ' ';
        }
        std::cout << '\n';
    }


    // BFS example
    kagen::KaGen gen(MPI_COMM_WORLD);
    gen.UseCSRRepresentation();
    auto kagen_graph = gen.GenerateGrid2D_NM(32, 12);

    PrintGraphAttributes(kagen_graph);
    

    // Setup edge array
    std::vector<size_t> edge_dist(comm.size() + 1);
    std::vector<size_t> adjncy_sizes(comm.size());
    comm.allgather(kamping::send_buf(kagen_graph.adjncy.size()), kamping::recv_buf(adjncy_sizes));

    for (size_t i = 1; i < edge_dist.size(); ++i) {
        edge_dist[i] = edge_dist[i - 1] + adjncy_sizes[i - 1];
    }

    for (auto x: edge_dist) {
        std::cout<< "EDGE DSIT " << x << '\n';
    }

    BlockDistribution edge_strategy = BlockDistribution(edge_dist);
    distributed::DistributedArray<size_t> edge_array = distributed::DistributedArray<size_t>(std::make_unique<BlockDistribution>(edge_strategy), comm);
    std::vector<size_t> recasted_edge_ids(kagen_graph.adjncy.size());
    for (size_t i = 0; i < recasted_edge_ids.size(); ++i) {
        recasted_edge_ids[i] = (size_t) kagen_graph.adjncy[i];
    }

    edge_array.initialize_local(recasted_edge_ids, comm.rank());
    edge_array.print_local();


    // Setup vertex array
    std::vector<size_t> vertex_dist(comm.size() + 1);
    comm.allgather(kamping::send_buf(kagen_graph.vertex_range.first), kamping::recv_buf(vertex_dist));
    vertex_dist[vertex_dist.size() - 1] = kagen_graph.NumberOfGlobalVertices();

    for (auto x: vertex_dist) {
        std::cout<< "Vertex DSIT " << x << '\n';
    }

    BlockDistribution vertex_strategy = BlockDistribution(vertex_dist);
    distributed::DistributedArray<VertexEdgeMapping> vertex_array = distributed::DistributedArray<VertexEdgeMapping>(std::make_unique<BlockDistribution>(vertex_strategy), comm);
    
    std::vector<VertexEdgeMapping> local_vertex_array;
    for (size_t i = 0; i < kagen_graph.xadj.size() - 1; ++i) {
        local_vertex_array.emplace_back(edge_dist[comm.rank()] + kagen_graph.xadj[i], edge_dist[comm.rank()] + kagen_graph.xadj[i + 1]);
    }
    vertex_array.initialize_local(local_vertex_array, comm.rank());

    vertex_array.print_local_vertex();



    // Build graph
    DistributedCSRGraph graph = DistributedCSRGraph(std::move(vertex_array), std::move(edge_array), std::make_shared<BlockDistribution>(vertex_strategy));
    DistributedBFS bfs = DistributedBFS(std::make_unique<DistributedCSRGraph>(std::move(graph)), comm, std::vector<size_t>(1, 0));

    // TODO: benchmark random geographic graphs
    // TODO: add options to specify graphs from command line

    bfs.run();
    // for (const auto& vertex_distance : bfs.getDistances()) {
    //     std::cout << "Vertex: " << vertex_distance.first << ", Distance: " << vertex_distance.second << std::endl;
    // }
    
    return 0;
}