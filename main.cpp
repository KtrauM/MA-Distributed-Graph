#include <memory>
#include <kamping/communicator.hpp>
#include <kamping/collectives/allreduce.hpp>
#include <kamping/collectives/alltoall.hpp>
#include <kamping/collectives/reduce.hpp>
#include "distribution_strategy.hpp"
#include "distributed_array.hpp"

int main(int argc, char** argv) {
    kamping::Environment env(argc, argv);
    auto comm = kamping::comm_world();
    constexpr size_t total_elements = 125;
    
    // Block distribution example
    std::vector<size_t> block_dist = {0, 33, 66, 100, total_elements};
    BlockDistribution block_strat = BlockDistribution(block_dist);
    distributed::DistributedArray<int> block_arr(std::make_unique<BlockDistribution>(block_strat), comm);

    // Set values
    if (comm.rank() == 0) {
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

    // Round robin distribution example
    // RoundRobinDistribution cyclic_strat = RoundRobinDistribution(total_elements, comm.size());
    // distributed::DistributedArray<int> cyclic_arr(std::make_unique<RoundRobinDistribution>(cyclic_strat), comm);

    // // Set values
    // if (comm.rank() == 0) {
    //     cyclic_arr.set(0, 42);    // Sent to rank 0
    //     cyclic_arr.set(1, 42);    // Sent to rank 1
    //     cyclic_arr.set(2, 50);    // Sent to rank 2
    //     cyclic_arr.set(3, 50);    // Sent to rank 3
    // }

    // // Exchange updates
    // cyclic_arr.exchange();
    // cyclic_arr.print_local();

    // // Gather results
    // auto global_block_cyclic = cyclic_arr.gather(0);
    // if (comm.rank() == 0) {
    //     std::cout << "Global array: " << global_block_cyclic.size() << "\n";
    //     for (auto const x: global_block_cyclic) {
    //         std::cout << x << ' ';
    //     }
    //     std::cout << '\n';
    // }


    return 0;
}