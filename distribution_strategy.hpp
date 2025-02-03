#pragma once

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

class DistributionStrategy {
public:
  virtual ~DistributionStrategy() = default;

  virtual size_t owner(size_t global_index) const = 0;

  virtual size_t local_size(int rank) const = 0;

  virtual size_t to_local_index(int rank, size_t global_index) const = 0;

  virtual size_t to_global_index(int rank, size_t local_index) const = 0;

  virtual size_t total_elements() const = 0;
};

class BlockDistribution : public DistributionStrategy {
public:
  BlockDistribution(std::vector<size_t> distribution) : _distribution(std::move(distribution)) {}

  // Worker with rank k stores the indices between _distribution[rank] (inclusive) and _distribution[rank + 1]
  // (exclusive)
  size_t owner(size_t global_index) const override {
    auto it = std::upper_bound(_distribution.begin(), _distribution.end(), global_index);
    return std::distance(_distribution.begin(), it) - 1;
  }

  size_t local_size(int rank) const override {
    return _distribution[rank + 1] - _distribution[rank];
  }

  size_t to_local_index(int rank, size_t global_index) const override { 
    std::cout << "_rank: " << rank << " global_index: " << global_index << " _dist[rank]: " << _distribution[rank] << '\n';
    return global_index - _distribution[rank]; }

  size_t to_global_index(int rank, size_t local_index) const override { return local_index + _distribution[rank]; }

  size_t total_elements() const override { return _distribution.back(); }

private:
  std::vector<size_t> _distribution;
};

class RoundRobinDistribution : public DistributionStrategy {
public:
  RoundRobinDistribution(size_t total_elements, int num_ranks) : _total_elements(total_elements), _num_ranks(num_ranks) {}

  size_t owner(size_t global_index) const override { return global_index % _num_ranks; }

  size_t local_size(int rank) const override {
    size_t local_elements = _total_elements / _num_ranks;
    if (rank < _total_elements % _num_ranks) {
      ++local_elements;
    }

    return local_elements;
  }

  size_t to_local_index(int rank, size_t global_index) const override {
    if (owner(global_index) != rank) {
      throw std::invalid_argument("Not owned by rank");
    }
    return global_index / _num_ranks;
  }

  size_t to_global_index(int rank, size_t local_index) const override { return local_index * _num_ranks + rank; }

  size_t total_elements() const override { return _total_elements; }

private:
  size_t _total_elements;
  int _num_ranks;
};