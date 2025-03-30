# Distributed Grarrph
A distributed graph processing framework that takes a novel approach by implementing graph algorithms using distributed array and set operations rather than traditional vertex-centric paradigms.

## Overview
Distributed Grarrph provides high-level abstractions for distributed graph processing by leveraging distributed array and set primitives. Instead of thinking in terms of vertices and edges, algorithms are expressed through common distributed array/set operations.

This approach offers several benefits:
- Cleaner algorithm implementations that focus on the core logic rather than distribution details
- Reusable distributed data structure primitives that can be composed to build complex algorithms
- Natural expression of graph algorithms in terms of bulk operations on collections
- Abstraction from low-level distributed memory management and communication

## Features
- Distributed array and set data structures with common operations
- Implementation of standard graph algorithms like BFS and Connected Components
- Integration with KaGen for distributed graph generation
- Benchmarking infrastructure to evaluate scaling performance against other frameworks, i.e HavoqGT and CombBLAS


### Note on Edge Count
KaGen generates edge lists with both forward and backward edges, so the actual edge list length is 2m where m is the specified number of edges.