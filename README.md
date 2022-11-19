# CUDA Projects

Graph Algorithms for popular graph questions have been implemented and executed using CUDA programs on GPUs.
The programs were timed usimg timers for both GPU and CPU time and the corresponding performances were observed.

> NOTE: Update the absolute path in this [bash script](run.sh) to match your system before running the following commands.

The assignment consists of the following 3 programs:

1. **Parallel BFS** - Program to perform Parallel BFS on a graph given in CSR format.

- Output is the levels of each node in the graph with root vertex at level 0.
- To execute, run the following command on a system with CUDA support enabled GPUs

```console
$ bash run.sh cuda_bfs.cu
```

- Expected Output for the given hardcoded input is:

```
Vertex Distance
   1        0
   2        1
   3        3
   4        2
   5        2
   6        3
   7        3
   8        1
   9        3
  10        1
  11        1
CPU Time Taken: 4.731000 ms
GPU Time Taken: 2.928640 ms
```

2. **Cut Vertices** - Program to obtain the cut vertices of a graph given in CSR format.

- Output is the list of cut vertices.

* To execute, run the following command on a system with CUDA support enabled GPUs

```console
$ bash run.sh cuda_cut_vertex.cu
```

- Expected Output for the given hardcoded input is:

```
Cut vertices: 2 3 5 7 10 14
CPU Time Taken: 3.650000 ms
GPU Time Taken: 1.169408 ms
```

3. **Biconnected Components** - Program to obtain the biconnected components of a graph given in CSR format.

- Output is the list of components identified by assigning a uniquer number to all edges in a component.

- To execute, run the following command on a system with CUDA support enabled GPUs

```console
$ bash run.sh cuda_bcc.cu
```

- Expected Output for the given hardcoded input is:

```
Edge    BCC
1-0     -1
1-2     -1
2-0     -1
3-2      3
4-2     -1
4-5     -1
5-1     -1
5-2     -1
6-7      6
6-5      6
7-5      6
8-7      8
8-10      8
9-10      9
10-7      8
11-12     11
11-3     11
12-3     11
13-11     11
13-12     11
13-14     11
14-12     11
15-14     15
15-16     15
16-14     15
CPU Time Taken: 5.898000 ms
GPU Time Taken: 2.012032 ms
```

Note: For the corner case of the component that contains the root vertex, we handle it separately and assign the ID -1 to it. For all other components the ID is a non-negative integer.
Additional Test Cases for bccs is available in [this file](test_cases.txt)

## Authors

- **CS19B037** Shreetesh M

- **CS19B012** Debeshee Das
