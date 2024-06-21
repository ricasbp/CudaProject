# Floyd-Warshall algorithm

The Floyd-Warshall algorithm finds shortest paths in a directed weighted graph without negative cycles.


# Explanation of the improvement in a GPU dynamic programming

The Floyd-Warshall algorithm is a common method for finding the shortest paths between all pairs of vertices in a graph. It has a time complexity of O(∣V∣3)O(∣V∣3), where ∣V∣∣V∣ is the number of vertices. In the algorithm, the distance matrix dis is updated for each pair of vertices in every iteration (the k iterations). There are a lot of explanations on youtube about this algorithm.

Since the updates for each pair of vertices are independent of each other, they can be done simultaneously. This makes the algorithm a great candidate for running on a GPU, which can handle many operations at the same time.

We can move the two innermost loops of the algorithm to the GPU. These loops have a combined complexity of O(∣V∣2)O(∣V∣2). Running these loops on a GPU with many threads (say around 1000) effectively reduces the complexity to O(∣V∣2/∣P∣)O(∣V∣2/∣P∣), where ∣P∣∣P∣ is the number of threads. For a typical GPU with 1000 threads, this simplifies to O(∣V∣)O(∣V∣) for the inner loops, and thus O(∣V∣2)O(∣V∣2) for the entire algorithm.

In summary, using a GPU can significantly speed up the Floyd-Warshall algorithm by parallelizing the inner loops, making it much faster for large graphs.


# How to Run

1. Compile the CUDA code:

`nvcc tutorial04.cu -o tutorial04`

2. Run the CUDA code:

.\tutorial04