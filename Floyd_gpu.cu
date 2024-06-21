#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <limits.h>
#include <time.h>
#include <cuda.h>

#define INF INT_MAX

// Function to create a random weighted adjacency matrix
int** createAdjacencyMatrix(int vertices, float edgeProbability) {
    // Allocate memory for the adjacency matrix
    int** matrix = (int**)malloc(vertices * sizeof(int*));
    for (int i = 0; i < vertices; i++) {
        matrix[i] = (int*)malloc(vertices * sizeof(int));
    }

    // Seed the random number generator
    srand(time(NULL));

    // Populate the adjacency matrix with random values
    for (int i = 0; i < vertices; i++) {
        for (int j = 0; j < vertices; j++) {
            if (i != j && ((float)rand() / RAND_MAX) < edgeProbability) {
                matrix[i][j] = (rand() % 5) + 1; // Random weight between 1 and 5
            } else {
                matrix[i][j] = (i == j) ? 0 : INF; // No edge or self-loop
            }
        }
    }

    return matrix;
}

// Function to print the adjacency matrix
void printMatrix(int** matrix, int vertices) {
    for (int i = 0; i < vertices; i++) {
        for (int j = 0; j < vertices; j++) {
            if (matrix[i][j] == INF) {
                printf("INF ");
            } else {
                printf("%d ", matrix[i][j]);
            }
        }
        printf("\n");
    }
}

// Function to free the allocated memory for the adjacency matrix
void freeMatrix(int** matrix, int vertices) {
    for (int i = 0; i < vertices; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

// Function to perform Floyd-Warshall algorithm on CPU
void floydWarshallCPU(int** graph, int graphSize, int** result) {
    // Initialize result matrix with the same values as the graph's adjacency matrix
    for (int i = 0; i < graphSize; i++) {
        for (int j = 0; j < graphSize; j++) {
            result[i][j] = graph[i][j];
        }
    }

    // Apply Floyd-Warshall algorithm
    for (int k = 0; k < graphSize; k++) {
        for (int i = 0; i < graphSize; i++) {
            for (int j = 0; j < graphSize; j++) {
                if (result[i][k] != INF && result[k][j] != INF
                    && (result[i][j] > result[i][k] + result[k][j])) {
                    result[i][j] = result[i][k] + result[k][j];
                }
            }
        }
    }
}

// CUDA kernel for Floyd-Warshall algorithm
__global__ void floydWarshallKernel(int* dist, int k, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < n && j < n) {
        int ij = i * n + j;
        int ik = i * n + k;
        int kj = k * n + j;
        if (dist[ik] != INF && dist[kj] != INF && dist[ij] > dist[ik] + dist[kj]) {
            dist[ij] = dist[ik] + dist[kj];
        }
    }
}

// Function to perform Floyd-Warshall algorithm on GPU
void floydWarshallGPU(int** graph, int graphSize, int** result) {
    int* d_dist;
    int size = graphSize * graphSize * sizeof(int);

    // Allocate memory on the device
    cudaMalloc((void**)&d_dist, size);

    // Flatten the 2D graph matrix into a 1D array for device processing
    int* flatGraph = (int*)malloc(size);
    for (int i = 0; i < graphSize; i++) {
        for (int j = 0; j < graphSize; j++) {
            flatGraph[i * graphSize + j] = graph[i][j];
        }
    }

    // Copy the graph to device memory
    cudaMemcpy(d_dist, flatGraph, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((graphSize + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (graphSize + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Run the kernel for each intermediate vertex
    for (int k = 0; k < graphSize; k++) {
        floydWarshallKernel<<<numBlocks, threadsPerBlock>>>(d_dist, k, graphSize);
        cudaDeviceSynchronize();
    }

    // Copy the result back to host memory
    cudaMemcpy(flatGraph, d_dist, size, cudaMemcpyDeviceToHost);

    // Unflatten the 1D result array into the 2D result matrix
    for (int i = 0; i < graphSize; i++) {
        for (int j = 0; j < graphSize; j++) {
            result[i][j] = flatGraph[i * graphSize + j];
        }
    }

    // Free device memory
    cudaFree(d_dist);
    free(flatGraph);
}

int main() {
    int graphSize;
    float edgeProbability;

    // Get the number of vertices and the edge probability from the user
    printf("Enter the number of vertices: ");
    scanf("%d", &graphSize);
    printf("Enter the edge probability (0.0 to 1.0): ");
    scanf("%f", &edgeProbability);

    // Create the adjacency matrix
    int** graphMatrix = createAdjacencyMatrix(graphSize, edgeProbability);
    assert(graphMatrix);

    // Print the adjacency matrix
    printf("Adjacency Matrix of the Graph Created: \n");
    printMatrix(graphMatrix, graphSize);

    // Allocate memory for cpu_matrix_result (assuming it's a 2D array of size graphSize x graphSize)
    int** cpu_matrix_result = (int**)malloc(graphSize * sizeof(int*));
    for (int i = 0; i < graphSize; i++) {
        cpu_matrix_result[i] = (int*)malloc(graphSize * sizeof(int));
    }
    assert(cpu_matrix_result);

    // Measure the time taken by the CPU implementation
    clock_t startCPU = clock();
    floydWarshallCPU(graphMatrix, graphSize, cpu_matrix_result);
    clock_t endCPU = clock();
    double cpu_time_used = ((double)(endCPU - startCPU)) / CLOCKS_PER_SEC;

    // Print the result matrix from CPU
    printf("Shortest Paths Matrix (Floyd-Warshall Algorithm on CPU):\n");
    printMatrix(cpu_matrix_result, graphSize);

    // Allocate memory for gpu_matrix_result (assuming it's a 2D array of size graphSize x graphSize)
    int** gpu_matrix_result = (int**)malloc(graphSize * sizeof(int*));
    for (int i = 0; i < graphSize; i++) {
        gpu_matrix_result[i] = (int*)malloc(graphSize * sizeof(int));
    }
    assert(gpu_matrix_result);

    // Measure the time taken by the GPU implementation
    cudaEvent_t startGPU, stopGPU;
    cudaEventCreate(&startGPU);
    cudaEventCreate(&stopGPU);

    cudaEventRecord(startGPU);
    floydWarshallGPU(graphMatrix, graphSize, gpu_matrix_result);
    cudaEventRecord(stopGPU);
    cudaEventSynchronize(stopGPU);

    float gpu_time_used;
    cudaEventElapsedTime(&gpu_time_used, startGPU, stopGPU);

    // Print the result matrix from GPU
    printf("Shortest Paths Matrix (Floyd-Warshall Algorithm on GPU):\n");
    printMatrix(gpu_matrix_result, graphSize);

    // Print the time taken by both implementations
    printf("Time taken by CPU: %f seconds\n", cpu_time_used);
    printf("Time taken by GPU: %f milliseconds\n", gpu_time_used);

    // Free the allocated memory
    freeMatrix(graphMatrix, graphSize);
    freeMatrix(cpu_matrix_result, graphSize);
    freeMatrix(gpu_matrix_result, graphSize);

    cudaEventDestroy(startGPU);
    cudaEventDestroy(stopGPU);

    return 0;
}
