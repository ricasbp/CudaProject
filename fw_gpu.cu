#include <assert.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <winsock2.h>
#include "workshop.h"

#define GRAPH_SIZE 1000
#define EDGE_COST(graph, graph_size, a, b) graph[a * graph_size + b]
#define D(a, b) EDGE_COST(output, graph_size, a, b)

//#define THREADS_PER_BLOCK 256

#define INF 0x1fffffff

void generate_random_graph(int *output, int graph_size) {
  int i, j;

  srand(0xdadadada);

  for (i = 0; i < graph_size; i++) {
    for (j = 0; j < graph_size; j++) {
      if (i == j) {
        D(i, j) = 0;
      } else {
        int r;
        r = rand() % 40;
        if (r > 20) {
          r = INF;
        }

        D(i, j) = r;
      }
    }
  }
}

__global__ void fw_kernel(int *output, int graph_size, int k) {
  
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;

  if (j < graph_size && i < graph_size) {

    int s = D(i, k) + D(k, j);
    D(i, j) = s * (s < D(i, j)) + D(i, j) * (s >= D(i, j));
  }
}

void floyd_warshall_gpu(const int *graph, int graph_size, int *output) {

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);

  int threadsPerBlock = deviceProp.maxThreadsPerBlock;
  int size = sizeof(int) * graph_size * graph_size;
  int *dev_output;

  // Create a gridDim with 2 dimensions
  dim3 gridDim ((graph_size + threadsPerBlock - 1) / threadsPerBlock, graph_size);

  // -------------------------------------------------------------------------------

  // Allocate memory in GPU
  HANDLE_ERROR( cudaMalloc(&dev_output, size));

  // Copy graph from CPU to GPU
  cudaMemcpy(dev_output, graph, size, cudaMemcpyHostToDevice);

  // Calls to kernel
  for (int k = 0; k < graph_size; k++) {
    fw_kernel<<<gridDim, threadsPerBlock>>>(dev_output, graph_size, k);
  }

  // Copy results from GPU to CPU
  cudaMemcpy(output, dev_output, size, cudaMemcpyDeviceToHost);

  // Free allocated GPU memory
  cudaFree(dev_output);

}

void floyd_warshall_cpu(const int *graph, int graph_size, int *output) {
  int i, j, k;

  memcpy(output, graph, sizeof(int) * graph_size * graph_size);

  for (k = 0; k < graph_size; k++) {
    for (i = 0; i < graph_size; i++) {
      for (j = 0; j < graph_size; j++) {
        if (D(i, k) + D(k, j) < D(i, j)) {
          D(i, j) = D(i, k) + D(k, j);
        }
      }
    }
  }
}

void print_matrix(int v[GRAPH_SIZE])
{
    int c, d;
    for(c = 0; c < GRAPH_SIZE; c++){
      for (d = 0; d < GRAPH_SIZE; d++) { 
        if(v[c * GRAPH_SIZE + d] == INF) {
          printf("? ");
        } else {
          printf("%d ", v[c * GRAPH_SIZE + d]);
        }
      }
      printf("\n");
    }

    printf("\n");
}

int main(int argc, char **argv) {
  LARGE_INTEGER li;
  __int64 CounterStart = 0;
  if (!QueryPerformanceFrequency(&li))
    printf("QueryPerformanceFrequency failed\n");
  double PCFreq = (double) li.QuadPart;

  int *graph, *output_cpu, *output_gpu;
  int size;

  size = sizeof(int) * GRAPH_SIZE * GRAPH_SIZE;

  graph = (int *)malloc(size);
  assert(graph);

  output_cpu = (int *)malloc(size);
  assert(output_cpu);
  memset(output_cpu, 0, size);

  output_gpu = (int *)malloc(size);
  assert(output_gpu);

  generate_random_graph(graph, GRAPH_SIZE);

  //print_matrix(graph);

  double* resultsCPU = new double[1000];
  double* resultsGPU = new double[1000];
  int n = 1;

  fprintf(stderr, "running on cpu...\n");

  for(int i = 0; i < n; i++) {
    // start time
    QueryPerformanceCounter(&li);
	  CounterStart = li.QuadPart;

    floyd_warshall_cpu(graph, GRAPH_SIZE, output_cpu);

    //print_matrix(output_cpu);

    // finish time
	  QueryPerformanceCounter(&li);
    double time = ((double) (li.QuadPart - CounterStart)) / PCFreq;
	  //printf("Elapsed time: %.6f s\n", time);  
    resultsCPU[i] = time;
  }

  fprintf(stderr, "running on gpu...\n");
  
  for(int i = 0; i < n; i++) {
    // start time
    QueryPerformanceCounter(&li);
    CounterStart = li.QuadPart;

    floyd_warshall_gpu(graph, GRAPH_SIZE, output_gpu);

    //print_matrix(output_gpu);

    // finish time
    QueryPerformanceCounter(&li);
    double time = ((double) (li.QuadPart - CounterStart)) / PCFreq;
	  //printf("Elapsed time: %.6f s\n", time);  
    resultsGPU[i] = time;
  }

  double sumCPU = 0, sumGPU = 0;
  for(int i = 0; i < n; i++) {
    sumCPU += resultsCPU[i];
    sumGPU += resultsGPU[i];
  }
  sumCPU /= n;
  sumGPU /= n;

  printf("\nGraph size: %d\n", GRAPH_SIZE);
  printf("Average CPU: %.3f\n", sumCPU);
  printf("Average GPU: %.3f\n", sumGPU);
  printf("Speedup: %.2f\n", sumCPU / sumGPU);
        
  if (memcmp(output_cpu, output_gpu, size) != 0) {
    fprintf(stderr, "FAIL!\n");
  }

  return 0;
}
