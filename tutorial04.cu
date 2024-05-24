#include <cuda.h>
#include <stdio.h>
#include <iostream>

using namespace std;

/* Our Kernel, that was run like:    
    AddIntsCUDA<<<count / 256 + 1, 256>>>(d_a, d_b, count);
*/
__global__ void AddIntsCUDA(int* a, int *b, int count){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < count){  //Make sure the ID's that are calculated outside the array, are not used
        a[id] += b[id];
    }
}

int main(){

    // Reserve Space for the arrays in the CPU
    srand(time(NULL));
    int count = 100;
    int *h_a = new int [count];
    int *h_b = new int [count];

    for(int i = 0; i < count; i++){
        h_a[i] = rand() % 1000;
        h_b[i] = rand() % 1000;
    }

    cout << "prior to addition:" <<endl;
    for(int i = 0; i < 5; i++){
        cout << h_a[i] << " " << h_b[i] << endl;
    }

    int *d_a, *d_b;

    // Reserve Space for the GPU arrays
    if(cudaMalloc(&d_a, sizeof(int) * count) != cudaSuccess){
        cout<<"Error Dummie! for a";
        return 0;
    }
    if(cudaMalloc(&d_b, sizeof(int) * count) != cudaSuccess){
        cout<<"Error Dummie! for b";
        cudaFree(d_a);
        return 0;
    }
    
    // Put the values in the arrays
    //d_a: Destination, h_a: pointer to source, its a intenger, direction: host to device

    if(cudaMemcpy(d_a, h_a, sizeof(int) * count, cudaMemcpyHostToDevice) != cudaSuccess){
        cout<<"Error Dummie! for memCpy";
        cudaFree(d_a);
        cudaFree(d_b);
        return 0;
    }

    if(cudaMemcpy(d_b, h_b, sizeof(int) * count, cudaMemcpyHostToDevice) != cudaSuccess){
        cout<<"Error Dummie! for memCpy";
        cudaFree(d_a);
        cudaFree(d_b);
        return 0;
    }

    // Launching Blocks to work with whatever array size (our count) has declared.
    // 100 / 256 = 0 (because of integer) + 1 = 1.
    // Ricas: I'm assuming the reasoning is:
    //        if we had 1000 array size, we would want to use multiple blocks, instead of just 1.
    AddIntsCUDA<<<count / 256 + 1, 256>>>(d_a, d_b, count);


    if(cudaMemcpy(h_a, d_a, sizeof(int) * count, cudaMemcpyDeviceToHost) != cudaSuccess){
        cout<<"Error Dummie! for memCpy2";
        delete[] h_a;
        delete[] h_b;
        cudaFree(d_a);
        cudaFree(d_b);
        return 0;
    }

    if(cudaMemcpy(h_b, d_b, sizeof(int) * count, cudaMemcpyDeviceToHost) != cudaSuccess){
        cout<<"Error Dummie! for memCpy2";
        delete[] h_a;
        delete[] h_b;
        cudaFree(d_a);
        cudaFree(d_b);
        return 0;
    }

    // Print the new vallues on our CPU array;
    for(int i = 0; i < 5; i++){
        cout<< "Final array is: " << h_a[i] << endl;
    }

    cudaFree(d_a);
    cudaFree(d_b);

    delete[] h_a;
    delete[] h_b;

    return 0;
}




