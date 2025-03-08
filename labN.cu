/*
 * Author: Miras Khamit
 * Class: CSS 535
 *
 * Usage:
 *  
 *   ncu --set full ./a.out  <ArraySize> <ThreadSize> <Mode> > ...txt
 * Example:
 *   ncu --set full ./a.out  10000 32 1 > naive.txt
 *
 * Compilation:
 *     nvcc labN.cu
 */
#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>


__global__ void gemv_naive(const double* A, const double* x, double* y, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        double sum = 0.0;
        for (int j = 0; j < cols; j++) {
            sum += A[row * cols + j] * x[j];
        }
        y[row] = sum;
    }
}

__global__ void gemv_global_mem_opt(const double* A, const double* x, double* y, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        double sum = 0.0;
        // Unrolling the loop to improve efficiency when accessing global memory.
        // Accessing global memory in chunks of 4 elements per iteration.
        for (int j = 0; j < cols; j += 4) {
            sum += A[row * cols + j] * x[j];
            if (j + 1 < cols) sum += A[row * cols + j + 1] * x[j + 1];
            if (j + 2 < cols) sum += A[row * cols + j + 2] * x[j + 2];
            if (j + 3 < cols) sum += A[row * cols + j + 3] * x[j + 3];
        }
        y[row] = sum;
    }
    // Global memory optimization strategy:
    // 1. We utilize loop unrolling to improve memory access efficiency, minimizing the number of global memory accesses.
    // 2. Each thread accesses 4 elements at a time (if available) in order to maximize memory throughput.
    // 3. Make sure that threads access contiguous memory locations for better memory coalescing.
}

__global__ void gemv_shared_mem(const double* A, const double* x, double* y, int rows, int cols) {
    __shared__ double x_shared[512];  // Shared memory for the vector x
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows) {
        double sum = 0.0;
        // Load chunks of vector 'x' into shared memory in a coalesced manner.
        for (int j = 0; j < cols; j += blockDim.x) {
            if (threadIdx.x + j < cols) {
                x_shared[threadIdx.x] = x[threadIdx.x + j];
            }
            __syncthreads();
            
            // Perform the multiplication for the block's rows and columns using shared memory.
            for (int k = 0; k < blockDim.x && (j + k) < cols; k++) {
                sum += A[row * cols + j + k] * x_shared[k];
            }
            __syncthreads();
        }
        y[row] = sum;
    }
    // Shared memory optimization strategy:
    // 1. Vector `x` is loaded into shared memory for fast access by the threads within a block.
    // 2. This eliminates redundant global memory reads and improves performance by utilizing fast shared memory.
    // 3. Threads within the block work together to load the required portion of `x`, reducing memory latency.
    // 4. Synchronization (with __syncthreads()) ensures that all threads have loaded their data before performing computations.
}

__global__ void gemv_register_opt(const double* A, const double* x, double* y, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        double sum = 0.0f;

        // Loop unrolling to increase the number of operations per thread and improve register usage.
        int j;
        for (j = 0; j <= cols - 4; j += 4) {
            // Each thread uses registers to hold intermediate results, reducing the need for slow global memory accesses.
            sum += A[row * cols + j] * x[j];
            sum += A[row * cols + j + 1] * x[j + 1];
            sum += A[row * cols + j + 2] * x[j + 2];
            sum += A[row * cols + j + 3] * x[j + 3];
        }
        
        // Handle remaining elements if cols is not a multiple of 4
        for (; j < cols; ++j) {
            sum += A[row * cols + j] * x[j];
        }

        y[row] = sum;
    }
    // Register optimization strategy:
    // 1. Loop unrolling reduces the number of iterations, increasing the instruction-level parallelism.
    // 2. By storing intermediate results in registers, we avoid costly global memory accesses.
    // 3. Register usage is particularly beneficial for small to medium-sized matrices as it eliminates memory bottlenecks.
    // 4. Optimizing with register usage helps to improve thread execution speed, minimizing register spilling and maximizing performance.
}

void run_kernel(void (*kernel)(const double*, const double*, double*, int, int), const char* name, int array_size, int block_size) {
    double *A, *x, *y;
    double *d_A, *d_x, *d_y;
    
    cudaMallocHost(&A, array_size * array_size * sizeof(double));
    cudaMallocHost(&x, array_size * sizeof(double));
    cudaMallocHost(&y, array_size * sizeof(double));
    
    cudaMalloc(&d_A, array_size * array_size * sizeof(double));
    cudaMalloc(&d_x, array_size * sizeof(double));
    cudaMalloc(&d_y, array_size * sizeof(double));
    
    for (int i = 0; i < array_size * array_size; i++) A[i] = static_cast<double>(rand()) / RAND_MAX;
    for (int i = 0; i < array_size; i++) x[i] = static_cast<double>(rand()) / RAND_MAX;
    
    cudaMemcpy(d_A, A, array_size * array_size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, array_size * sizeof(double), cudaMemcpyHostToDevice);
    
    dim3 blockSize(block_size);
    dim3 gridSize((array_size + block_size - 1) / block_size);
    
    auto start = std::chrono::high_resolution_clock::now();
    kernel<<<gridSize, blockSize>>>(d_A, d_x, d_y, array_size, array_size);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    
    cudaMemcpy(y, d_y, array_size * sizeof(double), cudaMemcpyDeviceToHost);
    std::chrono::duration<double, std::milli> duration = end - start;
    
    std::cout << "Execution time of " << name << " with array_size=" << array_size << " and block_size=" << block_size << ": " << duration.count() << " ms" << std::endl;
    
    cudaFreeHost(A); cudaFreeHost(x); cudaFreeHost(y);
    cudaFree(d_A); cudaFree(d_x); cudaFree(d_y);
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <ArraySize> <ThreadSize> <Mode>" << std::endl;
        return 1;
    }
    int array_size = std::atoi(argv[1]);
    int block_size = std::atoi(argv[2]);
    int mode = std::atoi(argv[3]);
    switch (mode) {
        case 1:
            run_kernel(gemv_naive, "Naive GEMV", array_size, block_size);
            break;
        case 2:
            run_kernel(gemv_shared_mem, "Shared Memory Optimized GEMV", array_size, block_size);
            break;
        case 3:
            run_kernel(gemv_global_mem_opt, "Global Memory Optimized GEMV", array_size, block_size);
            break;
        case 4:
            run_kernel(gemv_register_opt, "Register Optimized GEMV", array_size, block_size);
            break;
        default:
            std::cerr << "Invalid Mode!" << std::endl;
            return 1;
    }
    return 0;
}