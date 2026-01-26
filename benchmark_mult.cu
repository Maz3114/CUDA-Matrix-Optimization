#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <stdio.h>

#define TILE_WIDTH 16

// Macro for error checking
#define CHECK(call) \
{ \
    const cudaError_t error = call; \
    if (error != cudaSuccess) \
    { \
        printf("Error: %s:%d, ", __FILE__, __LINE__); \
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1); \
    } \
}

// Your Tiled Kernel
__global__ void matrixMulTiled(float *A, float *B, float *C, int width) {
    int Row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int Col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    float Cvalue = 0.0;

    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    for (int ph = 0; ph < width / TILE_WIDTH; ++ph) {
        As[threadIdx.y][threadIdx.x] = A[Row * width + (ph * TILE_WIDTH + threadIdx.x)];
        Bs[threadIdx.y][threadIdx.x] = B[(ph * TILE_WIDTH + threadIdx.y) * width + Col];
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            Cvalue += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }
    C[Row * width + Col] = Cvalue;
}

int main() {
    int width = 2048; 
    int bytes = width * width * sizeof(float);
    
    printf("Initializing Data...\n");
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);
    float *h_C_ref = (float*)malloc(bytes);

    for(int i=0; i<width*width; i++) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }

    float *d_A, *d_B, *d_C;
    CHECK(cudaMalloc(&d_A, bytes));
    CHECK(cudaMalloc(&d_B, bytes));
    CHECK(cudaMalloc(&d_C, bytes));

    CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    // 1. RUN YOUR KERNEL
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid(width/TILE_WIDTH, width/TILE_WIDTH);

    printf("Running YOUR Kernel (Size %d x %d)...\n", width, width);
    CHECK(cudaEventRecord(start));
    matrixMulTiled<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, width);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    
    // Check for Kernel Launch Errors
    CHECK(cudaGetLastError());

    float myTime = 0;
    CHECK(cudaEventElapsedTime(&myTime, start, stop));
    printf("Your Time:   %f ms\n", myTime);

    // 2. RUN NVIDIA cuBLAS
    cublasHandle_t handle;
    cublasCreate(&handle); // Note: Should check status here too, but skipping for brevity
    const float alpha = 1.0f;
    const float beta  = 0.0f;

    printf("Running NVIDIA cuBLAS...\n");
    CHECK(cudaEventRecord(start));
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, width, width, width, 
                &alpha, d_B, width, d_A, width, &beta, d_C, width);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    
    float cublasTime = 0;
    CHECK(cudaEventElapsedTime(&cublasTime, start, stop));
    printf("cuBLAS Time: %f ms\n", cublasTime);

    printf("Slowdown: %.2fx slower than NVIDIA\n", myTime / cublasTime);

    cublasDestroy(handle);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C); free(h_C_ref);
    return 0;
}
