#CUDA MM RECTANGULAR MATRICES
%%cu
#include <stdio.h>
#include <cuda_runtime.h>

#define BLOCKS 32 // block size for matrix multiplication

__global__ void MatrixMulKernel(float* M, float* N, float* P, int NumRowsM, int NumColsM, int NumColsN) {
    // Calculate the row index of the P element and M
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    // Calculate the column index of P and N
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    if (Row < NumRowsM && Col < NumColsN) {
        float Pvalue = 0;
        // each thread computes one element of the block sub-matrix
        for (int k = 0; k < NumColsM; ++k) {
            Pvalue += M[Row * NumColsM + k] * N[k * NumColsN + Col];
        }
        P[Row * NumColsN + Col] = Pvalue;
    }
}

void MatrixMultiplication(float* M, float* N, float* P, int NumRowsM, int NumColsM, int NumColsN) {
    int sizeM = NumRowsM * NumColsM * sizeof(float);
    int sizeN = NumColsM * NumColsN * sizeof(float);
    int sizeP = NumRowsM * NumColsN * sizeof(float);
    float* d_M, * d_N, * d_P;
    cudaMalloc(&d_M, sizeM);
    cudaMalloc(&d_N, sizeN);
    cudaMalloc(&d_P, sizeP);

    cudaMemcpy(d_M, M, sizeM, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, N, sizeN, cudaMemcpyHostToDevice);

    dim3 dimBlock(BLOCKS, BLOCKS);
    dim3 dimGrid((NumColsN + dimBlock.x - 1) / dimBlock.x, (NumRowsM + dimBlock.y - 1) / dimBlock.y);

    // Create CUDA event handles for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start time
    cudaEventRecord(start);

    MatrixMulKernel << <dimGrid, dimBlock >> > (d_M, d_N, d_P, NumRowsM, NumColsM, NumColsN);

    // Record stop time
    cudaEventRecord(stop);

    cudaMemcpy(P, d_P, sizeP, cudaMemcpyDeviceToHost);

    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);

    // Wait for completion of kernel execution
    cudaEventSynchronize(stop);

    // Calculate elapsed time in milliseconds
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("CUDA C Matrix Multiplication Execution time is: %.3f ms for size of %d x %d and %d x %d\n", elapsedTime, NumRowsM, NumColsM, NumColsM, NumColsN);
}

int main() {
    int NumRowsM = 1024;
    int NumColsM = 512;
    int NumColsN = 2048;

    float* M = (float*)malloc(NumRowsM * NumColsM * sizeof(float));
    float* N = (float*)malloc(NumColsM * NumColsN * sizeof(float));
    float* P = (float*)malloc(NumRowsM * NumColsN * sizeof(float));

    // Initialize matrices with some values
    for (int i = 0; i < NumRowsM * NumColsM; i++)
        M[i] = i % 10;
    for (int i = 0; i < NumColsM * NumColsN; i++)
        N[i] = i % 10;

    MatrixMultiplication(M, N, P, NumRowsM, NumColsM, NumColsN);

    free(M);
    free(N);
    free(P);

    return 0;
}