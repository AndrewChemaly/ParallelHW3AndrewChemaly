#CUDA MM TILED RECTANGULAR MATRICES
%%cu
#include <stdio.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 32 // Tile size for matrix multiplication

__global__ void MatrixMulKernel(float* M, float* N, float* P, int NumRowsM, int NumColsM, int NumColsN) {
    __shared__ float ds_M[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_N[TILE_WIDTH][TILE_WIDTH];
    int bx = blockIdx.x; 
    int by = blockIdx.y;
    int tx = threadIdx.x; 
    int ty = threadIdx.y;
    int Row = by * blockDim.y + ty;
    int Col = bx * blockDim.x + tx;
    float Pvalue = 0;
    
    // Loop over the M and N tiles required to compute the P element
    for (int p = 0; p < (NumColsM + TILE_WIDTH - 1)/TILE_WIDTH; ++p) {
        // Collaborative loading of M and N tiles into shared memory
        if(Row < NumRowsM && p*TILE_WIDTH+tx < NumColsM) {
            ds_M[ty][tx] = M[Row * NumColsM + p * TILE_WIDTH + tx];
        }
        else {
            ds_M[ty][tx] = 0.0;
        }
        if(Col < NumColsN && p*TILE_WIDTH+ty < NumColsM) {
            ds_N[ty][tx] = N[(p * TILE_WIDTH + ty) * NumColsN + Col];
        }
        else {
            ds_N[ty][tx] = 0.0;
        }
        __syncthreads();
        for (int i = 0; i < TILE_WIDTH; ++i)
            Pvalue += ds_M[ty][i] * ds_N[i][tx];
        __syncthreads();
    }
    if(Row < NumRowsM && Col < NumColsN) {
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

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
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
    printf("Tiled Matrix Multiplication Execution time is: %.3f ms for size of %d x %d and %d x %d\n", elapsedTime, NumRowsM, NumColsM, NumColsM, NumColsN);
}

int main() {
    int NumRowsM = 1024;
    int NumColsM = 512;
    int NumColsN = 2048;
    float* M = (float*)malloc(NumRowsM * NumColsM * sizeof(float));
    float* N = (float*)malloc(NumColsM * NumColsN * sizeof(float));
    float* P = (float*)malloc(NumRowsM * NumColsN * sizeof(float));

    for (int i = 0; i < NumRowsM * NumColsM; i++)
        M[i] = i % 10; // Initialize matrix M with some values

    for (int i = 0; i < NumColsM * NumColsN; i++)
        N[i] = i % 10; // Initialize matrix N with some values

    MatrixMultiplication(M, N, P, NumRowsM, NumColsM, NumColsN);

    free(M);
    free(N);
    free(P);

    return 0;
}