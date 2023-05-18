#sequential MM
%%writefile seqMM.c
#include <stdio.h>
#include <time.h>

void MatrixMultiplication(float* M, float* N, float* P, int NumRowsM, int NumColsM, int NumColsN) {
    for (int i = 0; i < NumRowsM; ++i) {
        for (int j = 0; j < NumColsN; ++j) {
            float sum = 0;
            for (int k = 0; k < NumColsM; ++k) {
                sum += M[i * NumColsM + k] * N[k * NumColsN + j];
            }
            P[i * NumColsN + j] = sum;
        }
    }
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

    clock_t start_time = clock();

    MatrixMultiplication(M, N, P, NumRowsM, NumColsM, NumColsN);

    clock_t end_time = clock();
    double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC * 1000.0;

    printf("Sequential Matrix Multiplication Execution time is: %.3f ms for size of %d x %d and %d x %d", elapsed_time, NumRowsM, NumColsM, NumColsM, NumColsN);

    free(M);
    free(N);
    free(P);

    return 0;
}