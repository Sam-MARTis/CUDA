#include <stdio.h>
#define N 64
#define TPB 32

__device__ float distance(float x1, float x2){
    return sqrt((x2-x1)*(x2-x1));
}

__global__ void distanceKernel(float *d_out, float *d_in, float ref, int len){
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    const float x = d_in[i];
    d_out[i] = distance(x, ref);
    printf(" = %2d: dist from %f to %f is %f.\n", i, ref, x, d_out[i]);
}

__host__ void distanceArray(float *out, float *in, float ref, int len){
    float *d_in = 0;
    float *d_out = 0;

    cudaMalloc(&d_in,  len*sizeof(float));
    cudaMalloc(&d_out, len*sizeof(float));
    
    cudaMemcpy(d_in, in, len*sizeof(float), cudaMemcpyHostToDevice);

    distanceKernel<<<N/TPB, TPB>>>(d_out, d_in, ref, N);

    cudaMemcpy(out, d_out, len*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);
}