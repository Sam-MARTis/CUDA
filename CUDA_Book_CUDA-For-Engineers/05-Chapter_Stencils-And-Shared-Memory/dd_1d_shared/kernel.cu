#include "kernel.h"
#define TPB 64
#define RAD 1

__global__ void ddKernel(float *d_out, float *d_in, int size, float h){
    const int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if(idx>size) return;



    extern __shared__ float s_in[];
    const int s_idx = threadIdx.x + RAD;


    s_in[s_idx] = d_in[idx];
    if(threadIdx.x<RAD){
        s_in[s_idx-RAD] = d_in[idx-RAD];
        s_in[s_idx+blockDim.x] = d_in[idx+blockDim.x];
    }

    __syncthreads();
    d_out[idx] = (s_in[s_idx-1]+s_in[s_idx+1] - 2.f*s_in[s_idx])/(h*h);
}

void ddParallel(float *out, const float *in, int n, float h){
    float *d_in = 0, *d_out = 0;
    cudaMalloc(&d_in, n*sizeof(float));
    cudaMalloc(&d_out, n*sizeof(float));

    cudaMemcpy(d_in, in, n*sizeof(float), cudaMemcpyHostToDevice);

    const size_t smemSize = (TPB + 2*RAD)*sizeof(float);

    ddKernel<<<(n+TPB-1)/TPB, TPB, smemSize>>>(d_out, d_in, n, h);

    cudaMemcpy(out, d_out, n*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_out);
    cudaFree(d_in);
    
}
