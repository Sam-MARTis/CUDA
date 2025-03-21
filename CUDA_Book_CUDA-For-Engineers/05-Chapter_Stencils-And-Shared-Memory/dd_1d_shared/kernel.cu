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
    d_out[idx] = (s_in[s_idx-1]+s_in[s_idx+1] - 2.f*s_in[idx])/(h*h);
}
