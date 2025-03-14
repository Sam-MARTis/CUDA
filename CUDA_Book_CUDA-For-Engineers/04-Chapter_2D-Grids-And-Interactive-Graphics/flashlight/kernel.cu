#include "kernel.h"
#define TX 32
#define TY 32

__device__ int clip(int n){
    return n>255? 255 : (n<0?0:n);
}

__global__ void distanceKernel(uchar4 *d_out, int w, int h, int2 pos){
    const int c = blockDim.x *blockIdx.x + threadIdx.x;
    const int r = blockDim.y *blockIdx.y + threadIdx.y;
    const int idx = r*w + c;

    const int distance = sqrt((c-pos.x)*(c-pos.x) + (r-pos.y)*(r-pos.y));
    const unsigned char intensity = clip(255 - distance);
    d_out[idx].x = intensity;
    d_out[idx].y = intensity;
    d_out[idx].z = 0;
    d_out[idx].w = 255;
}

void kernelLauncher(uchar4 *d_out, int w, int h, int2 pos){
    const dim3 blockSize(TX, TY);
    const dim3 gridSize = dim3((w-1+TX)/TX, (h-1+TY)/TY);
    distanceKernel<<<gridSize, blockSize>>>(d_out, w, h, pos);
}