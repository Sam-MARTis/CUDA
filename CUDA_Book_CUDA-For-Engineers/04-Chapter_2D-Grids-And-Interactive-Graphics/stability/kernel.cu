#include "kernel.h"
#define TX 32
#define TY 32
#define LEN 5.f
#define TIME_STEP 0.005f
#define FINAL_TIME 10.f


__device__ unsigned char clip(int n){
    return n>255? 255 : (n<0?0:n);
}
__device__ float scale(int i, int w){
    return LEN*(2*(((1.f*i)/w) - 0.5f));
}

__device__ float f(float x, float y, float param, int sys){
    if(sys==1){
        return x-2*param*y;
    }
    if(sys==2){
        return -x + param*(1-x*x)*y;
    }
    else{
        return -x-2*param*y;
    }
}

__device__ float2 euler(float x, float y, float dt, float tFinal, float param, float sys){
    float dx = 0.f;
    float dy = 0.f;
    for(float t = 0; t<tFinal; t+= dt){
        dx = dt*y;
        dy = dt*f(x, y, param, sys);
        x+= dx;
        y+= dy;
    }
    return make_float2(x, y);
}




__global__ void stabilityImageKernel(uchar4 *d_out, int w, int h, float p, int s){
    const int c = blockDim.x *blockIdx.x + threadIdx.x;
    const int r = blockDim.y *blockIdx.y + threadIdx.y;
    if((c>=w) || (r>= h)) return;
    const int idx = r*w + c;
    const float x0 = scale(c, w);
    const float y0 = scale(r, h);
    const float S = sqrtf(x0*x0 + y0*y0);
    const float2 pos = euler(x0, y0, TIME_STEP, FINAL_TIME, p, s);
    const float dS = sqrtf(pos.x*pos.x + pos.y*pos.y);
    const float dSFrac = dS/S;

    d_out[idx].x = clip(dSFrac*255);
    d_out[idx].y = ((c==w/2) ||(r==h/2)) ? 255: 0;
    d_out[idx].z =clip((1/dSFrac)*255);
    d_out[idx].w = 255;
}

void kernelLauncher(uchar4 *d_out, int w, int h, float p, int s){
    const dim3 blockSize(TX, TY);
    const dim3 gridSize = dim3((w-1+TX)/TX, (h-1+TY)/TY);
    stabilityImageKernel<<<gridSize, blockSize>>>(d_out, w, h, p, s);
}