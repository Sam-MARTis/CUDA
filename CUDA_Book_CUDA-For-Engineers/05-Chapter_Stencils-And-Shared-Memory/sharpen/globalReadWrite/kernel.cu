#include "kernel.h"
#define TX 32
#define TY 32
#define RAD 1

int divUp(int a, int b){
    return (a-1+b)/b;
}

__device__
unsigned char clip(int n){
    return n>255 ? 255: (n<0 ? 0: n);
}

__device__
int idxClip(int idx, int idxMax){
    return idx>(idxMax-1) ? (idxMax-1) : (idx<0 ? 0: idx);
}
__device__
int flatten(int col, int row, int width, int height){
    return idxClip(col, width) +  idxClip(row, height)*width;
}

__global__
void sharpenKernel(uchar4 *d_out, const uchar4 *d_in, const float *d_filter, int w, int h){
    const int c = threadIdx.x + blockDim.x*blockIdx.x;
    const int r = threadIdx.y + blockDim.y*blockIdx.y;
    if(c>=w|| r>=h){
        return;
    }
    const int idx = flatten(c, r, w, h);
    const int filtSize = 2*RAD + 1;
    float rgb[3] = {0.f, 0.f, 0.f};

    for(int rf = -RAD; rf<= RAD; rf++){
        for(int cf = -RAD; cf<= RAD; cf++){
            const int dIdx = flatten(c+cf,r+rf, w, h );
            const int fIdx = flatten(RAD+cf, RAD+rf, filtSize, filtSize);
            const uchar4 color = d_in[dIdx];
            const float weigth = d_filter[fIdx];
            rgb[0] += color.x*weigth;
            rgb[1] += color.y*weigth;
            rgb[2] += color.z*weigth;
        

        }
    }
    d_out[idx].x = clip(rgb[0]);
    d_out[idx].x = clip(rgb[1]);
    d_out[idx].x = clip(rgb[2]);
}


