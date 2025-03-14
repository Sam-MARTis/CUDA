#define W 500
#define H 500
#define TX 32
#define TY 32


__global__ void distanceKernel(float *d_out, int w, int h, float2 pos){
    const int c = blockDim.x*blockIdx.x + threadIdx.x;
    const int r = blockDim.y*blockIdx.y + threadIdx.y;
    const int idx = r*w + c;
    if(c>=w || r>=h){
        return ;
    }

    d_out[idx] = sqrt((c-pos.x)*(c-pos.x) + (r-pos.y)*(r-pos.y));
}

int main(){
    float *out = (float *)calloc(W*H, sizeof(float));
    float *d_out;

    cudaMalloc(&d_out, W*H*sizeof(float));
    const float2 pos = {0.0f, 0.0f};
    const dim3 blockSize(TX, TY);
    const int bx = (W-1 + TX)/TX;
    const int by = (H-1 + TY)/TY;
    const dim3 gridSize = dim3(bx, by);

    distanceKernel<<<gridSize, blockSize>>>(d_out, W, H, pos);

    cudaMemcpy(out, d_out, cudaMemcpyDeviceToHost);
    cudaFree(d_out);
    free(out);
    return 0;
}