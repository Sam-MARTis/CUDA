#include "kernel.h"
#define cimg_display 0
// #include "CImg.h"
#include "CImg.h"
#include <cuda_runtime.h>
#include <stdlib.h>

int main(){
    cimg_library:: CImg<unsigned char> image("./MeTrophyElectriphy.bmp");
    const int w = image.width();
    const int h = image.height();

    uchar4 *arr = (uchar4*)malloc(w*h* sizeof(uchar4));
    for(int r = 0; r<h; r++){
        for(int c=0; c<w; c++){
            arr[r*w + c].x = image(c, r, 0);
            arr[r*w + c].y = image(c, r, 1);
            arr[r*w + c].z = image(c, r, 2);
        }
        
    }

    sharpenParallel(arr, w, h);
    for(int r = 0; r<h; r++){
        for(int c=0; c<w; c++){
            image(c, r, 0) = arr[r*w + c].x;
            image(c, r, 1) = arr[r*w + c].y;
            image(c, r, 2) = arr[r*w + c].z;
        }
    }
    image.save_bmp("sharpened.bmp");
    free(arr);
    return 0;
}


