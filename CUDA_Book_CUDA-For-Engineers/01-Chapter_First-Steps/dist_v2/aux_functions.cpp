#include "aux_functions.h"
#include<math.h>

float scale(int i, int n){
    return ((float) i)/(n-1);
}

float distance(float x1, float x2){
    return ((x1-x2)*(x1-x2));
}

void distanceArray(float *in, float *out, float ref, int n){
    for(int i = 0; i< n; i++){
        out[i] = distance(in[i], ref);
    }
}