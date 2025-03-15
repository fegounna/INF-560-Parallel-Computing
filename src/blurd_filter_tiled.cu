#include <stdio.h>
#include <stdlib.h>
#include "blur_filter.h"
#include <cuda_runtime.h>
#include <math.h>
#define CONV(j, k, width) ((j) * (width) + (k))

__global__ void blur_kernel(pixel* oldImg, pixel* newImg, int width, int height, int size) {
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (j >= height || k >= width) return;

    // top part
    if (j>=size && j < height/10-size && k>=size && k < width-size) {
        int t_r = 0;
        int t_g = 0;
        int t_b = 0;

        for (int stencil_j = -size ; stencil_j <= size ; stencil_j++ )
        {
            for (int stencil_k = -size ; stencil_k <= size ; stencil_k++ )
            {
                t_r += oldImg[CONV(j+stencil_j,k+stencil_k,width)].r ;
                t_g += oldImg[CONV(j+stencil_j,k+stencil_k,width)].g ;
                t_b += oldImg[CONV(j+stencil_j,k+stencil_k,width)].b ;
            }
        }
        newImg[CONV(j,k,width)].r = t_r / ( (2*size+1)*(2*size+1) ) ;
        newImg[CONV(j,k,width)].g = t_g / ( (2*size+1)*(2*size+1) ) ;
        newImg[CONV(j,k,width)].b = t_b / ( (2*size+1)*(2*size+1) ) ;
    }
    // bottom part
    else if(j>=height*0.9+size && j<height-size && k>=size && k<width-size){
        int t_r = 0 ;
        int t_g = 0 ;
        int t_b = 0 ;
        for (int stencil_j = -size ; stencil_j <= size ; stencil_j++ )
        {
            for (int stencil_k = -size ; stencil_k <= size ; stencil_k++ )
            {
                t_r += oldImg[CONV(j+stencil_j,k+stencil_k,width)].r ;
                t_g += oldImg[CONV(j+stencil_j,k+stencil_k,width)].g ;
                t_b += oldImg[CONV(j+stencil_j,k+stencil_k,width)].b ;
            }
        }
        newImg[CONV(j,k,width)].r = t_r / ( (2*size+1)*(2*size+1) ) ;
        newImg[CONV(j,k,width)].g = t_g / ( (2*size+1)*(2*size+1) ) ;
        newImg[CONV(j,k,width)].b = t_b / ( (2*size+1)*(2*size+1) ) ;
    }
    // middle part
    else{
        newImg[CONV(j, k, width)].r = oldImg[CONV(j, k, width)].r;
        newImg[CONV(j, k, width)].g = oldImg[CONV(j, k, width)].g;
        newImg[CONV(j, k, width)].b = oldImg[CONV(j, k, width)].b;
    }
}


__global__ void checkConvergence(
    pixel *oldImg, pixel *newImg, int *end, int width, int height, int threshold) 
{
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.x * blockDim.x + threadIdx.x + 1;

    if (j >= height - 1 || k >= width - 1) return;

    int idx = CONV(j, k, width);

    float diff_r = fabsf(newImg[idx].r - oldImg[idx].r);
    float diff_g = fabsf(newImg[idx].g - oldImg[idx].g);
    float diff_b = fabsf(newImg[idx].b - oldImg[idx].b);

    if (diff_r > threshold || diff_g > threshold || diff_b > threshold) {
        atomicExch(end, 0); 
    }

    oldImg[idx] = newImg[idx];
}



void apply_blur_filter_cuda(animated_gif *image, int size, int threshold) {
    for (int i = 0; i < image->n_images; i++) {
        int width = image->width[i];
        int height = image->height[i];
        pixel *oldImg = image->p[i];
        pixel * d_old;
        pixel * d_new;

        cudaMalloc((void**)&d_old, width * height * sizeof(pixel));
        cudaMalloc((void**)&d_new, width * height * sizeof(pixel));

        cudaMemcpy(d_old,oldImg, width * height * sizeof(pixel), cudaMemcpyHostToDevice);

        dim3 threadsPerBlock(16, 16);
        dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                       (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

        int end;
        int *d_end;
        cudaMalloc(&d_end, sizeof(int));
        do {
            end = 1;
            cudaMemcpy(d_end, &end, sizeof(int), cudaMemcpyHostToDevice);

            blur_kernel<<<numBlocks, threadsPerBlock>>>(d_old, d_new, width, height, size);
            cudaDeviceSynchronize();
            checkConvergence<<<numBlocks, threadsPerBlock>>>(d_old, d_new,d_end, width, height, threshold);
            cudaDeviceSynchronize();
            cudaMemcpy(&end, d_end, sizeof(int), cudaMemcpyDeviceToHost);

        } while (threshold > 0 && !end); 

        cudaMemcpy(oldImg, d_new, width * height * sizeof(pixel), cudaMemcpyDeviceToHost);

        cudaFree(d_old);
        cudaFree(d_new);
        cudaFree(d_end);
    }
} 