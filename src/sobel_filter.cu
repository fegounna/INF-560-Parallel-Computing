#include <stdio.h>
#include <stdlib.h>
#include "sobel_filter.h"
#include <cuda_runtime.h>
#include <math.h>
#define CONV(j, k, width) ((j) * (width) + (k))

static __global__ void sobel_kernel(pixel* image_pixels, pixel* sobel, int width, int height) {
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if (j < height && k < width) {
        if (j >= 1 && j < height - 1 && k >= 1 && k < width - 1) {
            int pixel_blue_no = image_pixels[CONV(j-1, k-1, width)].b;
            int pixel_blue_n  = image_pixels[CONV(j-1, k  , width)].b;
            int pixel_blue_ne = image_pixels[CONV(j-1, k+1, width)].b;
            int pixel_blue_so = image_pixels[CONV(j+1, k-1, width)].b;
            int pixel_blue_s  = image_pixels[CONV(j+1, k  , width)].b;
            int pixel_blue_se = image_pixels[CONV(j+1, k+1, width)].b;
            int pixel_blue_o  = image_pixels[CONV(j  , k-1, width)].b;
            int pixel_blue    = image_pixels[CONV(j  , k  , width)].b;
            int pixel_blue_e  = image_pixels[CONV(j  , k+1, width)].b;

            float deltaX_blue = -pixel_blue_no + pixel_blue_ne - 2*pixel_blue_o + 2*pixel_blue_e - pixel_blue_so + pixel_blue_se;             
            float deltaY_blue = pixel_blue_se + 2*pixel_blue_s + pixel_blue_so - pixel_blue_ne - 2*pixel_blue_n - pixel_blue_no;
            float val_blue = sqrt(deltaX_blue * deltaX_blue + deltaY_blue * deltaY_blue) / 4;

            if (val_blue > 50) {
                sobel[CONV(j, k, width)].r = 255;
                sobel[CONV(j, k, width)].g = 255;
                sobel[CONV(j, k, width)].b = 255;
            } else {
                sobel[CONV(j, k, width)].r = 0;
                sobel[CONV(j, k, width)].g = 0;
                sobel[CONV(j, k, width)].b = 0;
            }
        } else {
            sobel[CONV(j, k, width)] = image_pixels[CONV(j, k, width)];
        }
    }
} 

void apply_sobel_filter_cuda(animated_gif *image) {
    for (int i = 0; i < image->n_images; i++) {
        int width = image->width[i];
        int height = image->height[i];
        pixel *image_pixels = image->p[i];
        pixel * d_sobel;
        pixel * d_image;

        cudaMalloc((void**)&d_image, width * height * sizeof(pixel));
        cudaMalloc((void**)&d_sobel, width * height * sizeof(pixel));

        cudaMemcpy(d_image,image_pixels, width * height * sizeof(pixel), cudaMemcpyHostToDevice);

        dim3 threadsPerBlock(32, 32);
        dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                       (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

        sobel_kernel<<<numBlocks, threadsPerBlock>>>(d_image, d_sobel, width, height);

        cudaMemcpy(image_pixels, d_sobel, width * height * sizeof(pixel), cudaMemcpyDeviceToHost);

        cudaFree(d_image);
        cudaFree(d_sobel);
    }
}