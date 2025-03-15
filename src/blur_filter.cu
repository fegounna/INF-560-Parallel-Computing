#include <stdio.h>
#include <stdlib.h>
#include "blur_filter.h"
#include <cuda_runtime.h>
#include <math.h>


#define CONV(j, k, width) ((j) * (width) + (k))
#define BLOCK_DIM 16
#define MAX_BLUR_SIZE 5
#define SHARED_MEM_SIZE (BLOCK_DIM + 2 * MAX_BLUR_SIZE + 1)  // Added +1 for padding

#define CONV(j, k, width) ((j) * (width) + (k))
#define BLOCK_DIM 16
#define MAX_BLUR_SIZE 5
#define SHARED_MEM_SIZE (BLOCK_DIM + 2 * MAX_BLUR_SIZE)

__global__ void blur_kernel(pixel* oldImg, pixel* newImg, int width, int height, int size) {
    __shared__ pixel smem[SHARED_MEM_SIZE][SHARED_MEM_SIZE];

    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x * BLOCK_DIM - size;
    int by = blockIdx.y * BLOCK_DIM - size;

    // Global indices with halo
    int gx = bx + tx;
    int gy = by + ty;

    // Load data into shared memory with clamping
    if (gx >= 0 && gx < width && gy >= 0 && gy < height)
        smem[ty][tx] = oldImg[CONV(gy, gx, width)];
    else
        smem[ty][tx] = {0, 0, 0};

    __syncthreads();

    // Calculate output indices (original block without halo)
    int out_x = blockIdx.x * BLOCK_DIM + threadIdx.x - size;
    int out_y = blockIdx.y * BLOCK_DIM + threadIdx.y - size;

    if (out_x >= 0 && out_x < width && out_y >= 0 && out_y < height) {
        bool in_top = (out_y >= size) && (out_y < height/10 - size) && (out_x >= size) && (out_x < width - size);
        bool in_bottom = (out_y >= height*0.9 + size) && (out_y < height - size) && (out_x >= size) && (out_x < width - size);

        if (in_top || in_bottom) {
            int t_r = 0, t_g = 0, t_b = 0;
            int count = 0;

            for (int dy = -size; dy <= size; dy++) {
                for (int dx = -size; dx <= size; dx++) {
                    int sy = ty + dy + size;
                    int sx = tx + dx + size;

                    if (sy >= 0 && sy < SHARED_MEM_SIZE && sx >= 0 && sx < SHARED_MEM_SIZE) {
                        t_r += smem[sy][sx].r;
                        t_g += smem[sy][sx].g;
                        t_b += smem[sy][sx].b;
                        count++;
                    }
                }
            }

            if (count > 0) {
                newImg[CONV(out_y, out_x, width)].r = t_r / count;
                newImg[CONV(out_y, out_x, width)].g = t_g / count;
                newImg[CONV(out_y, out_x, width)].b = t_b / count;
            }
        } else {
            newImg[CONV(out_y, out_x, width)] = oldImg[CONV(out_y, out_x, width)];
        }
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