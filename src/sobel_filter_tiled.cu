#include <stdio.h>
#include <stdlib.h>
#include "sobel_filter.h"
#include <cuda_runtime.h>
#include <math.h>
#define CONV(j, k, width) ((j) * (width) + (k))

static __global__ void sobel_kernel(pixel* image_pixels, pixel* sobel, int width, int height) {
    const int BLOCK_DIM = 16; // Adjust based on GPU capabilities
    __shared__ pixel smem[BLOCK_DIM+2][BLOCK_DIM+2]; // Halo included

    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x * BLOCK_DIM, by = blockIdx.y * BLOCK_DIM;

    // Global indices for main tile
    int gx = bx + tx;
    int gy = by + ty;

    // Load main tile into shared memory (center region)
    if (gx < width && gy < height) {
        smem[ty+1][tx+1] = image_pixels[CONV(gy, gx, width)];
    } else {
        smem[ty+1][tx+1] = {0, 0, 0}; // Assume pixel constructor
    }

    // Load halo regions
    // Left halo (tx=0)
    if (tx == 0) {
        int hx = bx - 1, hy = by + ty;
        if (hx >= 0 && hy < height) smem[ty+1][0] = image_pixels[CONV(hy, hx, width)];
        else smem[ty+1][0] = {0, 0, 0};
    }

    // Right halo (tx=BLOCK_DIM-1)
    if (tx == BLOCK_DIM-1) {
        int hx = bx + BLOCK_DIM, hy = by + ty;
        if (hx < width && hy < height) smem[ty+1][BLOCK_DIM+1] = image_pixels[CONV(hy, hx, width)];
        else smem[ty+1][BLOCK_DIM+1] = {0, 0, 0};
    }

    // Top halo (ty=0)
    if (ty == 0) {
        int hx = bx + tx, hy = by - 1;
        if (hy >= 0 && hx < width) smem[0][tx+1] = image_pixels[CONV(hy, hx, width)];
        else smem[0][tx+1] = {0, 0, 0};
    }

    // Bottom halo (ty=BLOCK_DIM-1)
    if (ty == BLOCK_DIM-1) {
        int hx = bx + tx, hy = by + BLOCK_DIM;
        if (hy < height && hx < width) smem[BLOCK_DIM+1][tx+1] = image_pixels[CONV(hy, hx, width)];
        else smem[BLOCK_DIM+1][tx+1] = {0, 0, 0};
    }

    // Load corners
    if (tx == 0 && ty == 0) {
        // Top-left
        int hx = bx - 1, hy = by - 1;
        if (hx >= 0 && hy >= 0) smem[0][0] = image_pixels[CONV(hy, hx, width)];
        else smem[0][0] = {0, 0, 0};
    }
    if (tx == BLOCK_DIM-1 && ty == 0) {
        // Top-right
        int hx = bx + BLOCK_DIM, hy = by - 1;
        if (hx < width && hy >= 0) smem[0][BLOCK_DIM+1] = image_pixels[CONV(hy, hx, width)];
        else smem[0][BLOCK_DIM+1] = {0, 0, 0};
    }
    if (tx == 0 && ty == BLOCK_DIM-1) {
        // Bottom-left
        int hx = bx - 1, hy = by + BLOCK_DIM;
        if (hx >= 0 && hy < height) smem[BLOCK_DIM+1][0] = image_pixels[CONV(hy, hx, width)];
        else smem[BLOCK_DIM+1][0] = {0, 0, 0};
    }
    if (tx == BLOCK_DIM-1 && ty == BLOCK_DIM-1) {
        // Bottom-right
        int hx = bx + BLOCK_DIM, hy = by + BLOCK_DIM;
        if (hx < width && hy < height) smem[BLOCK_DIM+1][BLOCK_DIM+1] = image_pixels[CONV(hy, hx, width)];
        else smem[BLOCK_DIM+1][BLOCK_DIM+1] = {0, 0, 0};
    }

    __syncthreads();

    // Calculate output indices
    int x = bx + tx;
    int y = by + ty;
    if (x >= width || y >= height) return;

    // Process pixel
    if (y >= 1 && y < height-1 && x >= 1 && x < width-1) {
        // Use shared memory for Sobel calculation
        int b_no = smem[ty][tx].b;
        int b_n  = smem[ty][tx+1].b;
        int b_ne = smem[ty][tx+2].b;
        int b_so = smem[ty+2][tx].b;
        int b_s  = smem[ty+2][tx+1].b;
        int b_se = smem[ty+2][tx+2].b;
        int b_o  = smem[ty+1][tx].b;
        int b_e  = smem[ty+1][tx+2].b;

        float dx = -b_no + b_ne - 2*b_o + 2*b_e - b_so + b_se;
        float dy = b_se + 2*b_s + b_so - b_ne - 2*b_n - b_no;
        float val = sqrtf(dx*dx + dy*dy) / 4.0f;

        if (val > 50.0f) sobel[CONV(y, x, width)] = {255, 255, 255};
        else sobel[CONV(y, x, width)] = {0, 0, 0};
    } else {
        sobel[CONV(y, x, width)] = image_pixels[CONV(y, x, width)];
    }
}


void apply_sobel_filter_tiling(animated_gif *image) {
    for (int i = 0; i < image->n_images; i++) {
        int width = image->width[i];
        int height = image->height[i];
        pixel *image_pixels = image->p[i];
        pixel * d_sobel;
        pixel * d_image;

        cudaMalloc((void**)&d_image, width * height * sizeof(pixel));
        cudaMalloc((void**)&d_sobel, width * height * sizeof(pixel));

        cudaMemcpy(d_image,image_pixels, width * height * sizeof(pixel), cudaMemcpyHostToDevice);

        dim3 threadsPerBlock(16, 16);
        dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                       (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

        sobel_kernel<<<numBlocks, threadsPerBlock>>>(d_image, d_sobel, width, height);

        cudaMemcpy(image_pixels, d_sobel, width * height * sizeof(pixel), cudaMemcpyDeviceToHost);

        cudaFree(d_image);
        cudaFree(d_sobel);
    }
}