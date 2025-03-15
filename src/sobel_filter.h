#ifndef SOBEL_FILTER_H
#define SOBEL_FILTER_H


#include "gif_lib.h"   
#include "pixel.h"

#ifdef __cplusplus
extern "C" {
#endif

void apply_sobel_filter_cuda(animated_gif *image);
void apply_sobel_filter_tiling(animated_gif *image);

#ifdef __cplusplus
}
#endif

#endif
