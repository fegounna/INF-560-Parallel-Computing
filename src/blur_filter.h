#ifndef BLUR_FILTER_H
#define BLUR_FILTER_H

#include "gif_lib.h"   
#include "pixel.h"

#ifdef __cplusplus
extern "C" {
#endif


void apply_blur_filter_cuda(animated_gif *image, int size, int threshold);

#ifdef __cplusplus
}
#endif

#endif
