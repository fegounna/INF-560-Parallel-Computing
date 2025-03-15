#ifndef PIXEL_H
#define PIXEL_H

#include <gif_lib.h>  // Ensure this is included for GifFileType

/* Represent one pixel from the image */
typedef struct pixel
{
    int r; /* Red */
    int g; /* Green */
    int b; /* Blue */
} pixel;

/* Represent one GIF image (animated or not) */
typedef struct animated_gif
{
    int n_images;  /* Number of images */
    int *width;    /* Width of each image */
    int *height;   /* Height of each image */
    pixel **p;     /* Pixels of each image */
    GifFileType *g; /* Internal representation. DO NOT MODIFY */
} animated_gif;

#endif // PIXEL_H
