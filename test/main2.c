
/*
 * INF560
 *
 * Image Filtering Project
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include "gif_lib.h"
#include <mpi.h> 
#include <omp.h>
/* Set this macro to 1 to enable debugging information */
#define SOBELF_DEBUG 0

/* Represent one pixel from the image */
typedef struct pixel {
    int r; /* Red */
    int g; /* Green */
    int b; /* Blue */
} pixel;

/* Represent one GIF image (animated or not) */
typedef struct animated_gif {
    int n_images;    /* Number of images */
    int * width;     /* Width of each image */
    int * height;    /* Height of each image */
    pixel ** p;      /* Pixels of each image */
    GifFileType * g; /* Internal representation. DO NOT MODIFY */
} animated_gif;
animated_gif *
load_pixels( char * filename ) 
{
    GifFileType * g;
    ColorMapObject * colmap;
    int error;
    int n_images;
    int * width;
    int * height;
    pixel ** p;
    int i;
    animated_gif * image;

    /* Open the GIF image (read mode) */
    g = DGifOpenFileName( filename, &error );
    if ( g == NULL ) {
        fprintf( stderr, "Error DGifOpenFileName %s\n", filename );
        return NULL;
    }

    /* Read the GIF image */
    error = DGifSlurp( g );
    if ( error != GIF_OK ) {
        fprintf( stderr, "Error DGifSlurp: %d <%s>\n", error, GifErrorString(g->Error) );
        return NULL;
    }

    /* Grab the number of images and the size of each image */
    n_images = g->ImageCount;

    width = (int *)malloc( n_images * sizeof( int ) );
    if ( width == NULL ) {
        fprintf( stderr, "Unable to allocate width of size %d\n", n_images );
        return 0;
    }

    height = (int *)malloc( n_images * sizeof( int ) );
    if ( height == NULL ) {
        fprintf( stderr, "Unable to allocate height of size %d\n", n_images );
        return 0;
    }

    /* Fill the width and height */
    for ( i = 0 ; i < n_images ; i++ ) {
        width[i] = g->SavedImages[i].ImageDesc.Width;
        height[i] = g->SavedImages[i].ImageDesc.Height;
#if SOBELF_DEBUG
        printf( "Image %d: l:%d t:%d w:%d h:%d interlace:%d localCM:%p\n",
                i, g->SavedImages[i].ImageDesc.Left,
                g->SavedImages[i].ImageDesc.Top,
                g->SavedImages[i].ImageDesc.Width,
                g->SavedImages[i].ImageDesc.Height,
                g->SavedImages[i].ImageDesc.Interlace,
                g->SavedImages[i].ImageDesc.ColorMap );
#endif
    }

    /* Get the global colormap */
    colmap = g->SColorMap;
    if ( colmap == NULL ) {
        fprintf( stderr, "Error global colormap is NULL\n" );
        return NULL;
    }
#if SOBELF_DEBUG
    printf( "Global color map: count:%d bpp:%d sort:%d\n",
            g->SColorMap->ColorCount,
            g->SColorMap->BitsPerPixel,
            g->SColorMap->SortFlag );
#endif

    /* Allocate the array of pixels to be returned */
    p = (pixel **)malloc( n_images * sizeof( pixel * ) );
    if ( p == NULL ) {
        fprintf( stderr, "Unable to allocate array of %d images\n", n_images );
        return NULL;
    }
    
    for ( i = 0 ; i < n_images ; i++ ) {
        p[i] = (pixel *)malloc( width[i] * height[i] * sizeof( pixel ) );
        if ( p[i] == NULL ) {
            fprintf( stderr, "Unable to allocate %d-th array of %d pixels\n",
                     i, width[i] * height[i] );
            return NULL;
        }
    }
    
    /* Fill pixels */
    #pragma omp parallel for
    for ( i = 0 ; i < n_images ; i++ ) {
        int j;
        // if ( g->SavedImages[i].ImageDesc.ColorMap ) {
        //     /* TODO: No support for local color map */
        //     fprintf( stderr, "Error: application does not support local colormap\n" );
        //     return NULL;
        //     /* colmap = g->SavedImages[i].ImageDesc.ColorMap; */
        // }
        for ( j = 0 ; j < width[i] * height[i] ; j++ ) {
            int c;
            c = g->SavedImages[i].RasterBits[j];
            p[i][j].r = colmap->Colors[c].Red;
            p[i][j].g = colmap->Colors[c].Green;
            p[i][j].b = colmap->Colors[c].Blue;
        }
    }

    /* Allocate image info */
    image = (animated_gif *)malloc( sizeof(animated_gif) );
    if ( image == NULL ) {
        fprintf( stderr, "Unable to allocate memory for animated_gif\n" );
        return NULL;
    }

    /* Fill image fields */
    image->n_images = n_images;
    image->width = width;
    image->height = height;
    image->p = p;
    image->g = g;
#if SOBELF_DEBUG
    printf( "-> GIF w/ %d image(s) with first image of size %d x %d\n",
            image->n_images, image->width[0], image->height[0] );
#endif

    return image;
}

int 
output_modified_read_gif( char * filename, GifFileType * g ) 
{
    GifFileType * g2;
    int error2;
#if SOBELF_DEBUG
    printf( "Starting output to file %s\n", filename );
#endif
    g2 = EGifOpenFileName( filename, false, &error2 );
    if ( g2 == NULL ) {
        fprintf( stderr, "Error EGifOpenFileName %s\n", filename );
        return 0;
    }
    g2->SWidth = g->SWidth;
    g2->SHeight = g->SHeight;
    g2->SColorResolution = g->SColorResolution;
    g2->SBackGroundColor = g->SBackGroundColor;
    g2->AspectByte = g->AspectByte;
    g2->SColorMap = g->SColorMap;
    g2->ImageCount = g->ImageCount;
    g2->SavedImages = g->SavedImages;
    g2->ExtensionBlockCount = g->ExtensionBlockCount;
    g2->ExtensionBlocks = g->ExtensionBlocks;
    error2 = EGifSpew( g2 );
    if ( error2 != GIF_OK ) {
        fprintf( stderr, "Error after writing g2: %d <%s>\n", error2, GifErrorString(g2->Error) );
        return 0;
    }
    return 1;
}

#define CONV(l,c,nb_c) ((l)*(nb_c)+(c))

void apply_gray_filter(animated_gif *image) {
    int i, j;
    pixel **p = image->p;

    #pragma omp parallel for
    for (i = 0; i < image->n_images; i++) {
        for (j = 0; j < image->width[i] * image->height[i]; j++) {
            int moy = (p[i][j].r + p[i][j].g + p[i][j].b) / 3;
            p[i][j].r = moy;
            p[i][j].g = moy;
            p[i][j].b = moy;
        }
    }
}

void apply_gray_line(animated_gif *image) {
    int i, j, k;
    pixel **p = image->p;

    #pragma omp parallel for
    for (i = 0; i < image->n_images; i++) {
        for (j = 0; j < 10; j++) {
            for (k = image->width[i] / 2; k < image->width[i]; k++) {
                p[i][CONV(j, k, image->width[i])].r = 0;
                p[i][CONV(j, k, image->width[i])].g = 0;
                p[i][CONV(j, k, image->width[i])].b = 0;
            }
        }
    }
}
void apply_blur_filter(animated_gif *image, int size, int threshold) {
    int i;
    pixel **p = image->p;

    for (i = 0; i < image->n_images; i++) {
        int width = image->width[i];
        int height = image->height[i];
        pixel *new_pixels = (pixel *)malloc(width * height * sizeof(pixel));

        int n_iter = 0, end;
        do {
            end = 1;
            n_iter++;

            #pragma omp parallel for collapse(2) shared(new_pixels,p)
            for (int j = size; j < height - size; j++) {
                for (int k = size; k < width - size; k++) {
                    int t_r = 0, t_g = 0, t_b = 0;
                    for (int stencil_j = -size; stencil_j <= size; stencil_j++) {
                        for (int stencil_k = -size; stencil_k <= size; stencil_k++) {
                            int idx = CONV(j + stencil_j, k + stencil_k, width);
                            t_r += p[i][idx].r;
                            t_g += p[i][idx].g;
                            t_b += p[i][idx].b;
                        }
                    }
                    int avg_factor = (2 * size + 1) * (2 * size + 1);
                    int idx = CONV(j, k, width);
                    new_pixels[idx].r = t_r / avg_factor;
                    new_pixels[idx].g = t_g / avg_factor;
                    new_pixels[idx].b = t_b / avg_factor;
                }
            }

            #pragma omp parallel for collapse(2) reduction(&:end)
            for (int j = size; j < height - size; j++) {
                for (int k = size; k < width - size; k++) {
                    int idx = CONV(j, k, width);
                    if (abs(new_pixels[idx].r - p[i][idx].r) > threshold ||
                        abs(new_pixels[idx].g - p[i][idx].g) > threshold ||
                        abs(new_pixels[idx].b - p[i][idx].b) > threshold) {
                        end = 0;
                    }
                    p[i][idx] = new_pixels[idx];
                }
            }
        } while (threshold > 0 && !end);
        free(new_pixels);
    }
}


void apply_sobel_filter(animated_gif *image) {
    int i, j, k;
    pixel **p = image->p;

    #pragma omp parallel for
    for (int i = 0; i < image->n_images; i++) {
        int width = image->width[i];
        int height = image->height[i];
        pixel *sobel = (pixel *)malloc(width * height * sizeof(pixel));
    
        for (int j = 1; j < height - 1; j++) {
            for (int k = 1; k < width - 1; k++) {
                int pixel_blue_no = p[i][CONV(j-1, k-1, width)].b;
                int pixel_blue_n  = p[i][CONV(j-1, k  , width)].b;
                int pixel_blue_ne = p[i][CONV(j-1, k+1, width)].b;
                int pixel_blue_so = p[i][CONV(j+1, k-1, width)].b;
                int pixel_blue_s  = p[i][CONV(j+1, k  , width)].b;
                int pixel_blue_se = p[i][CONV(j+1, k+1, width)].b;
                int pixel_blue_o  = p[i][CONV(j,   k-1, width)].b;
                int pixel_blue    = p[i][CONV(j,   k  , width)].b;
                int pixel_blue_e  = p[i][CONV(j,   k+1, width)].b;
    
                float deltaX_blue = -pixel_blue_no + pixel_blue_ne - 2 * pixel_blue_o + 2 * pixel_blue_e - pixel_blue_so + pixel_blue_se;
                float deltaY_blue = pixel_blue_se + 2 * pixel_blue_s + pixel_blue_so - pixel_blue_ne - 2 * pixel_blue_n - pixel_blue_no;
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
            }
        }
    
        for (int j = 1; j < height - 1; j++) {
            for (int k = 1; k < width - 1; k++) {
                p[i][CONV(j, k, width)] = sobel[CONV(j, k, width)];
            }
        }
        free(sobel);
    }
    
}
int store_pixels(char *filename, animated_gif *image) {
    int n_colors = 0;
    pixel **p;
    int i, j, k;
    GifColorType *colormap = (GifColorType *)malloc(256 * sizeof(GifColorType));
    if (colormap == NULL) {
        fprintf(stderr, "Unable to allocate 256 colors\n");
        return 0;
    }

    for (i = 0; i < 256; i++) {
        colormap[i].Red = 255;
        colormap[i].Green = 255;
        colormap[i].Blue = 255;
    }

    int moy = (image->g->SColorMap->Colors[image->g->SBackGroundColor].Red +
               image->g->SColorMap->Colors[image->g->SBackGroundColor].Green +
               image->g->SColorMap->Colors[image->g->SBackGroundColor].Blue) / 3;
    moy = moy < 0 ? 0 : (moy > 255 ? 255 : moy);

    colormap[0].Red = moy;
    colormap[0].Green = moy;
    colormap[0].Blue = moy;
    image->g->SBackGroundColor = 0;

    #pragma omp parallel for
    for (j = 0; j < image->g->ExtensionBlockCount; j++) {
        if (image->g->ExtensionBlocks[j].Function == GRAPHICS_EXT_FUNC_CODE) {
            int tr_color = image->g->ExtensionBlocks[j].Bytes[3];
            if (tr_color >= 0 && tr_color < 255) {
                int found = -1;
                int moy = (image->g->SColorMap->Colors[tr_color].Red +
                           image->g->SColorMap->Colors[tr_color].Green +
                           image->g->SColorMap->Colors[tr_color].Blue) / 3;
                moy = moy < 0 ? 0 : (moy > 255 ? 255 : moy);
                
                for (k = 0; k < n_colors; k++) {
                    if (moy == colormap[k].Red && moy == colormap[k].Green && moy == colormap[k].Blue) {
                        found = k;
                        break;
                    }
                }

                #pragma omp critical
                {
                    if (found == -1 && n_colors < 256) {
                        colormap[n_colors].Red = moy;
                        colormap[n_colors].Green = moy;
                        colormap[n_colors].Blue = moy;
                        image->g->ExtensionBlocks[j].Bytes[3] = n_colors;
                        n_colors++;
                    } else {
                        image->g->ExtensionBlocks[j].Bytes[3] = found;
                    }
                }
            }
        }
    }

    p = image->p;
    #pragma omp parallel for
    for (i = 0; i < image->n_images; i++) {
        for (j = 0; j < image->width[i] * image->height[i]; j++) {
            int found = 0;
            for (k = 0; k < n_colors; k++) {
                if (p[i][j].r == colormap[k].Red && p[i][j].g == colormap[k].Green && p[i][j].b == colormap[k].Blue) {
                    found = 1;
                    break;
                }
            }

            #pragma omp critical
            {
                if (!found && n_colors < 256) {
                    colormap[n_colors].Red = p[i][j].r;
                    colormap[n_colors].Green = p[i][j].g;
                    colormap[n_colors].Blue = p[i][j].b;
                    n_colors++;
                }
            }
        }
    }

    int bit_size = GifBitSize(n_colors);
    if (bit_size > 31) {
        fprintf(stderr, "Error: Bit size too large: %d\n", bit_size);
        return 0;
    }
    n_colors = (1 << bit_size);

    ColorMapObject *cmo = GifMakeMapObject(n_colors, colormap);
    if (cmo == NULL) {
        fprintf(stderr, "Error while creating a ColorMapObject w/ %d color(s)\n", n_colors);
        return 0;
    }
    image->g->SColorMap = cmo;

    if (!output_modified_read_gif(filename, image->g)) {
        return 0;
    }
    return 1;
}
char* build_output_path(const char *input_filename, const char *output_folder) {
    const char *filename = strrchr(input_filename, '/');
    if (filename) {
        filename++; 
    } else {
        filename = input_filename;
    }

    size_t path_length = strlen(output_folder) + 1 + strlen(filename) + 1;
    char *full_path = (char *)malloc(path_length);
    if (!full_path) {
        return NULL;
    }

    snprintf(full_path, path_length, "%s/%s", output_folder, filename);
    return full_path;
}

/*
 * Main entry point
 */
int 
main( int argc, char ** argv )
{
    /* initialize MPI */


    MPI_Init(&argc, &argv); 
    int rank, size, root = 0, num_file;

    animated_gif *image;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    char * input_filename; 
    char * output_filename;
    struct timeval t1, t2;
    double duration;
    num_file = argc - 2;
    output_filename = argv[argc - 1];

    if (rank <= num_file && rank != 0) {
        
        input_filename = argv[rank];
        printf("input_filename: %s\n", input_filename);
        gettimeofday(&t1, NULL);
        image = load_pixels(input_filename);
        if (image == NULL) { return 1; }
        gettimeofday(&t2, NULL);
        duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec - t1.tv_usec) / 1e6);
        printf("GIF loaded from file %s with %d image(s) in %lf s\n", 
            input_filename, image->n_images, duration);

        // Set the number of OpenMP threads based on the total number of pixels
        int total_pixels = 0;
        for (int i = 0; i < image->n_images; i++) {
            total_pixels += image->width[i] * image->height[i];
        }
        int num_threads = total_pixels / 10000; // Example: 1 thread per 10000 pixels
        if (num_threads < 1) num_threads = 1; // Ensure at least one thread
        omp_set_num_threads(num_threads);

        apply_gray_filter(image);
        
        apply_blur_filter(image, 5, 20);
        apply_sobel_filter(image);
        gettimeofday(&t2, NULL);
        duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec - t1.tv_usec) / 1e6);
        printf("SOBEL done in %lf s\n", duration);

        gettimeofday(&t1, NULL);
        output_filename = build_output_path(input_filename, output_filename);

        if (!store_pixels(output_filename, image)) { return 1; }
        gettimeofday(&t2, NULL);
        duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec - t1.tv_usec) / 1e6);
        printf("Export done in %lf s in file %s\n", duration, output_filename);
    }
    
    MPI_Finalize();
    return 0;
}

