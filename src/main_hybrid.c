#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <mpi.h> 
#include "gif_lib.h"
#include <omp.h>
#include<string.h>

#define SOBELF_DEBUG 0


#define SMALL_FRAME_THRESHOLD 100000 
#define LARGE_FRAME_THRESHOLD 500000  
#define MAX_FILENAME_LEN 128




typedef struct pixel
{
    int r ; /* Red */
    int g ; /* Green */
    int b ; /* Blue */
} pixel ;

/* Represent one GIF image (animated or not */
typedef struct animated_gif
{
    int n_images ; /* Number of images */
    int * width ; /* Width of each image */
    int * height ; /* Height of each image */
    pixel ** p ; /* Pixels of each image */
    GifFileType * g ; /* Internal representation.
                         DO NOT MODIFY */
} animated_gif ;





/* We can define a macro to access pixel array more safely */
#define CONV(l,c,nb_c) ((l)*(nb_c)+(c))



typedef struct {
    int is_small_batch;       /* 1 if this task is a batch of small files, 0 otherwise */
    int file_index;           /* if not a batch, which single file does this task for */
    int frame_index;          /* which frame: -1=all frames, or 0..n_images-1 */
    int start_row;          
    int height;
    int width;            
} gif_task;


#define TAG_TASK  101  /* Master->Worker: send a gif_task */
#define TAG_DONE  102  /* Worker->Master: done with a task */
#define TAG_QUIT  103  /* Master->Worker: no more tasks */


/*
 * Load a GIF image from a file and return a
 * structure of type animated_gif.
 */
animated_gif *
load_pixels( char * filename ) 
{
    GifFileType * g ;
    ColorMapObject * colmap ;
    int error ;
    int n_images ;
    int * width ;
    int * height ;
    pixel ** p ;
    int i ;
    animated_gif * image ;

    /* Open the GIF image (read mode) */
    g = DGifOpenFileName( filename, &error ) ;
    if ( g == NULL ) 
    {
        fprintf( stderr, "Error DGifOpenFileName %s\n", filename ) ;
        return NULL ;
    }

    /* Read the GIF image */
    error = DGifSlurp( g ) ;
    if ( error != GIF_OK )
    {
        fprintf( stderr, 
                "Error DGifSlurp: %d <%s>\n", error, GifErrorString(g->Error) ) ;
        return NULL ;
    }

    /* Grab the number of images and the size of each image */
    n_images = g->ImageCount ;

    width = (int *)malloc( n_images * sizeof( int ) ) ;
    if ( width == NULL )
    {
        fprintf( stderr, "Unable to allocate width of size %d\n",
                n_images ) ;
        return 0 ;
    }

    height = (int *)malloc( n_images * sizeof( int ) ) ;
    if ( height == NULL )
    {
        fprintf( stderr, "Unable to allocate height of size %d\n",
                n_images ) ;
        return 0 ;
    }

    /* Fill the width and height */
    for ( i = 0 ; i < n_images ; i++ ) 
    {
        width[i] = g->SavedImages[i].ImageDesc.Width ;
        height[i] = g->SavedImages[i].ImageDesc.Height ;

#if SOBELF_DEBUG
        printf( "Image %d: l:%d t:%d w:%d h:%d interlace:%d localCM:%p\n",
                i, 
                g->SavedImages[i].ImageDesc.Left,
                g->SavedImages[i].ImageDesc.Top,
                g->SavedImages[i].ImageDesc.Width,
                g->SavedImages[i].ImageDesc.Height,
                g->SavedImages[i].ImageDesc.Interlace,
                g->SavedImages[i].ImageDesc.ColorMap
                ) ;
#endif
    }


    /* Get the global colormap */
    colmap = g->SColorMap ;
    if ( colmap == NULL ) 
    {
        fprintf( stderr, "Error global colormap is NULL\n" ) ;
        return NULL ;
    }

#if SOBELF_DEBUG
    printf( "Global color map: count:%d bpp:%d sort:%d\n",
            g->SColorMap->ColorCount,
            g->SColorMap->BitsPerPixel,
            g->SColorMap->SortFlag
            ) ;
#endif

    /* Allocate the array of pixels to be returned */
    p = (pixel **)malloc( n_images * sizeof( pixel * ) ) ;
    if ( p == NULL )
    {
        fprintf( stderr, "Unable to allocate array of %d images\n",
                n_images ) ;
        return NULL ;
    }

    for ( i = 0 ; i < n_images ; i++ ) 
    {
        p[i] = (pixel *)malloc( width[i] * height[i] * sizeof( pixel ) ) ;
        if ( p[i] == NULL )
        {
        fprintf( stderr, "Unable to allocate %d-th array of %d pixels\n",
                i, width[i] * height[i] ) ;
        return NULL ;
        }
    }
    
    /* Fill pixels */

    /* For each image */
    for ( i = 0 ; i < n_images ; i++ )
    {
        int j ;

        /* Get the local colormap if needed */
        if ( g->SavedImages[i].ImageDesc.ColorMap )
        {

            /* TODO No support for local color map */
            fprintf( stderr, "Error: application does not support local colormap\n" ) ;
            return NULL ;

            colmap = g->SavedImages[i].ImageDesc.ColorMap ;
        }

        /* Traverse the image and fill pixels */
        for ( j = 0 ; j < width[i] * height[i] ; j++ ) 
        {
            int c ;

            c = g->SavedImages[i].RasterBits[j] ;

            p[i][j].r = colmap->Colors[c].Red ;
            p[i][j].g = colmap->Colors[c].Green ;
            p[i][j].b = colmap->Colors[c].Blue ;
        }
    }

    /* Allocate image info */
    image = (animated_gif *)malloc( sizeof(animated_gif) ) ;
    if ( image == NULL ) 
    {
        fprintf( stderr, "Unable to allocate memory for animated_gif\n" ) ;
        return NULL ;
    }

    /* Fill image fields */
    image->n_images = n_images ;
    image->width = width ;
    image->height = height ;
    image->p = p ;
    image->g = g ;

#if SOBELF_DEBUG
    printf( "-> GIF w/ %d image(s) with first image of size %d x %d\n",
            image->n_images, image->width[0], image->height[0] ) ;
#endif

    return image ;
}

int 
output_modified_read_gif( char * filename, GifFileType * g ) 
{
    GifFileType * g2 ;
    int error2 ;

#if SOBELF_DEBUG
    printf( "Starting output to file %s\n", filename ) ;
#endif

    g2 = EGifOpenFileName( filename, false, &error2 ) ;
    if ( g2 == NULL )
    {
        fprintf( stderr, "Error EGifOpenFileName %s\n",
                filename ) ;
        return 0 ;
    }

    g2->SWidth = g->SWidth ;
    g2->SHeight = g->SHeight ;
    g2->SColorResolution = g->SColorResolution ;
    g2->SBackGroundColor = g->SBackGroundColor ;
    g2->AspectByte = g->AspectByte ;
    g2->SColorMap = g->SColorMap ;
    g2->ImageCount = g->ImageCount ;
    g2->SavedImages = g->SavedImages ;
    g2->ExtensionBlockCount = g->ExtensionBlockCount ;
    g2->ExtensionBlocks = g->ExtensionBlocks ;

    error2 = EGifSpew( g2 ) ;
    if ( error2 != GIF_OK ) 
    {
        fprintf( stderr, "Error after writing g2: %d <%s>\n", 
                error2, GifErrorString(g2->Error) ) ;
        return 0 ;
    }

    return 1 ;
}

int
store_pixels( char * filename, animated_gif * image )
{
    int n_colors = 0 ;
    pixel ** p ;
    int i, j, k ;
    GifColorType * colormap ;

    /* Initialize the new set of colors */
    colormap = (GifColorType *)malloc( 256 * sizeof( GifColorType ) ) ;
    if ( colormap == NULL ) 
    {
        fprintf( stderr,
                "Unable to allocate 256 colors\n" ) ;
        return 0 ;
    }

    /* Everything is white by default */
    #pragma omp parallel for private(i)
    for ( i = 0 ; i < 256 ; i++ ) 
    {
        colormap[i].Red = 255 ;
        colormap[i].Green = 255 ;
        colormap[i].Blue = 255 ;
    }

    /* Change the background color and store it */
    int moy ;
    moy = (
            image->g->SColorMap->Colors[ image->g->SBackGroundColor ].Red
            +
            image->g->SColorMap->Colors[ image->g->SBackGroundColor ].Green
            +
            image->g->SColorMap->Colors[ image->g->SBackGroundColor ].Blue
            )/3 ;
    if ( moy < 0 ) moy = 0 ;
    if ( moy > 255 ) moy = 255 ;

#if SOBELF_DEBUG
    printf( "[DEBUG] Background color (%d,%d,%d) -> (%d,%d,%d)\n",
            image->g->SColorMap->Colors[ image->g->SBackGroundColor ].Red,
            image->g->SColorMap->Colors[ image->g->SBackGroundColor ].Green,
            image->g->SColorMap->Colors[ image->g->SBackGroundColor ].Blue,
            moy, moy, moy ) ;
#endif

    colormap[0].Red = moy ;
    colormap[0].Green = moy ;
    colormap[0].Blue = moy ;

    image->g->SBackGroundColor = 0 ;

    n_colors++ ;

    /* Process extension blocks in main structure */
    #pragma omp parallel for private(j, k, moy) shared(n_colors)
    for ( j = 0 ; j < image->g->ExtensionBlockCount ; j++ )
    {
        int f ;

        f = image->g->ExtensionBlocks[j].Function ;
        if ( f == GRAPHICS_EXT_FUNC_CODE )
        {
            int tr_color = image->g->ExtensionBlocks[j].Bytes[3] ;

            if ( tr_color >= 0 &&
                    tr_color < 255 )
            {

                int found = -1 ;

                moy = 
                    (
                     image->g->SColorMap->Colors[ tr_color ].Red
                     +
                     image->g->SColorMap->Colors[ tr_color ].Green
                     +
                     image->g->SColorMap->Colors[ tr_color ].Blue
                    ) / 3 ;
                if ( moy < 0 ) moy = 0 ;
                if ( moy > 255 ) moy = 255 ;

#if SOBELF_DEBUG
                printf( "[DEBUG] Transparency color image %d (%d,%d,%d) -> (%d,%d,%d)\n",
                        i,
                        image->g->SColorMap->Colors[ tr_color ].Red,
                        image->g->SColorMap->Colors[ tr_color ].Green,
                        image->g->SColorMap->Colors[ tr_color ].Blue,
                        moy, moy, moy ) ;
#endif
                #pragma omp critical 
                {
                    for ( k = 0 ; k < n_colors ; k++ )
                    {
                        if ( 
                                moy == colormap[k].Red
                                &&
                                moy == colormap[k].Green
                                &&
                                moy == colormap[k].Blue
                        )
                        {
                            found = k ;
                        }
                    }
                    if ( found == -1  ) 
                    {
                        if ( n_colors >= 256 ) 
                        {
                            fprintf( stderr, 
                                    "Error: Found too many colors inside the image\n"
                                ) ;
                            exit(1);
                        }

    #if SOBELF_DEBUG
                        printf( "[DEBUG]\tNew color %d\n",
                                n_colors ) ;
    #endif

                        colormap[n_colors].Red = moy ;
                        colormap[n_colors].Green = moy ;
                        colormap[n_colors].Blue = moy ;


                        image->g->ExtensionBlocks[j].Bytes[3] = n_colors ;

                        n_colors++ ;
                    } else
                    {
    #if SOBELF_DEBUG
                        printf( "[DEBUG]\tFound existing color %d\n",
                                found ) ;
    #endif
                        image->g->ExtensionBlocks[j].Bytes[3] = found ;
                    }
                }
            }
        }
    }

    for ( i = 0 ; i < image->n_images ; i++ )
    {
        for ( j = 0 ; j < image->g->SavedImages[i].ExtensionBlockCount ; j++ )
        {
            int f ;

            f = image->g->SavedImages[i].ExtensionBlocks[j].Function ;
            if ( f == GRAPHICS_EXT_FUNC_CODE )
            {
                int tr_color = image->g->SavedImages[i].ExtensionBlocks[j].Bytes[3] ;

                if ( tr_color >= 0 &&
                        tr_color < 255 )
                {

                    int found = -1 ;

                    moy = 
                        (
                         image->g->SColorMap->Colors[ tr_color ].Red
                         +
                         image->g->SColorMap->Colors[ tr_color ].Green
                         +
                         image->g->SColorMap->Colors[ tr_color ].Blue
                        ) / 3 ;
                    if ( moy < 0 ) moy = 0 ;
                    if ( moy > 255 ) moy = 255 ;

#if SOBELF_DEBUG
                    printf( "[DEBUG] Transparency color image %d (%d,%d,%d) -> (%d,%d,%d)\n",
                            i,
                            image->g->SColorMap->Colors[ tr_color ].Red,
                            image->g->SColorMap->Colors[ tr_color ].Green,
                            image->g->SColorMap->Colors[ tr_color ].Blue,
                            moy, moy, moy ) ;
#endif

                    for ( k = 0 ; k < n_colors ; k++ )
                    {
                        if ( 
                                moy == colormap[k].Red
                                &&
                                moy == colormap[k].Green
                                &&
                                moy == colormap[k].Blue
                           )
                        {
                            found = k ;
                        }
                    }
                    if ( found == -1  ) 
                    {
                        if ( n_colors >= 256 ) 
                        {
                            fprintf( stderr, 
                                    "Error: Found too many colors inside the image\n"
                                   ) ;
                            return 0 ;
                        }

#if SOBELF_DEBUG
                        printf( "[DEBUG]\tNew color %d\n",
                                n_colors ) ;
#endif

                        colormap[n_colors].Red = moy ;
                        colormap[n_colors].Green = moy ;
                        colormap[n_colors].Blue = moy ;


                        image->g->SavedImages[i].ExtensionBlocks[j].Bytes[3] = n_colors ;

                        n_colors++ ;
                    } else
                    {
#if SOBELF_DEBUG
                        printf( "[DEBUG]\tFound existing color %d\n",
                                found ) ;
#endif
                        image->g->SavedImages[i].ExtensionBlocks[j].Bytes[3] = found ;
                    }
                }
            }
        }
    }

#if SOBELF_DEBUG
    printf( "[DEBUG] Number of colors after background and transparency: %d\n",
            n_colors ) ;
#endif

    p = image->p ;

    /* Find the number of colors inside the image */
    for ( i = 0 ; i < image->n_images ; i++ )
    {

#if SOBELF_DEBUG
        printf( "OUTPUT: Processing image %d (total of %d images) -> %d x %d\n",
                i, image->n_images, image->width[i], image->height[i] ) ;
#endif

        for ( j = 0 ; j < image->width[i] * image->height[i] ; j++ ) 
        {
            int found = 0 ;
            for ( k = 0 ; k < n_colors ; k++ )
            {
                if ( p[i][j].r == colormap[k].Red &&
                        p[i][j].g == colormap[k].Green &&
                        p[i][j].b == colormap[k].Blue )
                {
                    found = 1 ;
                }
            }

            if ( found == 0 ) 
            {
                if ( n_colors >= 256 ) 
                {
                    fprintf( stderr, 
                            "Error: Found too many colors inside the image\n"
                           ) ;
                    return 0 ;
                }

#if SOBELF_DEBUG
                printf( "[DEBUG] Found new %d color (%d,%d,%d)\n",
                        n_colors, p[i][j].r, p[i][j].g, p[i][j].b ) ;
#endif

                colormap[n_colors].Red = p[i][j].r ;
                colormap[n_colors].Green = p[i][j].g ;
                colormap[n_colors].Blue = p[i][j].b ;
                n_colors++ ;
            }
        }
    }

#if SOBELF_DEBUG
    printf( "OUTPUT: found %d color(s)\n", n_colors ) ;
#endif


    /* Round up to a power of 2 */
    if ( n_colors != (1 << GifBitSize(n_colors) ) )
    {
        n_colors = (1 << GifBitSize(n_colors) ) ;
    }

#if SOBELF_DEBUG
    printf( "OUTPUT: Rounding up to %d color(s)\n", n_colors ) ;
#endif

    /* Change the color map inside the animated gif */
    ColorMapObject * cmo ;

    cmo = GifMakeMapObject( n_colors, colormap ) ;
    if ( cmo == NULL )
    {
        fprintf( stderr, "Error while creating a ColorMapObject w/ %d color(s)\n",
                n_colors ) ;
        return 0 ;
    }

    image->g->SColorMap = cmo ;

    /* Update the raster bits according to color map */
    for ( i = 0 ; i < image->n_images ; i++ )
    {
        for ( j = 0 ; j < image->width[i] * image->height[i] ; j++ ) 
        {
            int found_index = -1 ;
            for ( k = 0 ; k < n_colors ; k++ ) 
            {
                if ( p[i][j].r == image->g->SColorMap->Colors[k].Red &&
                        p[i][j].g == image->g->SColorMap->Colors[k].Green &&
                        p[i][j].b == image->g->SColorMap->Colors[k].Blue )
                {
                    found_index = k ;
                }
            }

            if ( found_index == -1 ) 
            {
                fprintf( stderr,
                        "Error: Unable to find a pixel in the color map\n" ) ;
                return 0 ;
            }

            image->g->SavedImages[i].RasterBits[j] = found_index ;
        }
    }


    /* Write the final image */
    if ( !output_modified_read_gif( filename, image->g ) ) { return 0 ; }

    return 1 ;
}

void
apply_gray_filter( animated_gif * image )
{
    int i, j ;
    pixel ** p ;

    p = image->p ;

    for ( i = 0 ; i < image->n_images ; i++ )
    {
        #pragma omp parallel private(j)
        {
            #pragma omp for schedule(static)
            for ( j = 0 ; j < image->width[i] * image->height[i] ; j++ )
            {
                int moy ;
                moy = (p[i][j].r + p[i][j].g + p[i][j].b)/3 ;
                if ( moy < 0 ) moy = 0 ;
                if ( moy > 255 ) moy = 255 ;

                p[i][j].r = moy ;
                p[i][j].g = moy ;
                p[i][j].b = moy ;
            }
        }
    }
}


void apply_gray_line( animated_gif * image ) 
{
    int i, j, k ;
    pixel ** p ;

    p = image->p ;

    for ( i = 0 ; i < image->n_images ; i++ )
    {
        for ( j = 0 ; j < 10 ; j++ )
        {
            for ( k = image->width[i]/2 ; k < image->width[i] ; k++ )
            {
            p[i][CONV(j,k,image->width[i])].r = 0 ;
            p[i][CONV(j,k,image->width[i])].g = 0 ;
            p[i][CONV(j,k,image->width[i])].b = 0 ;
            }
        }
    }
}
void
apply_blur_filter( animated_gif * image, int size, int threshold )
{
    int i, j, k ;
    int width, height ;
    int end = 0 ;
    int n_iter = 0 ;

    pixel ** p ;
    pixel * new ;

    /* Get the pixels of all images */
    p = image->p ;


    /* Process all images */
    for ( i = 0 ; i < image->n_images ; i++ )
    {
        n_iter = 0 ;
        width = image->width[i] ;
        height = image->height[i] ;

        /* Allocate array of new pixels */
        new = (pixel *)malloc(width * height * sizeof( pixel ) ) ;


        /* Perform at least one blur iteration */
        do
        {
            end = 1 ;
            n_iter++ ;

    #pragma omp parallel for collapse(1) private(j,k) schedule(static, 1)
	for(j=0; j<height-1; j++)
	{
		for(k=0; k<width-1; k++)
		{
			new[CONV(j,k,width)].r = p[i][CONV(j,k,width)].r ;
			new[CONV(j,k,width)].g = p[i][CONV(j,k,width)].g ;
			new[CONV(j,k,width)].b = p[i][CONV(j,k,width)].b ;
		}
	}
            #pragma omp parallel 
            {
                /* Apply blur on top part of image (10%) */
                #pragma omp for collapse(1) private(j,k) schedule(static, 1)
                for(j=size; j<height/10-size; j++)
                {
                    for(k=size; k<width-size; k++)
                    {
                        int stencil_j, stencil_k ;
                        int t_r = 0 ;
                        int t_g = 0 ;
                        int t_b = 0 ;

                        for ( stencil_j = -size ; stencil_j <= size ; stencil_j++ )
                        {
                            for ( stencil_k = -size ; stencil_k <= size ; stencil_k++ )
                            {
                                t_r += p[i][CONV(j+stencil_j,k+stencil_k,width)].r ;
                                t_g += p[i][CONV(j+stencil_j,k+stencil_k,width)].g ;
                                t_b += p[i][CONV(j+stencil_j,k+stencil_k,width)].b ;
                            }
                        }

                        new[CONV(j,k,width)].r = t_r / ( (2*size+1)*(2*size+1) ) ;
                        new[CONV(j,k,width)].g = t_g / ( (2*size+1)*(2*size+1) ) ;
                        new[CONV(j,k,width)].b = t_b / ( (2*size+1)*(2*size+1) ) ;
                    }
                }

                /* Copy the middle part of the image */
                #pragma omp for collapse(1) private(j,k) schedule(static, 1) nowait
                for(j=height/10-size; j<(int)(height*0.9+size); j++)
                {
                    for(k=size; k<width-size; k++)
                    {
                        new[CONV(j,k,width)].r = p[i][CONV(j,k,width)].r ; 
                        new[CONV(j,k,width)].g = p[i][CONV(j,k,width)].g ; 
                        new[CONV(j,k,width)].b = p[i][CONV(j,k,width)].b ; 
                    }
                }

                /* Apply blur on the bottom part of the image (10%) */
                #pragma omp for collapse(1) private(j,k) schedule(static, 1) nowait
                for(j=height*0.9+size; j<height-size; j++)
                {
                    for(k=size; k<width-size; k++)
                    {
                        int stencil_j, stencil_k ;
                        int t_r = 0 ;
                        int t_g = 0 ;
                        int t_b = 0 ;

                        for ( stencil_j = -size ; stencil_j <= size ; stencil_j++ )
                        {
                            for ( stencil_k = -size ; stencil_k <= size ; stencil_k++ )
                            {
                                t_r += p[i][CONV(j+stencil_j,k+stencil_k,width)].r ;
                                t_g += p[i][CONV(j+stencil_j,k+stencil_k,width)].g ;
                                t_b += p[i][CONV(j+stencil_j,k+stencil_k,width)].b ;
                            }
                        }

                        new[CONV(j,k,width)].r = t_r / ( (2*size+1)*(2*size+1) ) ;
                        new[CONV(j,k,width)].g = t_g / ( (2*size+1)*(2*size+1) ) ;
                        new[CONV(j,k,width)].b = t_b / ( (2*size+1)*(2*size+1) ) ;
                    }
                }
            }

            #pragma omp parallel for collapse(1) schedule(static, 1) private(j,k)
            for(j=1; j<height-1; j++)
            {
                for(k=1; k<width-1; k++)
                {

                    float diff_r ;
                    float diff_g ;
                    float diff_b ;

                    diff_r = (new[CONV(j  ,k  ,width)].r - p[i][CONV(j  ,k  ,width)].r) ;
                    diff_g = (new[CONV(j  ,k  ,width)].g - p[i][CONV(j  ,k  ,width)].g) ;
                    diff_b = (new[CONV(j  ,k  ,width)].b - p[i][CONV(j  ,k  ,width)].b) ;

                    if ( diff_r > threshold || -diff_r > threshold 
                            ||
                             diff_g > threshold || -diff_g > threshold
                             ||
                              diff_b > threshold || -diff_b > threshold
                       ) {
                        end = 0 ;
                    }

                    p[i][CONV(j  ,k  ,width)].r = new[CONV(j  ,k  ,width)].r ;
                    p[i][CONV(j  ,k  ,width)].g = new[CONV(j  ,k  ,width)].g ;
                    p[i][CONV(j  ,k  ,width)].b = new[CONV(j  ,k  ,width)].b ;
                }
            }

        }
        while ( threshold > 0 && !end ) ;

#if SOBELF_DEBUG
	printf( "BLUR: number of iterations for image %d\n", n_iter ) ;
#endif

        free (new) ;
    }

}

void
apply_sobel_filter( animated_gif * image )
{
    int i, j, k ;
    int width, height ;

    pixel ** p ;

    p = image->p ;

    for ( i = 0 ; i < image->n_images ; i++ )
    {
        width = image->width[i] ;
        height = image->height[i] ;

        pixel * sobel ;

        sobel = (pixel *)malloc(width * height * sizeof( pixel ) ) ;
        #pragma omp parallel for collapse(2) private(j,k)
        for(j=1; j<height-1; j++)
        {
            for(k=1; k<width-1; k++)
            {
                int pixel_blue_no, pixel_blue_n, pixel_blue_ne;
                int pixel_blue_so, pixel_blue_s, pixel_blue_se;
                int pixel_blue_o , pixel_blue  , pixel_blue_e ;

                float deltaX_blue ;
                float deltaY_blue ;
                float val_blue;

                pixel_blue_no = p[i][CONV(j-1,k-1,width)].b ;
                pixel_blue_n  = p[i][CONV(j-1,k  ,width)].b ;
                pixel_blue_ne = p[i][CONV(j-1,k+1,width)].b ;
                pixel_blue_so = p[i][CONV(j+1,k-1,width)].b ;
                pixel_blue_s  = p[i][CONV(j+1,k  ,width)].b ;
                pixel_blue_se = p[i][CONV(j+1,k+1,width)].b ;
                pixel_blue_o  = p[i][CONV(j  ,k-1,width)].b ;
                pixel_blue    = p[i][CONV(j  ,k  ,width)].b ;
                pixel_blue_e  = p[i][CONV(j  ,k+1,width)].b ;

                deltaX_blue = -pixel_blue_no + pixel_blue_ne - 2*pixel_blue_o + 2*pixel_blue_e - pixel_blue_so + pixel_blue_se;             

                deltaY_blue = pixel_blue_se + 2*pixel_blue_s + pixel_blue_so - pixel_blue_ne - 2*pixel_blue_n - pixel_blue_no;

                val_blue = sqrt(deltaX_blue * deltaX_blue + deltaY_blue * deltaY_blue)/4;


                if ( val_blue > 50 ) 
                {
                    sobel[CONV(j  ,k  ,width)].r = 255 ;
                    sobel[CONV(j  ,k  ,width)].g = 255 ;
                    sobel[CONV(j  ,k  ,width)].b = 255 ;
                } else
                {
                    sobel[CONV(j  ,k  ,width)].r = 0 ;
                    sobel[CONV(j  ,k  ,width)].g = 0 ;
                    sobel[CONV(j  ,k  ,width)].b = 0 ;
                }
            }
        }

        for(j=1; j<height-1; j++)
        {
            for(k=1; k<width-1; k++)
            {
                p[i][CONV(j  ,k  ,width)].r = sobel[CONV(j  ,k  ,width)].r ;
                p[i][CONV(j  ,k  ,width)].g = sobel[CONV(j  ,k  ,width)].g ;
                p[i][CONV(j  ,k  ,width)].b = sobel[CONV(j  ,k  ,width)].b ;
            }
        }

        free (sobel) ;
    }

}
/* ---------------- build_output_path ---------------- */
char* build_output_path(const char *input_filename, const char *output_folder)
{
    const char *filename = strrchr(input_filename, '/');
    if (filename) filename++;
    else filename = input_filename;

    size_t path_length = strlen(output_folder) + 1 + strlen(filename) + 1;
    char *full_path = (char *)malloc(path_length);
    if (!full_path) {
        return NULL;
    }
    snprintf(full_path, path_length, "%s/%s", output_folder, filename);
    return full_path;
}


static void create_tasks_for_file(
    int file_index, 
    const char* filename, 
    int *total_pixels_array, 
    int n_images_array[], 
    int widths_array[][33], 
    int heights_array[][33], 
    gif_task **tasks_ref, 
    int *task_count_ref, 
    int *task_capacity
)
{
    /* 
     * This helper function will examine one GIF file's info, 
     * see how many frames, how large they are, 
     * then produce tasks accordingly and store them in (*tasks_ref).
     */
    int i, j;
    int total_pixels = total_pixels_array[file_index];
    int n_images = n_images_array[file_index];

    /* If total pixels < SMALL_FRAME_THRESHOLD => we won't create a normal task,
     * we let the main do batch merging.
     * So we do nothing here in that case. We only create tasks if large enough.
     */
    if (total_pixels < SMALL_FRAME_THRESHOLD) {
        // not a task, let thread in rank 0 handle this file
        return;
    }


    /* For each frame, check size. If frame is large => split by block. */
    for (i = 0; i < n_images; i++) {
        long frame_pixels = (long)widths_array[file_index][i] * heights_array[file_index][i];
        if (frame_pixels > LARGE_FRAME_THRESHOLD) {
            /* Split this frame into multiple blocks by row. */

            int block_count = frame_pixels / LARGE_FRAME_THRESHOLD + 1;
            int gap = heights_array[file_index][i] / block_count;
            int res = heights_array[file_index][i] % block_count;
            if (gap < 1) block_count = res;
            for (int b = 0; b < block_count; b++) {
                gif_task task;
                memset(&task, 0, sizeof(task));
                task.is_small_batch = 0;
                task.file_index  = file_index;
                task.frame_index = i;
                task.start_row = b * gap;
                task.width = widths_array[file_index][i];
                task.height = (b<res) ? gap+1 : gap;

                /* push task */
                if ((*task_count_ref) >= (*task_capacity)) {
                    (*task_capacity) *= 2;
                    (*tasks_ref) = (gif_task*)realloc((*tasks_ref), (*task_capacity)*sizeof(gif_task));
                }
                (*tasks_ref)[ (*task_count_ref)++ ] = task;
            }
        } else {
            /* single frame => one task */
            gif_task task;
            memset(&task, 0, sizeof(task));
            task.is_small_batch = 0;
            task.file_index  = file_index;
            task.frame_index = i;
            task.start_row = 0;
            task.width = widths_array[file_index][i];
            task.height = heights_array[file_index][i];
            if ((*task_count_ref) >= (*task_capacity)) {
                (*task_capacity) *= 2;
                (*tasks_ref) = (gif_task*)realloc((*tasks_ref), (*task_capacity)*sizeof(gif_task));
            }
            (*tasks_ref)[ (*task_count_ref)++ ] = task;
        }

    }
}
/* ---- The main function ---- */
    

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    struct timeval tstart, tend;
    
    if(argc<3){
        if(rank==0){
            fprintf(stderr,"Usage: mpirun -np X %s <output_folder> <gif_file1> [gif_file2] ...\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }
    if(rank==0){
        printf("Running with %d MPI processes.\n", size);
        gettimeofday(&tstart, NULL);
    }
    char *output_folder=argv[1];
    int num_files=argc-2; // number of input GIF files
    char **file_names=&argv[2]; // file_names[0] -> first GIF, file_names[1] -> second, etc.

    int *total_pixels_array=NULL;
    int *n_images_array=NULL;
    int (*widths_array)[33]=NULL;   // assume max 33 frames
    int (*heights_array)[33]=NULL;  // same
    animated_gif **images;
    int task_capacity=0;

 
    if(rank==0){
        total_pixels_array=(int*)malloc(num_files*sizeof(int));
        n_images_array=(int*)malloc(num_files*sizeof(int));
        widths_array = calloc(num_files, sizeof(*widths_array));
        heights_array=calloc(num_files, sizeof(*heights_array));
        images=(animated_gif**)malloc(num_files*sizeof(animated_gif*));
        
        for(int f=0; f<num_files; f++){
            images[f]=load_pixels(file_names[f]);
            animated_gif *img=images[f];
            if(!img){
                fprintf(stderr,"[Master] Cannot load file %s.\n", file_names[f]);
                // We'll set total_pixels=0 to skip
                total_pixels_array[f]=0;
                n_images_array[f]=0;
                continue;
            }
            int sum_pixels=0;
            n_images_array[f]=img->n_images;
            for(int i=0; i<img->n_images; i++){
                widths_array[f][i] = img->width[i];
                heights_array[f][i]= img->height[i];
                sum_pixels += img->width[i]*img->height[i];
            }
            total_pixels_array[f]=sum_pixels;
            task_capacity += sum_pixels/LARGE_FRAME_THRESHOLD + 1;

            if (total_pixels_array[f] < SMALL_FRAME_THRESHOLD) {
                #pragma omp task 
                {
                    apply_gray_filter(images[f]);
                    apply_blur_filter(images[f], 5, 50);
                    apply_sobel_filter(images[f]);
                    char *output_path = build_output_path(file_names[f], output_folder);
                    if (!output_path) {
                        fprintf(stderr, "Error: failed to build output path for %s\n", file_names[f]);
                    } else {
                        store_pixels(output_path, images[f]);
                        free(output_path);
                    }
                }
            }

        }
    }


    if(rank==0){
        /* Master: build a big 'task list' from metadata */
        gif_task *tasks=(gif_task*)malloc(task_capacity*sizeof(gif_task));
        if (tasks == NULL) {
            fprintf(stderr, "Error: Memory allocation failed! task_capacity=%d\n", task_capacity);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        int task_count=0;

        /* We will handle small-GIF merging with a 'batch' approach:
         * gather small files in a temp array, once we have enough or we see a bigger file, 
         * we produce a batch task.
         */

        for(int f=0; f<num_files; f++){
            int tp=total_pixels_array[f];
            if(tp<=0) continue; // skip broken
            if(tp<SMALL_FRAME_THRESHOLD){
                continue; // skip small ones, we'll handle them in batch
                
            } else {
                /* create tasks for this file (maybe multi-frame or splitted) */
                create_tasks_for_file(f, file_names[f], total_pixels_array, n_images_array, widths_array, heights_array,
                                       &tasks, &task_count, &task_capacity);
                
            }
        }

        


        // now we have 'task_count' tasks in 'tasks[]'.
        // dynamic distribution:
        int next_task=0;
        int active_workers= (size>1? size-1 : 0);

        // if there's no worker at all (size=1?), just do everything in master
        if(active_workers==0){
            // fallback: master runs all tasks
            for(int i=0; i<task_count; i++){
                fprintf(stderr,"[Master-only] no worker found. Not implemented.\n");
            }
        } else {
        
            // send out initial tasks to each worker
            for(int w=1; w<=active_workers; w++){
                if(next_task < task_count){
                    int len = tasks[next_task].height * tasks[next_task].width;
                    pixel* Send_pixel = images[tasks[next_task].file_index]->p[tasks[next_task].frame_index] 
                    + tasks[next_task].start_row * tasks[next_task].width;

                    int target_worker = next_task % active_workers + 1;
                    
                    MPI_Request sreq1, sreq2;
                    MPI_Isend(&tasks[next_task], sizeof(gif_task), MPI_BYTE, 
                              target_worker, TAG_TASK, MPI_COMM_WORLD, &sreq1);
                    MPI_Isend(Send_pixel, len * sizeof(pixel), MPI_BYTE, 
                              target_worker, TAG_TASK, MPI_COMM_WORLD, &sreq2);
                    next_task++;
                } else {
                    // no more tasks => send quit
                    gif_task dummy;
                    memset(&dummy, 0, sizeof(dummy));
                    MPI_Send(&dummy, sizeof(dummy), MPI_BYTE, w, TAG_QUIT, MPI_COMM_WORLD);
                }
            }
        
            int tasks_done=0;
            printf("[Master] Waiting for %d tasks to be done...\n", task_count);
            while(tasks_done < task_count){
                // wait for a done message
                gif_task done_task;
                MPI_Request rreq1, rreq2;
                MPI_Status st1, st2;
        
                
                MPI_Irecv(&done_task, sizeof(done_task), MPI_BYTE, MPI_ANY_SOURCE, TAG_DONE, 
                          MPI_COMM_WORLD, &rreq1);
                
                MPI_Wait(&rreq1, &st1);
        
                if (done_task.height <= 0 || done_task.width <= 0) {
                    fprintf(stderr, "Error: Received invalid task dimensions: height=%d, width=%d\n", 
                            done_task.height, done_task.width);
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                
                int len = done_task.height * done_task.width;
                pixel *received_pixels = (pixel *)malloc(len * sizeof(pixel));
        
                
                MPI_Irecv(received_pixels, len*sizeof(pixel), MPI_BYTE,
                          st1.MPI_SOURCE, TAG_DONE, MPI_COMM_WORLD, &rreq2);
                MPI_Wait(&rreq2, &st2);
                
                
                
                memcpy(
                    images[done_task.file_index]->p[done_task.frame_index] 
                      + done_task.start_row * done_task.width,
                    received_pixels,
                    len * sizeof(pixel)
                );
                free(received_pixels);
        
                int worker = st1.MPI_SOURCE; 
                tasks_done++;
        
                // send next task if any
                if(next_task < task_count){
                    if (tasks[next_task].height <= 0 || tasks[next_task].width <= 0) {
                        fprintf(stderr, "Error: Received invalid task dimensions: height=%d, width=%d\n", 
                            tasks[next_task].height, tasks[next_task].width);
                        MPI_Abort(MPI_COMM_WORLD, 1);
                    }
                    int len = tasks[next_task].height * tasks[next_task].width;

                    pixel* Send_pixel = images[tasks[next_task].file_index]->p[tasks[next_task].frame_index]
                    + tasks[next_task].start_row * tasks[next_task].width;

        
                    int target_worker = worker;

                    MPI_Request sreq3, sreq4;
                    MPI_Isend(&tasks[next_task], sizeof(gif_task), MPI_BYTE, 
                              target_worker, TAG_TASK, MPI_COMM_WORLD, &sreq3);
                    MPI_Isend(Send_pixel, len*sizeof(pixel), MPI_BYTE, 
                              target_worker, TAG_TASK, MPI_COMM_WORLD, &sreq4);
        
                    next_task++;
                } else {
                    // send quit
                    gif_task dummy;
                    memset(&dummy, 0, sizeof(dummy));
                    MPI_Send(&dummy, sizeof(dummy), MPI_BYTE, worker, TAG_QUIT, MPI_COMM_WORLD);
                }
            }
        }


        // Free memory since we only loaded metadata
        #pragma omp parallel for schedule(static, 1)
        for (int f = 0; f < num_files; f++) {
            if (images[f]) {

                store_pixels(build_output_path(file_names[f], output_folder), images[f]);
                free(images[f]->width);
                free(images[f]->height);
                for (int i = 0; i < images[f]->n_images; i++) {
                    free(images[f]->p[i]);
                }
                free(images[f]->p);
                DGifCloseFile(images[f]->g, NULL); // Close the GIF
                free(images[f]);
            }
        }
        free(images);
        
    } else {
        /* Worker ranks: repeatedly receive tasks and execute them. */
        while(1){
            gif_task tsk;
            MPI_Status st;
            pixel* Recv_pixel;
            MPI_Recv(&tsk, sizeof(tsk), MPI_BYTE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &st);
            
            if(st.MPI_TAG==TAG_QUIT){
                // no more tasks, break
                break;
            }
            int len = tsk.height * tsk.width;
            
            Recv_pixel = (pixel *)malloc(len * sizeof(pixel));
            MPI_Recv(Recv_pixel, len * sizeof(pixel), MPI_BYTE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &st);
            
            // process this task

            int fidx=tsk.file_index;
            char *fname=file_names[fidx];
            animated_gif *Task_image = (animated_gif *)malloc(sizeof(animated_gif));
            if (!Task_image) {
                fprintf(stderr, "Error: Unable to allocate memory for Task_image\n");
                free(Recv_pixel);
                continue;
            }



            Task_image->g = NULL;
            Task_image->p = (pixel **)malloc(sizeof(pixel *));

            Task_image->p[0] = Recv_pixel;
            Task_image->n_images = 1;
            Task_image->width = (int *)malloc(sizeof(int));
            Task_image->height = (int *)malloc(sizeof(int));

            Task_image->width[0] = tsk.width;
            Task_image->height[0] = tsk.height;

            // Process the task
            
            int nth = len / LARGE_FRAME_THRESHOLD;
            if (nth < 1) nth = 1;
            if (nth > omp_get_max_threads()) nth = omp_get_max_threads();
            omp_set_num_threads(nth);

            apply_gray_filter(Task_image);
            apply_blur_filter(Task_image, 5, 50);
            apply_sobel_filter(Task_image);


            MPI_Request sreq1, sreq2;
            MPI_Isend(&tsk, sizeof(gif_task), MPI_BYTE, 
            0, TAG_DONE, MPI_COMM_WORLD, &sreq1);
  
            
            MPI_Isend(Recv_pixel, len*sizeof(pixel), MPI_BYTE, 
                        0, TAG_DONE, MPI_COMM_WORLD, &sreq2);
            
            
            MPI_Waitall(2, (MPI_Request[]){sreq1, sreq2}, MPI_STATUSES_IGNORE);
            
            free(Recv_pixel); 
        }
    }
            
            
    if(rank==0){
        free(total_pixels_array);
        free(n_images_array);
        free(widths_array);
        free(heights_array);
        gettimeofday(&tend, NULL);
        double duration = ((double)tend.tv_sec + 1.0e-6*tend.tv_usec) - 
        ((double)tstart.tv_sec + 1.0e-6*tstart.tv_usec);
        printf("Total time: %.3f seconds\n", duration);
    }
    MPI_Finalize();
    return 0;
}