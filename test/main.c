/*
 * INF560
 *
 * Image Filtering Project
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <mpi.h> 
#include "gif_lib.h"
#include <omp.h>
#include<string.h>
/* Set this macro to 1 to enable debugging information */
#define SOBELF_DEBUG 0
#define SMALL_FRAME_THRESHOLD 50000      // 小帧：低于 50k 像素（可合并处理）
#define LARGE_FRAME_THRESHOLD 1000000    // 超大帧：超过 1e6 像素，需要拆分为块
#define BLOCK_HEIGHT 100                 // 拆分块的高度（行数）

/* Represent one pixel from the image */
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

#define CONV(l,c,nb_c) \
    (l)*(nb_c)+(c)

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

// int 
// main( int argc, char ** argv )
// {
//     /* initialize MPI */


//     MPI_Init(&argc, &argv); 
//     int rank, size, root = 0, num_file;

//     animated_gif *image;
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//     MPI_Comm_size(MPI_COMM_WORLD, &size);
//     char * input_filename; 
//     char * output_filename;
//     struct timeval t1, t2;
//     double duration;
//     num_file = argc - 2;
//     output_filename = argv[argc - 1];

//     if (rank <= num_file && rank != 0) {
        
//         input_filename = argv[rank];
//         gettimeofday(&t1, NULL);
//         image = load_pixels(input_filename);
//         if (image == NULL) { return 1; }
//         gettimeofday(&t2, NULL);
//         duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec - t1.tv_usec) / 1e6);
//         printf("GIF loaded from file %s with %d image(s) in %lf s\n", 
//             input_filename, image->n_images, duration);

//         // Set the number of OpenMP threads based on the total number of pixels
//         int total_pixels = 0;
//         for (int i = 0; i < image->n_images; i++) {
//             total_pixels += image->width[i] * image->height[i];
//         }
//         int num_threads = total_pixels / 1000000; 
//         if (num_threads < 1) num_threads = 1; 
//         omp_set_num_threads(num_threads);

//         apply_gray_filter(image);
        
//         apply_blur_filter(image, 5, 20);
//         apply_sobel_filter(image);
//         gettimeofday(&t2, NULL);
//         duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec - t1.tv_usec) / 1e6);
//         printf("SOBEL done in %lf s\n", duration);

//         gettimeofday(&t1, NULL);
//         output_filename = build_output_path(input_filename, output_filename);
//         if (!store_pixels(output_filename, image))
//         {
//             return 1;
//         }
//         gettimeofday(&t2, NULL);
//         duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec - t1.tv_usec) / 1e6);
//         printf("Export done in %lf s in file %s\n", duration, output_filename);
//     }
    
//     MPI_Finalize();
//     return 0;
// }
int main(int argc, char ** argv) {
    /* 初始化 MPI */
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    char *output_filename = argv[argc - 1];
    int num_files = argc - 2;  // 总文件数

    struct timeval t1, t2;
    double duration;

    if (rank == 0) {
        /* Rank 0 解析所有 GIF 文件信息 */
        int *task_pixels = (int *)malloc(num_files * sizeof(int));
        char **file_names = (char **)malloc(num_files * sizeof(char *));

        for (int i = 0; i < num_files; i++) {
            file_names[i] = argv[i + 1];
            animated_gif *temp_image = load_pixels(file_names[i]);
            if (temp_image == NULL) {
                fprintf(stderr, "Error loading file: %s\n", file_names[i]);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            // 计算 GIF 总像素数
            task_pixels[i] = 0;
            for (int j = 0; j < temp_image->n_images; j++) {
                task_pixels[i] += temp_image->width[j] * temp_image->height[j];
            }
        }

        /* Rank 0 动态分发任务 */
        int current_task = 0;
        for (int i = 1; i < size; i++) {
            if (current_task < num_files) {
                MPI_Send(&current_task, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                MPI_Send(file_names[current_task], 256, MPI_CHAR, i, 0, MPI_COMM_WORLD);
                current_task++;
            }
        }

        /* Rank 0 监听 Worker 完成任务并继续派发 */
        int completed_task;
        while (current_task < num_files) {
            MPI_Recv(&completed_task, 1, MPI_INT, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(&current_task, 1, MPI_INT, completed_task, 0, MPI_COMM_WORLD);
            MPI_Send(file_names[current_task], 256, MPI_CHAR, completed_task, 0, MPI_COMM_WORLD);
            current_task++;
        }

        /* 发送终止信号 */
        int stop_signal = -1;
        for (int i = 1; i < size; i++) {
            MPI_Send(&stop_signal, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        }

        free(task_pixels);
        free(file_names);
    } else {
        while (1) {
            int file_id;
            char input_filename[256];

            MPI_Recv(&file_id, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (file_id == -1) break; // 终止信号

            MPI_Recv(input_filename, 256, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            printf("Rank %d processing file: %s\n", rank, input_filename);
            gettimeofday(&t1, NULL);
            animated_gif *image = load_pixels(input_filename);
            if (image == NULL) {
                fprintf(stderr, "Error loading file: %s\n", input_filename);
                continue;
            }
            gettimeofday(&t2, NULL);
            duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec - t1.tv_usec) / 1e6);
            printf("GIF loaded in %lf s\n", duration);

            /* 计算 OpenMP 线程数 */
            int total_pixels = 0;
            for (int i = 0; i < image->n_images; i++) {
                total_pixels += image->width[i] * image->height[i];
            }
            int num_threads = total_pixels / 500000;  // 50,000 pixels per thread
            if (num_threads < 1) num_threads = 1;
            omp_set_num_threads(num_threads);

            /* 并行处理 GIF */
            apply_gray_filter(image);
            apply_blur_filter(image, 5, 20);
            apply_sobel_filter(image);
            gettimeofday(&t2, NULL);
            duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec - t1.tv_usec) / 1e6);
            printf("SOBEL done in %lf s\n", duration);

            gettimeofday(&t1, NULL);
            char *output_path = build_output_path(input_filename, output_filename);
            if (!store_pixels(output_path, image)) {
                fprintf(stderr, "Error storing file: %s\n", output_path);
                continue;
            }
            gettimeofday(&t2, NULL);
            duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec - t1.tv_usec) / 1e6);
            printf("Exported in %lf s to file %s\n", duration, output_path);

            /* 通知 Rank 0 任务完成 */
            MPI_Send(&rank, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
    return 0;
}

