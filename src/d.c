/*
 * INF560
 *
 * Image Filtering Project
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#include "gif_lib.h"

#include <mpi.h> 


/* Set this macro to 1 to enable debugging information */
#define SOBELF_DEBUG 0

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



void
apply_gray_filter( animated_gif * image )
{
    int i, j ;
    pixel ** p ;

    p = image->p ;

    for ( i = 0 ; i < image->n_images ; i++ )
    {
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


	for(j=0; j<height-1; j++)
	{
		for(k=0; k<width-1; k++)
		{
			new[CONV(j,k,width)].r = p[i][CONV(j,k,width)].r ;
			new[CONV(j,k,width)].g = p[i][CONV(j,k,width)].g ;
			new[CONV(j,k,width)].b = p[i][CONV(j,k,width)].b ;
		}
	}

            /* Apply blur on top part of image (10%) */
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
            for(j=height/10-size; j<height*0.9+size; j++)
            {
                for(k=size; k<width-size; k++)
                {
                    new[CONV(j,k,width)].r = p[i][CONV(j,k,width)].r ; 
                    new[CONV(j,k,width)].g = p[i][CONV(j,k,width)].g ; 
                    new[CONV(j,k,width)].b = p[i][CONV(j,k,width)].b ; 
                }
            }

            /* Apply blur on the bottom part of the image (10%) */
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

/*
 * Main entry point
 */
int 
main( int argc, char ** argv )
{
    /* initialize MPI*/
    MPI_Init(&argc, &argv); 
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // Get process rank
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Get total number of processes

    char * input_filename ; 
    char * output_filename ;
    animated_gif * image ;
    struct timeval t1, t2;
    double duration ;

    /*Load the image*/
    if (rank == 0) {
        /* Check command-line arguments */
        if ( argc < 3 )
        {
            fprintf( stderr, "Usage: %s input.gif output.gif \n", argv[0] ) ;
            return 1 ;
        }

        input_filename = argv[1] ;
        output_filename = argv[2] ;

        /* IMPORT Timer start */
        gettimeofday(&t1, NULL);

        /* Load file and store the pixels in array */
        image = load_pixels( input_filename ) ;
        if ( image == NULL ) { return 1 ; }

        /* IMPORT Timer stop */
        gettimeofday(&t2, NULL);

        duration = (t2.tv_sec -t1.tv_sec)+((t2.tv_usec-t1.tv_usec)/1e6);

        printf( "GIF loaded from file %s with %d image(s) in %lf s\n", 
                input_filename, image->n_images, duration ) ;
    }


    if (rank != 0)
    {
        image = /*new _image */
        image->width = malloc(n_images * sizeof(int));
        image->height = malloc(n_images * sizeof(int));
        image->p = malloc(n_images * sizeof(pixel *));
    }

    MPI_Bcast(&n_images->n_images, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(image->width, n_images, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(image->height, n_images, MPI_INT, 0, MPI_COMM_WORLD);


    int chunk_size = n_images / size;
    int remainder = n_images % size;
    int start = rank * chunk_size + (rank < remainder ? rank : remainder);
    int end = start + chunk_size + (rank < remainder ? 1 : 0);

    for (int i = start; i < end; i++)
    {
        int size = image->width[i] * image->height[i];
        if (rank != 0)
        {
            image->p[i] = malloc(size * sizeof(pixel));
        }
        // change with scatterv
        MPI_Bcast(image->p[i], size * sizeof(pixel), MPI_BYTE, 0, MPI_COMM_WORLD);
    }

    if (rank ==0) {
        /* FILTER Timer start */
        gettimeofday(&t1, NULL);
    }

    /* Convert the pixels into grayscale */
    apply_gray_filter(image, start, end) ;

    /* Apply blur filter with convergence value */
    apply_blur_filter( image, 5, 20, start, end) ;

    /* Apply sobel filter on pixels */
    apply_sobel_filter( image, start, end) ;


    // 

    if (rank == 0)
    {
        for (int i = 1; i < size; i++)
        {
            int proc_start = i * chunk_size + (i < remainder ? i : remainder);
            int proc_end = proc_start + chunk_size + (i < remainder ? 1 : 0);

            for (int j = proc_start; j < proc_end; j++)
            {
                int size = image->width[j] * image->height[j];
                MPI_Recv(image->p[j], size * sizeof(pixel), MPI_BYTE, i, j, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    }
    else
    {
        for (int i = start; i < end; i++)
        {
            int size = image->width[i] * image->height[i];
            MPI_Send(image->p[i], size * sizeof(pixel), MPI_BYTE, 0, i, MPI_COMM_WORLD);
        }
    }



    if (rank ==0) {
        /* FILTER Timer stop */
        gettimeofday(&t2, NULL);

        duration = (t2.tv_sec -t1.tv_sec)+((t2.tv_usec-t1.tv_usec)/1e6);

        printf( "SOBEL done in %lf s\n", duration ) ;
    }

    if(rank == 0){
        /* EXPORT Timer start */
        gettimeofday(&t1, NULL);

        /* Store file from array of pixels to GIF file */
        if ( !store_pixels( output_filename, image ) ) { return 1 ; }

        /* EXPORT Timer stop */
        gettimeofday(&t2, NULL);

        duration = (t2.tv_sec -t1.tv_sec)+((t2.tv_usec-t1.tv_usec)/1e6);

        printf( "Export done in %lf s in file %s\n", duration, output_filename ) ;
    }
    /* Finalization of MPI */
    MPI_Finalize(); 

    return 0 ;
}
