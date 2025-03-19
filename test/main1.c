
// /*
//  * INF560
//  *
//  * Image Filtering Project
//  */
// #include <stdio.h>
// #include <stdlib.h>
// #include <string.h>
// #include <math.h>
// #include <sys/time.h>
// #include "gif_lib.h"
// #include <mpi.h> 

// /* Set this macro to 1 to enable debugging information */
// #define SOBELF_DEBUG 0

// /* Represent one pixel from the image */
// typedef struct pixel {
//     int r; /* Red */
//     int g; /* Green */
//     int b; /* Blue */
// } pixel;

// /* Represent one GIF image (animated or not) */
// typedef struct animated_gif {
//     int n_images;    /* Number of images */
//     int * width;     /* Width of each image */
//     int * height;    /* Height of each image */
//     pixel ** p;      /* Pixels of each image */
//     GifFileType * g; /* Internal representation. DO NOT MODIFY */
// } animated_gif;

// animated_gif *
// load_pixels( char * filename ) 
// {
//     GifFileType * g;
//     ColorMapObject * colmap;
//     int error;
//     int n_images;
//     int * width;
//     int * height;
//     pixel ** p;
//     int i;
//     animated_gif * image;

//     /* Open the GIF image (read mode) */
//     g = DGifOpenFileName( filename, &error );
//     if ( g == NULL ) {
//         fprintf( stderr, "Error DGifOpenFileName %s\n", filename );
//         return NULL;
//     }

//     /* Read the GIF image */
//     error = DGifSlurp( g );
//     if ( error != GIF_OK ) {
//         fprintf( stderr, "Error DGifSlurp: %d <%s>\n", error, GifErrorString(g->Error) );
//         return NULL;
//     }

//     /* Grab the number of images and the size of each image */
//     n_images = g->ImageCount;

//     width = (int *)malloc( n_images * sizeof( int ) );
//     if ( width == NULL ) {
//         fprintf( stderr, "Unable to allocate width of size %d\n", n_images );
//         return 0;
//     }

//     height = (int *)malloc( n_images * sizeof( int ) );
//     if ( height == NULL ) {
//         fprintf( stderr, "Unable to allocate height of size %d\n", n_images );
//         return 0;
//     }

//     /* Fill the width and height */
//     for ( i = 0 ; i < n_images ; i++ ) {
//         width[i] = g->SavedImages[i].ImageDesc.Width;
//         height[i] = g->SavedImages[i].ImageDesc.Height;
// #if SOBELF_DEBUG
//         printf( "Image %d: l:%d t:%d w:%d h:%d interlace:%d localCM:%p\n",
//                 i, g->SavedImages[i].ImageDesc.Left,
//                 g->SavedImages[i].ImageDesc.Top,
//                 g->SavedImages[i].ImageDesc.Width,
//                 g->SavedImages[i].ImageDesc.Height,
//                 g->SavedImages[i].ImageDesc.Interlace,
//                 g->SavedImages[i].ImageDesc.ColorMap );
// #endif
//     }

//     /* Get the global colormap */
//     colmap = g->SColorMap;
//     if ( colmap == NULL ) {
//         fprintf( stderr, "Error global colormap is NULL\n" );
//         return NULL;
//     }
// #if SOBELF_DEBUG
//     printf( "Global color map: count:%d bpp:%d sort:%d\n",
//             g->SColorMap->ColorCount,
//             g->SColorMap->BitsPerPixel,
//             g->SColorMap->SortFlag );
// #endif

//     /* Allocate the array of pixels to be returned */
//     p = (pixel **)malloc( n_images * sizeof( pixel * ) );
//     if ( p == NULL ) {
//         fprintf( stderr, "Unable to allocate array of %d images\n", n_images );
//         return NULL;
//     }

//     for ( i = 0 ; i < n_images ; i++ ) {
//         p[i] = (pixel *)malloc( width[i] * height[i] * sizeof( pixel ) );
//         if ( p[i] == NULL ) {
//             fprintf( stderr, "Unable to allocate %d-th array of %d pixels\n",
//                      i, width[i] * height[i] );
//             return NULL;
//         }
//     }
    
//     /* Fill pixels */
//     for ( i = 0 ; i < n_images ; i++ ) {
//         int j;
//         if ( g->SavedImages[i].ImageDesc.ColorMap ) {
//             /* TODO: No support for local color map */
//             fprintf( stderr, "Error: application does not support local colormap\n" );
//             return NULL;
//             /* colmap = g->SavedImages[i].ImageDesc.ColorMap; */
//         }
//         for ( j = 0 ; j < width[i] * height[i] ; j++ ) {
//             int c;
//             c = g->SavedImages[i].RasterBits[j];
//             p[i][j].r = colmap->Colors[c].Red;
//             p[i][j].g = colmap->Colors[c].Green;
//             p[i][j].b = colmap->Colors[c].Blue;
//         }
//     }

//     /* Allocate image info */
//     image = (animated_gif *)malloc( sizeof(animated_gif) );
//     if ( image == NULL ) {
//         fprintf( stderr, "Unable to allocate memory for animated_gif\n" );
//         return NULL;
//     }

//     /* Fill image fields */
//     image->n_images = n_images;
//     image->width = width;
//     image->height = height;
//     image->p = p;
//     image->g = g;
// #if SOBELF_DEBUG
//     printf( "-> GIF w/ %d image(s) with first image of size %d x %d\n",
//             image->n_images, image->width[0], image->height[0] );
// #endif

//     return image;
// }

// int 
// output_modified_read_gif( char * filename, GifFileType * g ) 
// {
//     GifFileType * g2;
//     int error2;
// #if SOBELF_DEBUG
//     printf( "Starting output to file %s\n", filename );
// #endif
//     g2 = EGifOpenFileName( filename, false, &error2 );
//     if ( g2 == NULL ) {
//         fprintf( stderr, "Error EGifOpenFileName %s\n", filename );
//         return 0;
//     }
//     g2->SWidth = g->SWidth;
//     g2->SHeight = g->SHeight;
//     g2->SColorResolution = g->SColorResolution;
//     g2->SBackGroundColor = g->SBackGroundColor;
//     g2->AspectByte = g->AspectByte;
//     g2->SColorMap = g->SColorMap;
//     g2->ImageCount = g->ImageCount;
//     g2->SavedImages = g->SavedImages;
//     g2->ExtensionBlockCount = g->ExtensionBlockCount;
//     g2->ExtensionBlocks = g->ExtensionBlocks;
//     error2 = EGifSpew( g2 );
//     if ( error2 != GIF_OK ) {
//         fprintf( stderr, "Error after writing g2: %d <%s>\n", error2, GifErrorString(g2->Error) );
//         return 0;
//     }
//     return 1;
// }

// int
// store_pixels( char * filename, animated_gif * image )
// {
//     int n_colors = 0;
//     pixel ** p;
//     int i, j, k;
//     GifColorType * colormap;
//     colormap = (GifColorType *)malloc( 256 * sizeof( GifColorType ) );
//     if ( colormap == NULL ) {
//         fprintf( stderr, "Unable to allocate 256 colors\n" );
//         return 0;
//     }
//     for ( i = 0 ; i < 256 ; i++ ) {
//         colormap[i].Red = 255;
//         colormap[i].Green = 255;
//         colormap[i].Blue = 255;
//     }
//     int moy;
//     moy = ( image->g->SColorMap->Colors[ image->g->SBackGroundColor ].Red +
//             image->g->SColorMap->Colors[ image->g->SBackGroundColor ].Green +
//             image->g->SColorMap->Colors[ image->g->SBackGroundColor ].Blue )/3;
//     if ( moy < 0 ) moy = 0;
//     if ( moy > 255 ) moy = 255;
// #if SOBELF_DEBUG
//     printf( "[DEBUG] Background color (%d,%d,%d) -> (%d,%d,%d)\n",
//             image->g->SColorMap->Colors[ image->g->SBackGroundColor ].Red,
//             image->g->SColorMap->Colors[ image->g->SBackGroundColor ].Green,
//             image->g->SColorMap->Colors[ image->g->SBackGroundColor ].Blue,
//             moy, moy, moy );
// #endif
//     colormap[0].Red = moy;
//     colormap[0].Green = moy;
//     colormap[0].Blue = moy;
//     image->g->SBackGroundColor = 0;
//     n_colors++;
//     /* Process extension blocks in main structure */
//     for ( j = 0 ; j < image->g->ExtensionBlockCount ; j++ ) {
//         int f;
//         f = image->g->ExtensionBlocks[j].Function;
//         if ( f == GRAPHICS_EXT_FUNC_CODE ) {
//             int tr_color = image->g->ExtensionBlocks[j].Bytes[3];
//             if ( tr_color >= 0 && tr_color < 255 ) {
//                 int found = -1;
//                 moy = ( image->g->SColorMap->Colors[ tr_color ].Red +
//                         image->g->SColorMap->Colors[ tr_color ].Green +
//                         image->g->SColorMap->Colors[ tr_color ].Blue ) / 3;
//                 if ( moy < 0 ) moy = 0;
//                 if ( moy > 255 ) moy = 255;
// #if SOBELF_DEBUG
//                 printf( "[DEBUG] Transparency color image %d (%d,%d,%d) -> (%d,%d,%d)\n",
//                         i,
//                         image->g->SColorMap->Colors[ tr_color ].Red,
//                         image->g->SColorMap->Colors[ tr_color ].Green,
//                         image->g->SColorMap->Colors[ tr_color ].Blue,
//                         moy, moy, moy );
// #endif
//                 for ( k = 0 ; k < n_colors ; k++ ) {
//                     if ( moy == colormap[k].Red &&
//                          moy == colormap[k].Green &&
//                          moy == colormap[k].Blue )
//                     {
//                         found = k;
//                     }
//                 }
//                 if ( found == -1  ) {
//                     if ( n_colors >= 256 ) {
//                         fprintf( stderr, "Error: Found too many colors inside the image\n" );
//                         return 0;
//                     }
// #if SOBELF_DEBUG
//                     printf( "[DEBUG]\tNew color %d\n", n_colors );
// #endif
//                     colormap[n_colors].Red = moy;
//                     colormap[n_colors].Green = moy;
//                     colormap[n_colors].Blue = moy;
//                     image->g->ExtensionBlocks[j].Bytes[3] = n_colors;
//                     n_colors++;
//                 } else {
// #if SOBELF_DEBUG
//                     printf( "[DEBUG]\tFound existing color %d\n", found );
// #endif
//                     image->g->ExtensionBlocks[j].Bytes[3] = found;
//                 }
//             }
//         }
//     }
//     for ( i = 0 ; i < image->n_images ; i++ ) {
//         for ( j = 0 ; j < image->g->SavedImages[i].ExtensionBlockCount ; j++ ) {
//             int f;
//             f = image->g->SavedImages[i].ExtensionBlocks[j].Function;
//             if ( f == GRAPHICS_EXT_FUNC_CODE ) {
//                 int tr_color = image->g->SavedImages[i].ExtensionBlocks[j].Bytes[3];
//                 if ( tr_color >= 0 && tr_color < 255 ) {
//                     int found = -1;
//                     moy = ( image->g->SColorMap->Colors[ tr_color ].Red +
//                             image->g->SColorMap->Colors[ tr_color ].Green +
//                             image->g->SColorMap->Colors[ tr_color ].Blue ) / 3;
//                     if ( moy < 0 ) moy = 0;
//                     if ( moy > 255 ) moy = 255;
// #if SOBELF_DEBUG
//                     printf( "[DEBUG] Transparency color image %d (%d,%d,%d) -> (%d,%d,%d)\n",
//                             i,
//                             image->g->SColorMap->Colors[ tr_color ].Red,
//                             image->g->SColorMap->Colors[ tr_color ].Green,
//                             image->g->SColorMap->Colors[ tr_color ].Blue,
//                             moy, moy, moy );
// #endif
//                     for ( k = 0 ; k < n_colors ; k++ ) {
//                         if ( moy == colormap[k].Red &&
//                              moy == colormap[k].Green &&
//                              moy == colormap[k].Blue )
//                         {
//                             found = k;
//                         }
//                     }
//                     if ( found == -1  ) {
//                         if ( n_colors >= 256 ) {
//                             fprintf( stderr, "Error: Found too many colors inside the image\n" );
//                             return 0;
//                         }
// #if SOBELF_DEBUG
//                         printf( "[DEBUG]\tNew color %d\n", n_colors );
// #endif
//                         colormap[n_colors].Red = moy;
//                         colormap[n_colors].Green = moy;
//                         colormap[n_colors].Blue = moy;
//                         image->g->SavedImages[i].ExtensionBlocks[j].Bytes[3] = n_colors;
//                         n_colors++;
//                     } else {
// #if SOBELF_DEBUG
//                         printf( "[DEBUG]\tFound existing color %d\n", found );
// #endif
//                         image->g->SavedImages[i].ExtensionBlocks[j].Bytes[3] = found;
//                     }
//                 }
//             }
//         }
//     }
// #if SOBELF_DEBUG
//     printf( "[DEBUG] Number of colors after background and transparency: %d\n", n_colors );
// #endif
//     p = image->p;
//     for ( i = 0 ; i < image->n_images ; i++ ) {
// #if SOBELF_DEBUG
//         printf( "OUTPUT: Processing image %d (total of %d images) -> %d x %d\n",
//                 i, image->n_images, image->width[i], image->height[i] );
// #endif
//         for ( j = 0 ; j < image->width[i] * image->height[i] ; j++ ) {
//             int found = 0;
//             for ( k = 0 ; k < n_colors ; k++ ) {
//                 if ( p[i][j].r == colormap[k].Red &&
//                      p[i][j].g == colormap[k].Green &&
//                      p[i][j].b == colormap[k].Blue )
//                 {
//                     found = 1;
//                 }
//             }
//             if ( found == 0 ) {
//                 if ( n_colors >= 256 ) {
//                     fprintf( stderr, "Error: Found too many colors inside the image\n" );
//                     return 0;
//                 }
// #if SOBELF_DEBUG
//                 printf( "[DEBUG] Found new %d color (%d,%d,%d)\n",
//                         n_colors, p[i][j].r, p[i][j].g, p[i][j].b );
// #endif
//                 colormap[n_colors].Red = p[i][j].r;
//                 colormap[n_colors].Green = p[i][j].g;
//                 colormap[n_colors].Blue = p[i][j].b;
//                 n_colors++;
//             }
//         }
//     }
// #if SOBELF_DEBUG
//     printf( "OUTPUT: found %d color(s)\n", n_colors );
// #endif
//     if ( n_colors != (1 << GifBitSize(n_colors) ) ) {
//         n_colors = (1 << GifBitSize(n_colors) );
//     }
// #if SOBELF_DEBUG
//     printf( "OUTPUT: Rounding up to %d color(s)\n", n_colors );
// #endif
//     ColorMapObject * cmo;
//     cmo = GifMakeMapObject( n_colors, colormap );
//     if ( cmo == NULL ) {
//         fprintf( stderr, "Error while creating a ColorMapObject w/ %d color(s)\n", n_colors );
//         return 0;
//     }
//     image->g->SColorMap = cmo;
//     for ( i = 0 ; i < image->n_images ; i++ ) {
//         for ( j = 0 ; j < image->width[i] * image->height[i] ; j++ ) {
//             int found_index = -1;
//             for ( k = 0 ; k < n_colors ; k++ ) {
//                 if ( p[i][j].r == image->g->SColorMap->Colors[k].Red &&
//                      p[i][j].g == image->g->SColorMap->Colors[k].Green &&
//                      p[i][j].b == image->g->SColorMap->Colors[k].Blue )
//                 {
//                     found_index = k;
//                 }
//             }
//             if ( found_index == -1 ) {
//                 fprintf( stderr, "Error: Unable to find a pixel in the color map\n" );
//                 return 0;
//             }
//             image->g->SavedImages[i].RasterBits[j] = found_index;
//         }
//     }
//     if ( !output_modified_read_gif( filename, image->g ) ) { return 0; }
//     return 1;
// }

// void
// apply_gray_filter( animated_gif * image )
// {
//     int i, j;
//     pixel ** p;
//     p = image->p;
//     for ( i = 0 ; i < image->n_images ; i++ ) {
//         for ( j = 0 ; j < image->width[i] * image->height[i] ; j++ ) {
//             int moy;
//             moy = (p[i][j].r + p[i][j].g + p[i][j].b) / 3;
//             if ( moy < 0 ) moy = 0;
//             if ( moy > 255 ) moy = 255;
//             p[i][j].r = moy;
//             p[i][j].g = moy;
//             p[i][j].b = moy;
//         }
//     }
// }

// #define CONV(l,c,nb_c) ((l)*(nb_c)+(c))

// void apply_gray_line( animated_gif * image ) 
// {
//     int i, j, k;
//     pixel ** p;
//     p = image->p;
//     for ( i = 0 ; i < image->n_images ; i++ ) {
//         for ( j = 0 ; j < 10 ; j++ ) {
//             for ( k = image->width[i]/2 ; k < image->width[i] ; k++ ) {
//                 p[i][CONV(j,k,image->width[i])].r = 0;
//                 p[i][CONV(j,k,image->width[i])].g = 0;
//                 p[i][CONV(j,k,image->width[i])].b = 0;
//             }
//         }
//     }
// }

// void
// apply_blur_filter( animated_gif * image, int size, int threshold )
// {
//     int i, j, k;
//     int width, height;
//     int end = 0;
//     int n_iter = 0;
//     pixel ** p;
//     pixel * new;
//     p = image->p;
//     for ( i = 0 ; i < image->n_images ; i++ ) {
//         n_iter = 0;
//         width = image->width[i];
//         height = image->height[i];
//         new = (pixel *)malloc(width * height * sizeof( pixel ));
//         do {
//             end = 1;
//             n_iter++;
//             for(j = 0; j < height-1; j++) {
//                 for(k = 0; k < width-1; k++) {
//                     new[CONV(j,k,width)].r = p[i][CONV(j,k,width)].r;
//                     new[CONV(j,k,width)].g = p[i][CONV(j,k,width)].g;
//                     new[CONV(j,k,width)].b = p[i][CONV(j,k,width)].b;
//                 }
//             }
//             for(j = size; j < height/10 - size; j++) {
//                 for(k = size; k < width - size; k++) {
//                     int stencil_j, stencil_k;
//                     int t_r = 0, t_g = 0, t_b = 0;
//                     for ( stencil_j = -size; stencil_j <= size; stencil_j++ ) {
//                         for ( stencil_k = -size; stencil_k <= size; stencil_k++ ) {
//                             t_r += p[i][CONV(j+stencil_j,k+stencil_k,width)].r;
//                             t_g += p[i][CONV(j+stencil_j,k+stencil_k,width)].g;
//                             t_b += p[i][CONV(j+stencil_j,k+stencil_k,width)].b;
//                         }
//                     }
//                     new[CONV(j,k,width)].r = t_r / ((2*size+1)*(2*size+1));
//                     new[CONV(j,k,width)].g = t_g / ((2*size+1)*(2*size+1));
//                     new[CONV(j,k,width)].b = t_b / ((2*size+1)*(2*size+1));
//                 }
//             }
//             for(j = height/10 - size; j < height*0.9 + size; j++) {
//                 for(k = size; k < width - size; k++) {
//                     new[CONV(j,k,width)].r = p[i][CONV(j,k,width)].r; 
//                     new[CONV(j,k,width)].g = p[i][CONV(j,k,width)].g; 
//                     new[CONV(j,k,width)].b = p[i][CONV(j,k,width)].b; 
//                 }
//             }
//             for(j = height*0.9 + size; j < height - size; j++) {
//                 for(k = size; k < width - size; k++) {
//                     int stencil_j, stencil_k;
//                     int t_r = 0, t_g = 0, t_b = 0;
//                     for ( stencil_j = -size; stencil_j <= size; stencil_j++ ) {
//                         for ( stencil_k = -size; stencil_k <= size; stencil_k++ ) {
//                             t_r += p[i][CONV(j+stencil_j,k+stencil_k,width)].r;
//                             t_g += p[i][CONV(j+stencil_j,k+stencil_k,width)].g;
//                             t_b += p[i][CONV(j+stencil_j,k+stencil_k,width)].b;
//                         }
//                     }
//                     new[CONV(j,k,width)].r = t_r / ((2*size+1)*(2*size+1));
//                     new[CONV(j,k,width)].g = t_g / ((2*size+1)*(2*size+1));
//                     new[CONV(j,k,width)].b = t_b / ((2*size+1)*(2*size+1));
//                 }
//             }
//             for(j = 1; j < height-1; j++) {
//                 for(k = 1; k < width-1; k++) {
//                     float diff_r = new[CONV(j,k,width)].r - p[i][CONV(j,k,width)].r;
//                     float diff_g = new[CONV(j,k,width)].g - p[i][CONV(j,k,width)].g;
//                     float diff_b = new[CONV(j,k,width)].b - p[i][CONV(j,k,width)].b;
//                     if ( diff_r > threshold || -diff_r > threshold ||
//                          diff_g > threshold || -diff_g > threshold ||
//                          diff_b > threshold || -diff_b > threshold ) {
//                         end = 0;
//                     }
//                     p[i][CONV(j,k,width)].r = new[CONV(j,k,width)].r;
//                     p[i][CONV(j,k,width)].g = new[CONV(j,k,width)].g;
//                     p[i][CONV(j,k,width)].b = new[CONV(j,k,width)].b;
//                 }
//             }
//         } while ( threshold > 0 && !end );
// #if SOBELF_DEBUG
//         printf( "BLUR: number of iterations for image %d\n", n_iter );
// #endif
//         free(new);
//     }
// }

// void
// apply_sobel_filter( animated_gif * image )
// {
//     int i, j, k;
//     int width, height;
//     pixel ** p;
//     p = image->p;
//     for ( i = 0 ; i < image->n_images ; i++ ) {
//         width = image->width[i];
//         height = image->height[i];
//         pixel * sobel;
//         sobel = (pixel *)malloc(width * height * sizeof( pixel ) );
//         for(j = 1; j < height - 1; j++) {
//             for(k = 1; k < width - 1; k++) {
//                 int pixel_blue_no, pixel_blue_n, pixel_blue_ne;
//                 int pixel_blue_so, pixel_blue_s, pixel_blue_se;
//                 int pixel_blue_o, pixel_blue, pixel_blue_e;
//                 float deltaX_blue, deltaY_blue, val_blue;
//                 pixel_blue_no = p[i][CONV(j-1,k-1,width)].b;
//                 pixel_blue_n  = p[i][CONV(j-1,k  ,width)].b;
//                 pixel_blue_ne = p[i][CONV(j-1,k+1,width)].b;
//                 pixel_blue_so = p[i][CONV(j+1,k-1,width)].b;
//                 pixel_blue_s  = p[i][CONV(j+1,k  ,width)].b;
//                 pixel_blue_se = p[i][CONV(j+1,k+1,width)].b;
//                 pixel_blue_o  = p[i][CONV(j,k-1,width)].b;
//                 pixel_blue    = p[i][CONV(j,k,width)].b;
//                 pixel_blue_e  = p[i][CONV(j,k+1,width)].b;
//                 deltaX_blue = -pixel_blue_no + pixel_blue_ne - 2*pixel_blue_o + 2*pixel_blue_e - pixel_blue_so + pixel_blue_se;             
//                 deltaY_blue = pixel_blue_se + 2*pixel_blue_s + pixel_blue_so - pixel_blue_ne - 2*pixel_blue_n - pixel_blue_no;
//                 val_blue = sqrt(deltaX_blue * deltaX_blue + deltaY_blue * deltaY_blue) / 4;
//                 if ( val_blue > 50 ) {
//                     sobel[CONV(j,k,width)].r = 255;
//                     sobel[CONV(j,k,width)].g = 255;
//                     sobel[CONV(j,k,width)].b = 255;
//                 } else {
//                     sobel[CONV(j,k,width)].r = 0;
//                     sobel[CONV(j,k,width)].g = 0;
//                     sobel[CONV(j,k,width)].b = 0;
//                 }
//             }
//         }
//         for(j = 1; j < height - 1; j++) {
//             for(k = 1; k < width - 1; k++) {
//                 p[i][CONV(j,k,width)].r = sobel[CONV(j,k,width)].r;
//                 p[i][CONV(j,k,width)].g = sobel[CONV(j,k,width)].g;
//                 p[i][CONV(j,k,width)].b = sobel[CONV(j,k,width)].b;
//             }
//         }
//         free(sobel);
//     }
// }

// /*
//  * Main entry point
//  */
// int 
// main( int argc, char ** argv )
// {
//     /* initialize MPI */
//     MPI_Init(&argc, &argv); 
//     int *sendcounts_bytes, *displs_bytes, *sendcounts_images, *displs_images;
//     int rank, size, root = 0;
//     int n_images_chunk, n_bytes_chunk;
//     pixel *rcv_pixels;
//     int *width, *height;
//     animated_gif *image, *local_images;
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//     MPI_Comm_size(MPI_COMM_WORLD, &size);
//     char * input_filename; 
//     char * output_filename;
//     struct timeval t1, t2;
//     double duration;
//     pixel *all_pixels = NULL;   
//     pixel *all_pixels_recv = NULL; 

//     /* Load the image on rank 0 */
//     if (rank == 0) {
//         if ( argc < 3 ) {
//             fprintf( stderr, "Usage: %s input.gif output.gif \n", argv[0] );
//             return 1;
//         }
//         input_filename = argv[1];
//         output_filename = argv[2];
//         gettimeofday(&t1, NULL);
//         image = load_pixels( input_filename );
//         if ( image == NULL ) { return 1; }
//         gettimeofday(&t2, NULL);
//         duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec - t1.tv_usec)/1e6);
//         printf( "GIF loaded from file %s with %d image(s) in %lf s\n", 
//                 input_filename, image->n_images, duration );

//         int n_images = image->n_images;
//         int chunk_size = n_images / size;
//         int remainder = n_images % size;
//         sendcounts_images = (int*)malloc(size * sizeof(int));
//         sendcounts_bytes = (int*)malloc(size * sizeof(int));
//         displs_bytes = (int*)malloc((size+1) * sizeof(int)); displs_bytes[0] = 0;
//         displs_images = (int*)malloc((size+1) * sizeof(int)); displs_images[0] = 0;
//         for (int i = 0; i < size; i++) {
//             int start = i * chunk_size + (i < remainder ? i : remainder);
//             int end = start + chunk_size + (i < remainder ? 1 : 0);
//             sendcounts_images[i] = end - start;
//             sendcounts_bytes[i] = 0;
//             for (int j = start; j < end; j++) {
//                 sendcounts_bytes[i] += image->width[j] * image->height[j] * sizeof(pixel);
//             }
//             displs_bytes[i+1] = displs_bytes[i] + sendcounts_bytes[i];
//             displs_images[i+1] = displs_images[i] + sendcounts_images[i];
//         }
//         MPI_Scatter(sendcounts_images, 1, MPI_INT,
//                     &n_images_chunk, 1, MPI_INT,
//                     root, MPI_COMM_WORLD);
//         MPI_Scatter(sendcounts_bytes, 1, MPI_INT,
//                     &n_bytes_chunk, 1, MPI_INT,
//                     root, MPI_COMM_WORLD);

//         int total_bytes = displs_bytes[size];
//         all_pixels = (pixel *)malloc(total_bytes);
//         if (all_pixels == NULL) {
//             fprintf(stderr, "Unable to allocate contiguous send buffer\n");
//             exit(1);
//         }
//         int offset = 0;
//         for (int i = 0; i < image->n_images; i++) {
//             int img_bytes = image->width[i] * image->height[i] * sizeof(pixel);
//             memcpy(all_pixels + offset, image->p[i], img_bytes);
//             offset += image->width[i] * image->height[i];
//         }
//         rcv_pixels = (pixel*)malloc(n_bytes_chunk);
//         height = (int*)malloc(n_images_chunk * sizeof(int));
//         width = (int*)malloc(n_images_chunk * sizeof(int));
//         MPI_Scatterv(all_pixels, sendcounts_bytes, displs_bytes, MPI_BYTE,
//                      rcv_pixels, n_bytes_chunk, MPI_BYTE,
//                      root, MPI_COMM_WORLD);
//         MPI_Scatterv(image->height, sendcounts_images, displs_images, MPI_INT,
//                      height, n_images_chunk, MPI_INT,
//                      root, MPI_COMM_WORLD);
//         MPI_Scatterv(image->width, sendcounts_images, displs_images, MPI_INT,
//                      width, n_images_chunk, MPI_INT,
//                      root, MPI_COMM_WORLD);
//         free(all_pixels); 
//     } else {
//         MPI_Scatter(sendcounts_images, 1, MPI_INT,
//                     &n_images_chunk, 1, MPI_INT,
//                     root, MPI_COMM_WORLD);
//         MPI_Scatter(sendcounts_bytes, 1, MPI_INT,
//                     &n_bytes_chunk, 1, MPI_INT,
//                     root, MPI_COMM_WORLD);
//         rcv_pixels = (pixel*)malloc(n_bytes_chunk);
//         height = (int*)malloc(n_images_chunk * sizeof(int));
//         width = (int*)malloc(n_images_chunk * sizeof(int));
//         MPI_Scatterv(NULL, NULL, NULL, MPI_BYTE,
//                      rcv_pixels, n_bytes_chunk, MPI_BYTE,
//                      root, MPI_COMM_WORLD);
//         MPI_Scatterv(NULL, NULL, NULL, MPI_INT,
//                      height, n_images_chunk, MPI_INT,
//                      root, MPI_COMM_WORLD);
//         MPI_Scatterv(NULL, NULL, NULL, MPI_INT,
//                      width, n_images_chunk, MPI_INT,
//                      root, MPI_COMM_WORLD);
//     }

//     local_images = (animated_gif*) malloc(sizeof(animated_gif));
//     local_images->n_images = n_images_chunk;
//     local_images->width = width;
//     local_images->height = height;
//     local_images->p = (pixel **)malloc(n_images_chunk * sizeof(pixel *));
//     int pos = 0;
//     for (int i = 0; i < n_images_chunk; i++) {
//         local_images->p[i] = rcv_pixels + pos;
//         pos += width[i] * height[i];
//     }
//     if (rank == 0) {
//         gettimeofday(&t1, NULL);
//     }

//     apply_gray_filter(local_images);
//     apply_blur_filter(local_images, 5, 20);
//     apply_sobel_filter(local_images);

//     if (rank == 0) {
//         int total_bytes;

//         total_bytes = displs_bytes[size];
//         all_pixels_recv = (pixel *)malloc(total_bytes);
//         if (all_pixels_recv == NULL) {
//             fprintf(stderr, "Unable to allocate contiguous recv buffer\n");
//             exit(1);
//         }
//         MPI_Gatherv(rcv_pixels, n_bytes_chunk, MPI_BYTE,
//                     all_pixels_recv, sendcounts_bytes, displs_bytes, MPI_BYTE,
//                     root, MPI_COMM_WORLD);

//         int offset = 0;
//         for (int i = 0; i < image->n_images; i++) {
//             int img_pixels = image->width[i] * image->height[i];
//             memcpy(image->p[i], all_pixels_recv + offset, img_pixels * sizeof(pixel));
//             offset += img_pixels;
//         }
//         free(all_pixels_recv);
//     } else {
//         MPI_Gatherv(rcv_pixels, n_bytes_chunk, MPI_BYTE,
//                     NULL, NULL, NULL, MPI_BYTE,
//                     root, MPI_COMM_WORLD);
//     }
//     if (rank == 0) {
//         gettimeofday(&t2, NULL);
//         duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec - t1.tv_usec)/1e6);
//         printf( "SOBEL done in %lf s\n", duration );
//     }
//     if (rank == 0) {
//         gettimeofday(&t1, NULL);
//         if ( !store_pixels( output_filename, image ) ) { return 1; }
//         gettimeofday(&t2, NULL);
//         duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec - t1.tv_usec)/1e6);
//         printf( "Export done in %lf s in file %s\n", duration, output_filename );
//     }
//     MPI_Finalize();
//     return 0;
// }


 #include <stdio.h>
 #include <stdlib.h>
 #include <string.h>
 #include <math.h>
 #include <sys/time.h>
 #include "gif_lib.h"
 #include <mpi.h>
 #include <omp.h>
 #define SOBELF_DEBUG 0
 #define MAX_FILENAME 256
 
 /* 任务结构：存储一对输入/输出文件名 */
 typedef struct {
     char input[MAX_FILENAME];
     char output[MAX_FILENAME];
 } task_t;
 
 /* 当 task.input[0]=='\0' 时，表示终止任务 */
 
 /* Represent one pixel */
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
 
 /* ------------------ 以下为处理单个 GIF 文件的函数 ------------------ */
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
    for ( i = 0 ; i < n_images ; i++ ) {
        int j;
        if ( g->SavedImages[i].ImageDesc.ColorMap ) {
            /* TODO: No support for local color map */
            fprintf( stderr, "Error: application does not support local colormap\n" );
            return NULL;
            /* colmap = g->SavedImages[i].ImageDesc.ColorMap; */
        }
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

    #pragma omp parallel for private(j)
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

    #pragma omp parallel for private(j, k)
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

    #pragma omp parallel for schedule(dynamic)
    for (i = 0; i < image->n_images; i++) {
        int width = image->width[i];
        int height = image->height[i];
        pixel *new = (pixel *)malloc(width * height * sizeof(pixel));
        int n_iter = 0, end;
        do {
            end = 1;
            n_iter++;
            
            #pragma omp parallel for collapse(2)
            for (int j = 1; j < height - 1; j++) {
                for (int k = 1; k < width - 1; k++) {
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
                    new[idx].r = t_r / avg_factor;
                    new[idx].g = t_g / avg_factor;
                    new[idx].b = t_b / avg_factor;
                }
            }
            
            #pragma omp parallel for collapse(2) reduction(&:end)
            for (int j = 1; j < height - 1; j++) {
                for (int k = 1; k < width - 1; k++) {
                    int idx = CONV(j, k, width);
                    if (abs(new[idx].r - p[i][idx].r) > threshold ||
                        abs(new[idx].g - p[i][idx].g) > threshold ||
                        abs(new[idx].b - p[i][idx].b) > threshold) {
                        end = 0;
                    }
                    p[i][idx] = new[idx];
                }
            }
        } while (threshold > 0 && !end);
        free(new);
    }
}
void apply_sobel_filter(animated_gif *image) {
    int i, j, k;
    pixel **p = image->p;

    #pragma omp parallel for private(j, k)
    for (i = 0; i < image->n_images; i++) {
        int width = image->width[i];
        int height = image->height[i];
        pixel *sobel = (pixel *)malloc(width * height * sizeof(pixel));
        
        #pragma omp parallel for collapse(2) private(j, k)
        for (j = 1; j < height - 1; j++) {
            for (k = 1; k < width - 1; k++) {
                int pixel_blue_no, pixel_blue_n, pixel_blue_ne;
                int pixel_blue_so, pixel_blue_s, pixel_blue_se;
                int pixel_blue_o, pixel_blue, pixel_blue_e;
                float deltaX_blue, deltaY_blue, val_blue;

                pixel_blue_no = p[i][CONV(j-1,k-1,width)].b;
                pixel_blue_n  = p[i][CONV(j-1,k  ,width)].b;
                pixel_blue_ne = p[i][CONV(j-1,k+1,width)].b;
                pixel_blue_so = p[i][CONV(j+1,k-1,width)].b;
                pixel_blue_s  = p[i][CONV(j+1,k  ,width)].b;
                pixel_blue_se = p[i][CONV(j+1,k+1,width)].b;
                pixel_blue_o  = p[i][CONV(j,k-1,width)].b;
                pixel_blue    = p[i][CONV(j,k,width)].b;
                pixel_blue_e  = p[i][CONV(j,k+1,width)].b;

                deltaX_blue = -pixel_blue_no + pixel_blue_ne - 2*pixel_blue_o + 2*pixel_blue_e - pixel_blue_so + pixel_blue_se;
                deltaY_blue = pixel_blue_se + 2*pixel_blue_s + pixel_blue_so - pixel_blue_ne - 2*pixel_blue_n - pixel_blue_no;
                val_blue = sqrt(deltaX_blue * deltaX_blue + deltaY_blue * deltaY_blue) / 4;

                if (val_blue > 50) {
                    sobel[CONV(j,k,width)].r = 255;
                    sobel[CONV(j,k,width)].g = 255;
                    sobel[CONV(j,k,width)].b = 255;
                } else {
                    sobel[CONV(j,k,width)].r = 0;
                    sobel[CONV(j,k,width)].g = 0;
                    sobel[CONV(j,k,width)].b = 0;
                }
            }
        }

        #pragma omp parallel for collapse(2) private(j, k)
        for (j = 1; j < height - 1; j++) {
            for (k = 1; k < width - 1; k++) {
                p[i][CONV(j,k,width)] = sobel[CONV(j,k,width)];
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

    #pragma omp parallel for
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

    #pragma omp parallel for private(j, k)
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
    #pragma omp parallel for private(j, k)
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

 /* -------------------- MPI 动态调度 (Master-Worker) 部分 -------------------- */
 
 enum { TAG_TASK = 1, TAG_DONE = 2 };

 int main( int argc, char ** argv )
 {
     MPI_Init(&argc, &argv);
     int rank, size;
     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
     MPI_Comm_size(MPI_COMM_WORLD, &size);
 
    //  /* 单任务模式：仅2个参数时，直接处理 */
    //  if (size == 1) {
    //      if (argc != 3) {
    //          fprintf(stderr, "Usage: %s input.gif output.gif\n", argv[0]);
    //          MPI_Finalize();
    //          return 1;
    //      }
    //      struct timeval tstart, tend;
    //      gettimeofday(&tstart, NULL);
    //      animated_gif * image = load_pixels(argv[1]);
    //      if (image == NULL) {
    //          MPI_Finalize();
    //          return 1;
    //      }
    //      printf("GIF loaded from file %s with %d frame(s)\n", argv[1], image->n_images);
    //      apply_gray_filter(image);
    //      apply_blur_filter(image, 5, 20);
    //      apply_sobel_filter(image);
    //      if (!store_pixels(argv[2], image)) {
    //          MPI_Finalize();
    //          return 1;
    //      }
    //      gettimeofday(&tend, NULL);
    //      double dt = (tend.tv_sec - tstart.tv_sec) + ((tend.tv_usec - tstart.tv_usec)/1e6);
    //      printf("Processed %s -> %s in %lf s\n", argv[1], argv[2], dt);
    //      MPI_Finalize();
    //      return 0;
    //  }
 
     /* 多任务动态调度模式：参数个数>=3且为奇数（程序名 + 成对文件名） */
     int ntasks = 0;
     task_t * tasks = NULL;
     if (rank == 0) {
         ntasks = (argc - 1) / 2;
         tasks = (task_t *)malloc(ntasks * sizeof(task_t));
         for (int i = 0; i < ntasks; i++) {
             strncpy(tasks[i].input, argv[1 + 2*i], MAX_FILENAME - 1);
             tasks[i].input[MAX_FILENAME - 1] = '\0';
             strncpy(tasks[i].output, argv[1 + 2*i + 1], MAX_FILENAME - 1);
             tasks[i].output[MAX_FILENAME - 1] = '\0';
         }
     }
 
     /* 在多任务动态调度中，Master负责调度，每个任务为一个文件 */
     if (rank == 0) {
         struct timeval tstart, tend;
         gettimeofday(&tstart, NULL);
         int task_index = 0;
         int tasks_done = 0;
         MPI_Status status;
         /* 初始向每个 Worker分派任务 */
         for (int dest = 1; dest < size; dest++) {
             if (task_index < ntasks) {
                 MPI_Send(&tasks[task_index], sizeof(task_t), MPI_BYTE, dest, TAG_TASK, MPI_COMM_WORLD);
                 task_index++;
             } else {
                 task_t terminator;
                 terminator.input[0] = '\0'; /* 终止信号 */
                 MPI_Send(&terminator, sizeof(task_t), MPI_BYTE, dest, TAG_TASK, MPI_COMM_WORLD);
             }
         }
         /* 循环接收完成信号并分派新任务 */
         while (tasks_done < ntasks) {
             int dummy;
             int source;
             MPI_Recv(&dummy, 1, MPI_INT, MPI_ANY_SOURCE, TAG_DONE, MPI_COMM_WORLD, &status);
             source = status.MPI_SOURCE;
             tasks_done++;
             if (task_index < ntasks) {
                 MPI_Send(&tasks[task_index], sizeof(task_t), MPI_BYTE, source, TAG_TASK, MPI_COMM_WORLD);
                 task_index++;
             } else {
                 task_t terminator;
                 terminator.input[0] = '\0';
                 MPI_Send(&terminator, sizeof(task_t), MPI_BYTE, source, TAG_TASK, MPI_COMM_WORLD);
             }
         }
         gettimeofday(&tend, NULL);
         double dt = (tend.tv_sec - tstart.tv_sec) + ((tend.tv_usec - tstart.tv_usec)/1e6);
         printf("Master: All tasks processed in %lf s\n", dt);
         free(tasks);
     }
     /* Worker部分 */
     else {
         while (1) {
             task_t task;
             MPI_Status status;
             MPI_Recv(&task, sizeof(task_t), MPI_BYTE, 0, TAG_TASK, MPI_COMM_WORLD, &status);
             if (task.input[0] == '\0') {
                 break;  /* 收到终止信号 */
             }
             struct timeval tstart, tend;
             gettimeofday(&tstart, NULL);
             animated_gif * image = load_pixels(task.input);
             if (image == NULL) {
                 fprintf(stderr, "Rank %d: Error loading %s\n", rank, task.input);
             } else {
                 printf("Rank %d: GIF loaded from file %s with %d frame(s)\n", rank, task.input, image->n_images);
                 fflush(stdout);
                 apply_gray_filter(image);
                 apply_blur_filter(image, 5, 20);
                 apply_sobel_filter(image);
                 if (!store_pixels(task.output, image)) {
                     fprintf(stderr, "Rank %d: Error storing %s\n", rank, task.output);
                 }
                 gettimeofday(&tend, NULL);
                 double dt = (tend.tv_sec - tstart.tv_sec) + ((tend.tv_usec - tstart.tv_usec)/1e6);
                 printf("Rank %d: Processed %s -> %s in %lf s\n", rank, task.input, task.output, dt);
                 fflush(stdout);
             }
             int dummy = 1;
             MPI_Send(&dummy, 1, MPI_INT, 0, TAG_DONE, MPI_COMM_WORLD);
         }
     }
 
     MPI_Finalize();
     return 0;
 }

// int main(int argc, char **argv) {
//     MPI_Init(&argc, &argv);
//     int rank, size;
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//     MPI_Comm_size(MPI_COMM_WORLD, &size);

//     int ntasks = 0;
//     task_t *tasks = NULL;

//     if (rank == 0) {
//         /* 主节点读取任务 */
//         if (argc < 3 || ((argc - 1) % 2) != 0) {
//             fprintf(stderr, "Usage: %s input1.gif output1.gif [input2.gif output2.gif ...]\n", argv[0]);
//             MPI_Abort(MPI_COMM_WORLD, 1);
//         }

//         ntasks = (argc - 1) / 2;
//         tasks = (task_t *)malloc(ntasks * sizeof(task_t));

//         for (int i = 0; i < ntasks; i++) {
//             strncpy(tasks[i].input, argv[1 + 2 * i], MAX_FILENAME - 1);
//             tasks[i].input[MAX_FILENAME - 1] = '\0';
//             strncpy(tasks[i].output, argv[1 + 2 * i + 1], MAX_FILENAME - 1);
//             tasks[i].output[MAX_FILENAME - 1] = '\0';
//         }

//         struct timeval tstart, tend;
//         gettimeofday(&tstart, NULL);

//         int task_index = 0;
//         int tasks_done = 0;
//         MPI_Status status;

//         /* 先给每个 worker 分配任务 */
//         for (int dest = 1; dest < size; dest++) {
//             if (task_index < ntasks) {
//                 MPI_Send(&tasks[task_index], sizeof(task_t), MPI_BYTE, dest, TAG_TASK, MPI_COMM_WORLD);
//                 task_index++;
//             } else {
//                 task_t terminator;
//                 terminator.input[0] = '\0';
//                 MPI_Send(&terminator, sizeof(task_t), MPI_BYTE, dest, TAG_TASK, MPI_COMM_WORLD);
//             }
//         }

//         /* 处理完成信号并继续分配任务 */
//         while (tasks_done < ntasks) {
//             int dummy;
//             int source;
//             MPI_Recv(&dummy, 1, MPI_INT, MPI_ANY_SOURCE, TAG_DONE, MPI_COMM_WORLD, &status);
//             source = status.MPI_SOURCE;
//             tasks_done++;

//             if (task_index < ntasks) {
//                 MPI_Send(&tasks[task_index], sizeof(task_t), MPI_BYTE, source, TAG_TASK, MPI_COMM_WORLD);
//                 task_index++;
//             } else {
//                 task_t terminator;
//                 terminator.input[0] = '\0';
//                 MPI_Send(&terminator, sizeof(task_t), MPI_BYTE, source, TAG_TASK, MPI_COMM_WORLD);
//             }
//         }

//         gettimeofday(&tend, NULL);
//         double dt = (tend.tv_sec - tstart.tv_sec) + ((tend.tv_usec - tstart.tv_usec) / 1e6);
//         printf("Master: All tasks processed in %lf s\n", dt);
//         free(tasks);
//     } else {
//         /* Worker 节点 */
//         while (1) {
//             task_t task;
//             MPI_Status status;
//             MPI_Recv(&task, sizeof(task_t), MPI_BYTE, 0, TAG_TASK, MPI_COMM_WORLD, &status);

//             if (task.input[0] == '\0') {
//                 break;
//             }

//             struct timeval tstart, tend;
//             gettimeofday(&tstart, NULL);
//             animated_gif *image = load_pixels(task.input);

//             if (image == NULL) {
//                 fprintf(stderr, "Rank %d: Error loading %s\n", rank, task.input);
//             } else {
//                 printf("Rank %d: Processing %s with %d frame(s)\n", rank, task.input, image->n_images);
//                 fflush(stdout);

//                 apply_gray_filter(image);
//                 apply_blur_filter(image, 5, 20);
//                 apply_sobel_filter(image);                                                                                                   

//                 if (!store_pixels(task.output, image)) {
//                     fprintf(stderr, "Rank %d: Error storing %s\n", rank, task.output);
//                 }

//                 gettimeofday(&tend, NULL);
//                 double dt = (tend.tv_sec - tstart.tv_sec) + ((tend.tv_usec - tstart.tv_usec) / 1e6);
//                 printf("Rank %d: Processed %s -> %s in %lf s\n", rank, task.input, task.output, dt);
//                 fflush(stdout);
//             }

//             /* 发送完成信号，无论任务是否成功 */
//             int dummy = 1;
//             MPI_Send(&dummy, 1, MPI_INT, 0, TAG_DONE, MPI_COMM_WORLD);
//         }
//     }

//     MPI_Finalize();
//     return 0;
// }



