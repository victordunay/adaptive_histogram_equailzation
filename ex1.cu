#include "ex1.h"

// user defines
#define N_BINS (256)
#define MAP_TILE_WIDTH (16)
#define N_BLOCKS ( (IMG_HEIGHT * IMG_WIDTH) / (TILE_WIDTH * TILE_WIDTH) )
#define N_BLOCKS_X (IMG_WIDTH / TILE_WIDTH)
#define N_BLOCKS_Y (IMG_HEIGHT / TILE_WIDTH)
#define NORMALIZATION_FACTOR  ( (N_BINS - 1) / (TILE_WIDTH * TILE_WIDTH) )
#define N_THREADS_Y (16)
#define N_THREADS_X (TILE_WIDTH)
#define N_THREADS_Z (1)
#define N_TB_SERIAL (1)
#define N_TB_BULK (N_IMAGES)

 /**
  * @brief Create a histogram of the tile pixels. Assumes that the kernel runs with more than 256 threads
  * 
  * @param image_start The start index of the image the block processing
  * @param t_row  The tile's row
  * @param t_col  The tile's column
  * @param histogram - the histogram of the tile
  * @param image - The images array to process.
  */
 __device__ void create_histogram(int image_start, int t_row, int t_col ,int *histogram, uchar *image)
 {
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    // initialize histogram
    if(tid < N_BINS)
    {
        histogram[tid] = 0;
    }

    //calculates the pixel index that assigned to the thread 
    int row_base_offset = (t_row * TILE_WIDTH + threadIdx.y) * IMG_WIDTH ;
    int row_interval = N_THREADS_Y * IMG_WIDTH;
    int col_offset = t_col * TILE_WIDTH + threadIdx.x; 

    uchar pixel_value = 0;

    //The block has 16 rows, Therefore, it runs 4 times so every warp run on 4 different rows
    for(int i = 0; i < TILE_WIDTH/N_THREADS_Y; i++ ) 
    {
        pixel_value = image[image_start + row_base_offset + (i * row_interval) + col_offset];
        atomicAdd(&(histogram[pixel_value]), 1);
    } 
 }

 /**
  * @brief Calculates inclusive prefix sum of the given array. Saves the sum in the given array.
  *      Assumes n_threads > arr_size
  * 
  * @param arr The given array 
  * @param arr_size The size of the array
  */
__device__ void prefix_sum(int arr[], int arr_size) 
{
    int tid = blockDim.x * threadIdx.y + threadIdx.x;
    int increment = 0;

    for (int stride = 1; stride < arr_size; stride *= 2) 
    {
        if (tid >= stride && tid < arr_size) 
                increment = arr[tid - stride];
        __syncthreads(); 
        if (tid >= stride && tid < arr_size) 
                arr[tid] += increment;
        __syncthreads();
    }
    return;
}

/**
 * @brief Calculates a map from the cdf and saves it in the given index in the 'maps' array.
 * 
 * @param map_start The start index in the 'maps' array of the current image's map
 * @param t_row The tile's row
 * @param t_col The tile's column
 * @param cdf The cdf of the tile.
 * @param maps Array of the maps of all images
 * @return __device__ 
 */
__device__ void calculate_maps(int map_start, int t_row, int t_col, int *cdf, uchar *maps)
{
    uchar div_result = 0;
    int tid = blockDim.x * threadIdx.y + threadIdx.x;

    if (tid < N_BINS)
    {
        div_result = (uchar)( cdf[tid] * 255.0 / ( TILE_WIDTH * TILE_WIDTH ) );
        maps[map_start + (t_col + t_row * TILE_COUNT) * N_BINS + tid] = div_result;
    }   
}

/**
 * Perform interpolation on a single image
 *
 * @param maps 3D array ([TILES_COUNT][TILES_COUNT][256]) of    
 *             the tiles’ maps, in global memory.
 * @param in_img single input image, in global memory.
 * @param out_img single output buffer, in global memory.
 */
__device__ void interpolate_device(uchar* maps ,uchar *in_img, uchar* out_img);

/**
 * @brief process an image which assigned to the block index. It takes an image given in all_in, and return the processed image in all_out respectively.
 * 
 * @param all_in Array of input images, in global memory ([N_IMAGES][IMG_HEIGHT][IMG_WIDTH])
 * @param all_out Array of output images, in global memory ([N_IMAGES][IMG_HEIGHT][IMG_WIDTH])
 * @param maps 4D array ([N_IMAGES][TILES_COUNT][TILES_COUNT][256]) of    
 *             the tiles’ maps, in global memory. 
 */
__global__ void process_image_kernel(uchar *all_in, uchar *all_out, uchar *maps) 
{
    // the cumulative distribution function is calculated in place . 
    // thus the cdf shared memory is also used for histogram result.
    __shared__ int cdf[N_BINS];

    // indicates the first pixel index of the blockIdx.x's image
    int image_start = IMG_WIDTH * IMG_HEIGHT * blockIdx.x;

    // indicates the first map index of the blockIdx.x's image
    int map_start = TILE_COUNT * TILE_COUNT * N_BINS * blockIdx.x;

    for(int t_row = 0; t_row < TILE_COUNT; ++t_row)
    {
        for(int t_col = 0; t_col< TILE_COUNT; ++t_col)
        {
            create_histogram(image_start, t_row, t_col, cdf, all_in);

            __syncthreads();
            // update cdf content from histogram to cumulative distribution function
            prefix_sum(cdf, N_BINS);

            calculate_maps(map_start, t_row, t_col, cdf, maps); 

           __syncthreads();
        }
    }
    // perform pixel correction
    interpolate_device(maps + map_start, all_in + image_start, all_out + image_start);

    return; 
}

/* Task serial context struct with necessary CPU / GPU pointers to process a single image */
struct task_serial_context
 {
    uchar *image_in;
    uchar *image_out;
    uchar *maps;
};

/* Allocate GPU memory for a single input image and a single output image.
 * 
 * Returns: allocated and initialized task_serial_context. */
struct task_serial_context *task_serial_init()
{
    auto context = new task_serial_context;

    // allocate GPU memory for a single input image, a single output image, and maps
    CUDA_CHECK( cudaHostAlloc(&(context->image_in), IMG_WIDTH * IMG_WIDTH,0) );
    CUDA_CHECK( cudaHostAlloc(&(context->image_out), IMG_WIDTH * IMG_WIDTH,0) );
    CUDA_CHECK( cudaHostAlloc(&(context->maps), TILE_COUNT * TILE_COUNT * N_BINS,0) );

    return context;
}

/* Process all the images in the given host array and return the output in the
 * provided output host array */
void task_serial_process(struct task_serial_context *context, uchar *images_in, uchar *images_out)
{
    //   in a for loop:
    //   1. copy the relevant image from images_in to the GPU memory you allocated
    //   2. invoke GPU kernel on this image
    //   3. copy output from GPU memory to relevant location in images_out_gpu_serial
    
    dim3 GRID_SIZE(N_THREADS_X, N_THREADS_Y , N_THREADS_Z);

    int image_index = 0;

    for (; image_index < N_IMAGES ; ++image_index)
    {
         //   1. copy the relevant image from images_in to the GPU memory you allocated
        CUDA_CHECK( cudaMemcpy(context->image_in, &images_in[image_index * IMG_WIDTH * IMG_HEIGHT], IMG_WIDTH * IMG_HEIGHT, cudaMemcpyDeviceToDevice) );

        //   2. invoke GPU kernel on this image
        process_image_kernel<<<N_TB_SERIAL, GRID_SIZE>>>((context->image_in), (context->image_out), context->maps); 

        //   3. copy output from GPU memory to relevant location in images_out_gpu_serial
        CUDA_CHECK( cudaMemcpy(&images_out[image_index * IMG_WIDTH * IMG_HEIGHT],context->image_out, IMG_WIDTH * IMG_HEIGHT, cudaMemcpyDeviceToDevice) );
    }
}
/* Release allocated resources for the task-serial implementation. */
void task_serial_free(struct task_serial_context *context)
{
    //TODO: free resources allocated in task_serial_init
    cudaFree(context->image_in);
    cudaFree(context->image_out);
    cudaFree(context->maps);
    free(context);
}
/****************************************************************************************/
/*                                      Bulk                                            */
/****************************************************************************************/

/* Bulk GPU context struct with necessary CPU / GPU pointers to process all the images */
struct gpu_bulk_context {
    // TODO define bulk-GPU memory buffers
    uchar *image_in;
    uchar *image_out;
    uchar *maps;
};

/* Allocate GPU memory for all the input images, output images, and maps.
 * 
 * Returns: allocated and initialized gpu_bulk_context. */
struct gpu_bulk_context *gpu_bulk_init()
{
    auto context = new gpu_bulk_context;

    //TODO: allocate GPU memory for all the input images, output images, and maps
    CUDA_CHECK( cudaHostAlloc(&(context->image_in),N_IMAGES * IMG_WIDTH * IMG_WIDTH ,0) );
    CUDA_CHECK( cudaHostAlloc(&(context->image_out),N_IMAGES * IMG_WIDTH * IMG_WIDTH,0) );
    CUDA_CHECK( cudaHostAlloc(&(context->maps),N_IMAGES * TILE_COUNT * TILE_COUNT * N_BINS,0) );

    return context;
}

/* Process all the images in the given host array and return the output in the
 * provided output host array */
void gpu_bulk_process(struct gpu_bulk_context *context, uchar *images_in, uchar *images_out)
{
    //TODO: copy all input images from images_in to the GPU memory you allocated
    dim3 GRID_SIZE(N_THREADS_X, N_THREADS_Y , N_THREADS_Z);
       
    CUDA_CHECK( cudaMemcpy(context->image_in, images_in,N_IMAGES * IMG_WIDTH * IMG_HEIGHT, cudaMemcpyDeviceToDevice) );

    //TODO: invoke a kernel with N_IMAGES threadblocks, each working on a different image
    process_image_kernel<<<N_TB_BULK, GRID_SIZE>>>((context->image_in), (context->image_out), context->maps); 

    //TODO: copy output images from GPU memory to images_out
    CUDA_CHECK( cudaMemcpy(images_out,context->image_out,N_IMAGES * IMG_WIDTH * IMG_HEIGHT, cudaMemcpyDeviceToDevice) );
}

/* Release allocated resources for the bulk GPU implementation. */
void gpu_bulk_free(struct gpu_bulk_context *context)
{
    //TODO: free resources allocated in gpu_bulk_init
    cudaFree(context->image_in);
    cudaFree(context->image_out);
    cudaFree(context->maps);
    free(context);
}
