#include "ex1.h"

__device__ void prefix_sum(int arr[], int arr_size) 
{
    int tid = TILE_WIDTH * threadIdx.y + threadIdx.x;
    int increment;
    for (int stride = 1; stride < arr_size; stride *= 2) 
    {
        if (tid >= stride) 
            increment = arr[tid - stride];
        __syncthreads();
        if (tid >= stride) 
            arr[tid] += increment;
        __syncthreads();
    }
    return;
}

__device__ void convert_image_to_tiles(uchar *tiles,  uchar *images_in)
{

  //int tile_index = blockIdx.x + N_BLOCKS_Y * blockIdx.y
  int in_pixel_index = (blockIdx.y * TILE_WIDTH + threadIdx.y) * IMG_WIDTH + blockIdx.x * TILE_WIDTH + threadIdx.x; // why do access the pixel as an array instead of matrix?
  int tile_pixel_index = threadIdx.y * TILE_DIM + threadIdx.x;
  tiles[tile_pixel_index] = images_in[pixel_index];
}

__device__ void calculate_maps(int *s_maps, int *cdfs, uchar *g_maps)
{
    int tile_index = blockIdx.x + N_BLOCKS_Y * blockIdx.y
    int cdf_index = threadIdx.x + MAP_TILE_WIDTH * threadIdx.y;

    if (threadIdx.x < MAP_TILE_WIDTH && threadIdx.y < MAP_TILE_WIDTH)
    {
        s_maps[blockIdx.x][blockIdx.y] = cdfs[tile_index][cdf_index] * NORMALIZATION_FACTOR;
    }

    // copy shared maps to global maps
    if (threadIdx.x < MAP_TILE_WIDTH && threadIdx.y < MAP_TILE_WIDTH)
    {
        int g_maps_index = blockIdx.x * N_BINS + blockIdx.y * N_BINS * N_BLOCKS_X + cdf_index;
        g_maps[g_maps_index] = s_maps[blockIdx.x][blockIdx.y][cdf_index]; 
    }         
}
}
bins_array map[N_BLOCKS_X][N_BLOCKS_Y];

__device__ void calculate_cdf(int *cdfs)
{
    int tile_index = blockIdx.x + N_BLOCKS_Y * blockIdx.y    
    cudaMemset(&cdfs, 0, N_BINS * sizeof(int));
    __syncthreads();
    prefix_sum(&cdfs, N_BINS);
    
}

__device__ void create_histogram(int *histograms, uchar *tiles)
{
    //We can accelerate this compute - https://classroom.udacity.com/courses/cs344/lessons/5605891d-c8bf-4e0d-8fed-a47920df5979/concepts/b42e8f5a-9145-450e-8c18-f23e091d33ef
    //int tile_index = blockIdx.x + N_BLOCKS_Y * blockIdx.y    
    uchar pixel_value = 0;
    cudaMemset(histograms, 0, N_BINS * sizeof(int)); //I think it is a cpu function. what will the threads do?
    __syncthreads();

    pixel_value = tiles[threadIdx.x + TILE_WIDTH * threadIdx.y];
    atomicAdd(&(histograms[pixel_value]), 1);
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
 * @brief takes an image given in all_in, and return the processed image in all_out 
 * 
 * @param all_in single input image, in global memory.
 * @param all_out single output image, in global memory.
 * @param maps 3D array ([TILES_COUNT][TILES_COUNT][256]) of    
 *             the tiles’ maps, in global memory.
 * @return __global__ 
 */
__global__ void process_image_kernel(uchar *all_in, uchar *all_out, uchar *maps) 
{
    //every block allocates the shared-mem for itself
    //we need only two array for this calculation
    __shared__ int cdfs[N_BINS];
    __shared__ uchar tiles[TILE_WIDTH * TILE_WIDTH];

    convert_image_to_tiles(tiles, all_in)

    create_histogram(cdfs, tiles)

    calculate_cdf(cdfs)

    calculate_maps(cdfs, maps)

    interpolate_device(all_in, all_out, maps);

    return; 
}

/* Task serial context struct with necessary CPU / GPU pointers to process a single image */
struct task_serial_context
 {
    // TODO define task serial memory buffers
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

    //TODO: allocate GPU memory for a single input image, a single output image, and maps
    CUDA_CHECK( cudaHostAlloc(context->image_in, IMG_HEIGHT * IMG_WIDTH, 0) );
    CUDA_CHECK( cudaHostAlloc(context->image_out, IMG_HEIGHT * IMG_WIDTH, 0) );
    CUDA_CHECK( cudaHostAlloc(context->maps, TILES_COUNT * TILES_COUNT * N_BINS, 0) );

    return context;
}

/* Process all the images in the given host array and return the output in the
 * provided output host array */
void task_serial_process(struct task_serial_context *context, uchar *images_in, uchar *images_out)
{
    //TODO: in a for loop:
    //   1. copy the relevant image from images_in to the GPU memory you allocated
    //   2. invoke GPU kernel on this image
    //   3. copy output from GPU memory to relevant location in images_out_gpu_serial
    
    
    dim3 BLOCK_SIZE(N_BLOCKS_X, N_BLOCKS_Y);
    dim3 GRID_SIZE(TILE_WIDTH, TILE_WIDTH);

    int image_index = 0;

    for (; image_index < N_IMAGES ; ++image_index)
    {
         //   1. copy the relevant image from images_in to the GPU memory you allocated
        CUDA_CHECK( cudaMemcpy(context->image_in, &images_in[image_index * IMG_WIDTH * IMG_HEIGHT], IMG_WIDTH * IMG_HEIGHT * sizeof(uchar), cudaMemcpyHostToDevice) );
        
        //   2. invoke GPU kernel on this image
        process_image_kernel<<<BLOCK_SIZE, GRID_SIZE>>>(context->image_in, context->image_out, context->maps);
        
        //   3. copy output from GPU memory to relevant location in images_out_gpu_serial
        CUDA_CHECK( cudaMemcpy(context->image_out, &images_out[image_index * IMG_WIDTH * IMG_HEIGHT], IMG_WIDTH * IMG_HEIGHT * sizeof(uchar), cudaMemcpyDeviceToDevice) );
    }
}

/* Release allocated resources for the task-serial implementation. */
void task_serial_free(struct task_serial_context *context)
{
    //TODO: free resources allocated in task_serial_init

    free(context);
}

/* Bulk GPU context struct with necessary CPU / GPU pointers to process all the images */
struct gpu_bulk_context {
    // TODO define bulk-GPU memory buffers
};

/* Allocate GPU memory for all the input images, output images, and maps.
 * 
 * Returns: allocated and initialized gpu_bulk_context. */
struct gpu_bulk_context *gpu_bulk_init()
{
    auto context = new gpu_bulk_context;

    //TODO: allocate GPU memory for all the input images, output images, and maps

    return context;
}

/* Process all the images in the given host array and return the output in the
 * provided output host array */
void gpu_bulk_process(struct gpu_bulk_context *context, uchar *images_in, uchar *images_out)
{
    //TODO: copy all input images from images_in to the GPU memory you allocated
    //TODO: invoke a kernel with N_IMAGES threadblocks, each working on a different image
    //TODO: copy output images from GPU memory to images_out
}

/* Release allocated resources for the bulk GPU implementation. */
void gpu_bulk_free(struct gpu_bulk_context *context)
{
    //TODO: free resources allocated in gpu_bulk_init

    free(context);
}
