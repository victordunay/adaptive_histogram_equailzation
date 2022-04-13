#include "ex1.h"

__device__ void prefix_sum(int arr[], int arr_size) 
{
    int tid = TILE_WIDTH*threadIdx.y + threadIdx.x;
    int increment;
    for (int stride = 1; stride < arr_size; stride *= 2) {
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

  int x = blockIdx.x * TILE_WIDTH + threadIdx.x;
  int y = blockIdx.y * TILE_WIDTH + threadIdx.y;

  for (int tile_index = 0; tile_index < N_BLOCKS; ++tile_index )
  {
     tiles[tile_index][threadIdx.y * TILE_DIM + threadIdx.x] = images_in[y * IMG_WIDTH + x];
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
__device__ 
void interpolate_device(uchar* maps ,uchar *in_img, uchar* out_img);

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
    __shared__ histogram histograms[N_BLOCKS];
    __shared__ tile tiles[N_BLOCKS];

    convert_image_to_tiles(&tiles, &all_in)
    __syncthreads();
    // create histogram

    // use prefix scan to generate cdf

    // calculate map

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
    CUDA_CHECK(cudaHostAlloc(context->image_in, IMG_HEIGHT * IMG_WIDTH, 0));
    CUDA_CHECK(cudaHostAlloc(context->image_out, IMG_HEIGHT * IMG_WIDTH, 0));
    CUDA_CHECK(cudaHostAlloc(context->maps, TILES_COUNT * TILES_COUNT * N_BINS, 0));

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
    
    int image_index = 0;
    dim3 block_size(IMG_WIDTH / TILE_WIDTH, IMG_HEIGHT / TILE_WIDTH);
    dim3 grid_size(TILE_WIDTH, TILE_WIDTH);
  __shared__ tile tiles_array[N_BLOCKS];









    for (image_index = 0; image_index < N_IMAGES ; ++image_index)
    {
         //   1. copy the relevant image from images_in to the GPU memory you allocated
        cudaMemcpy(context->image_in, &images_in[image_index * IMG_WIDTH * IMG_HEIGHT], IMG_WIDTH * IMG_HEIGHT * sizeof(uchar), cudaMemcpyHostToDevice);
        
        //   2. invoke GPU kernel on this image , fix num of threadblocks & threads
        process_image_kernel<<<block_size, grid_size>>>(context->image_in, context->image_out, context->maps);
        
        //   3. copy output from GPU memory to relevant location in images_out_gpu_serial
        cudaMemcpy(context->image_out, &images_out[image_index * IMG_WIDTH * IMG_HEIGHT], IMG_WIDTH * IMG_HEIGHT * sizeof(uchar), cudaMemcpyDeviceToDevice);

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
