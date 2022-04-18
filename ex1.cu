#include "ex1.h"
#include "stdlib.h"

// user defines
#define N_BINS (256)
#define MAP_TILE_WIDTH (16)
#define N_BLOCKS ( (IMG_HEIGHT * IMG_WIDTH) / (TILE_WIDTH * TILE_WIDTH) )
#define N_BLOCKS_X (IMG_WIDTH / TILE_WIDTH)
#define N_BLOCKS_Y (IMG_HEIGHT / TILE_WIDTH)
#define NORMALIZATION_FACTOR  ( (N_BINS - 1) / (TILE_WIDTH * TILE_WIDTH) )
#define N_THREADS_Y (16)


/**
 * @brief Create a histogram from the fitting tile of image given
 * 
 * @param histograms  the histogram arry to fill
 * @param image the image to make the histagram from 
 * @return __device__ 
 */
 __device__ void create_histogram(int *histograms, uchar *image)
 {
    //We can accelerate this compute - https://classroom.udacity.com/courses/cs344/lessons/5605891d-c8bf-4e0d-8fed-a47920df5979/concepts/b42e8f5a-9145-450e-8c18-f23e091d33ef
    uchar pixel_value = 0;
    int in_pixel_index_y = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int in_pixel_index_x = blockIdx.x * TILE_WIDTH + threadIdx.x; 

    for(int i = 0; i < TILE_WIDTH/N_THREADS_Y; i++ )
    {
        pixel_value = image[in_pixel_index_x + (in_pixel_index_y + i*N_THREADS_Y) * IMG_WIDTH];
        atomicAdd(&(histograms[pixel_value]), 1);
    }  
 }

__device__ void prefix_sum(int arr[], int arr_size) 
{
    int tid = blockDim.x * threadIdx.y + threadIdx.x;
    int increment;
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

__device__ void calculate_maps(int *cdf, uchar *maps)
{
    uchar div_result = (uchar) 0;
    int tid = blockDim.x * threadIdx.y + threadIdx.x;
    if (tid < N_BINS)
    {
        div_result = (uchar)(cdf[tid] * 255.0/(64*64));
        cdf[tid] = (int) div_result;
        maps[(blockIdx.x + blockIdx.y * TILE_COUNT)*N_BINS + tid] = div_result;
    }   
    __syncthreads();     
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

    __shared__ int cdf[N_BINS];
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    // initialize cdf
    if(tid < N_BINS)
    {
        cdf[tid] = 0;
    }
    __syncthreads();
    create_histogram(cdf, all_in);
    __syncthreads();
    prefix_sum(cdf, N_BINS);
    calculate_maps(cdf, maps);
    interpolate_device(maps,all_in, all_out);
    __syncthreads();
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
    CUDA_CHECK( cudaHostAlloc(&(context->image_in), IMG_WIDTH * IMG_WIDTH*sizeof(uchar),0) );
    CUDA_CHECK( cudaHostAlloc(&(context->image_out), IMG_WIDTH * IMG_WIDTH*sizeof(uchar),0) );
    CUDA_CHECK( cudaHostAlloc(&(context->maps), TILE_COUNT * TILE_COUNT * N_BINS*sizeof(uchar),0) );

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
    
    
    dim3 BLOCK_SIZE(N_BLOCKS_X, N_BLOCKS_Y, 1);
    dim3 GRID_SIZE(TILE_WIDTH, N_THREADS_Y , 1);

    int image_index = 0;

    for (; image_index < N_IMAGES ; ++image_index)
    {
         //   1. copy the relevant image from images_in to the GPU memory you allocated
        CUDA_CHECK( cudaMemcpy(context->image_in, &images_in[image_index * IMG_WIDTH * IMG_HEIGHT], IMG_WIDTH * IMG_HEIGHT, cudaMemcpyDeviceToDevice) );
        
        //   2. invoke GPU kernel on this image
        process_image_kernel<<<BLOCK_SIZE, GRID_SIZE>>>((context->image_in), (context->image_out), context->maps); 

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
