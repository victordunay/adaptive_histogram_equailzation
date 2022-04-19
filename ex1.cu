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
 __device__ void create_histogram(int image_start, int t_row, int t_col ,int *histograms, uchar *image)
 {
    //We can accelerate this compute - https://classroom.udacity.com/courses/cs344/lessons/5605891d-c8bf-4e0d-8fed-a47920df5979/concepts/b42e8f5a-9145-450e-8c18-f23e091d33ef
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    // initialize cdf
    if(tid < N_BINS)
    {
        histograms[tid] = 0;
    }

    int row_base_offset = (t_row * TILE_WIDTH + threadIdx.y) * IMG_WIDTH ;
    int row_interval = N_THREADS_Y * IMG_WIDTH;
    int col_offset = t_col * TILE_WIDTH + threadIdx.x; 
    uchar pixel_value = 0;
    for(int i = 0; i < TILE_WIDTH/N_THREADS_Y; i++ )
    {
        pixel_value = image[image_start + row_base_offset + (i * row_interval) + col_offset];
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

__device__ void calculate_maps(int map_start, int t_row, int t_col, int *cdf, uchar *maps)
{
    uchar div_result = (uchar) 0;
    int tid = blockDim.x * threadIdx.y + threadIdx.x;
    if (tid < N_BINS)
    {
        div_result = (uchar)(cdf[tid] * 255.0/(TILE_WIDTH*TILE_WIDTH));
        maps[map_start + (t_col + t_row * TILE_COUNT)*N_BINS + tid] = div_result;
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
    int image_start = IMG_WIDTH * IMG_HEIGHT * blockIdx.x;
    int map_start = TILE_COUNT * TILE_COUNT * N_BINS * blockIdx.x;
    for(int t_row = 0; t_row< TILE_COUNT; ++t_row)
    {
        for(int t_col = 0; t_col< TILE_COUNT; ++t_col)
        {
            create_histogram(image_start,t_row, t_col, cdf, all_in);
            __syncthreads();
            prefix_sum(cdf, N_BINS);
            calculate_maps(map_start, t_row, t_col,cdf, maps); 
            __syncthreads();
        }
    }
    interpolate_device(&maps[map_start],&all_in[image_start], &all_out[image_start]);
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
    CUDA_CHECK( cudaHostAlloc(&(context->image_in), IMG_WIDTH * IMG_WIDTH,0) );
    CUDA_CHECK( cudaHostAlloc(&(context->image_out), IMG_WIDTH * IMG_WIDTH,0) );
    CUDA_CHECK( cudaHostAlloc(&(context->maps), TILE_COUNT * TILE_COUNT * N_BINS,0) );

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
    
    dim3 GRID_SIZE(TILE_WIDTH, N_THREADS_Y , 1);

    int image_index = 0;

    for (; image_index < N_IMAGES ; ++image_index)
    {
         //   1. copy the relevant image from images_in to the GPU memory you allocated
        CUDA_CHECK( cudaMemcpy(context->image_in, &images_in[image_index * IMG_WIDTH * IMG_HEIGHT], IMG_WIDTH * IMG_HEIGHT, cudaMemcpyDeviceToDevice) );

        //   2. invoke GPU kernel on this image
        process_image_kernel<<<1, GRID_SIZE>>>((context->image_in), (context->image_out), context->maps); 
        cudaDeviceSynchronize();

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
/*                                      bulk                                            */
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
    dim3 GRID_SIZE(TILE_WIDTH, N_THREADS_Y , 1);
       
    CUDA_CHECK( cudaMemcpy(context->image_in, images_in,N_IMAGES * IMG_WIDTH * IMG_HEIGHT, cudaMemcpyDeviceToDevice) );

    //TODO: invoke a kernel with N_IMAGES threadblocks, each working on a different image
    process_image_kernel<<<N_IMAGES, GRID_SIZE>>>((context->image_in), (context->image_out), context->maps); 
    cudaDeviceSynchronize();

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
