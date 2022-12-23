#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cufft.h>
#include <cufftXt.h>

#include <cuda_runtime.h>

//#include <helper_cuda.h>
//#include <helper_functions.h>

#include <curand.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/complex.h>

#include <stdio.h>

#include <iostream>
#include <functional>

#define CELLSIZE 6
#define VECDIM 3 


int main() {

    //gpu_params();

    //float timerValueGPU, timerValueCPU;
    //cudaEvent_t start, stop;
    //cudaEventCreate(&start);
    //cudaEventCreate(&stop);

    //cudaSetDevice(0);

    int grid_dimX, grid_dimY, grid_dimZ; // actual number of cells is 

    grid_dimX = 128, grid_dimY = 128, grid_dimZ = 1;

    float deltaX, deltaY, deltaZ;

    deltaX = 1, deltaY = 1, deltaZ = 1;

    //TEST_F_grid_maker();

    //--------------- EXCHANGE FIELD

    //thrust::host_vector<float3> mag_grid(grid_dimZ * grid_dimX * grid_dimY);

    //thrust::device_vector<float3> exch_field(grid_dimZ * grid_dimX * grid_dimY);

    //cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    //cudaArray_t cuArray;
    ////cudaMallocArray(&cuArray, &channelDesc, width, height);
    

    //16384 x 16384 x 16384 -- maximum CudaArray3D size
    
    //float3 test = make_float3(1, 2, 3);
    //std::cout << sizeof(float3) << "\n";

    //thrust::host_vector<float3> ha(27), hb(27);
    //for (int i = 0; i < 27; i++) {
    //    ha[i] = make_float3(i, i, i);
    //    hb[i] = make_float3(1,2,3);
    //}
    //thrust::device_vector<float3> da(27), db(27), doutput(27);
    //da = ha;
    //db = hb;
    //dim3 blockSize(3, 3);
    //dim3 gridSize(1, 1, 3);

    //cudaEventRecord(start, 0);

    //add <<< gridSize, blockSize >>> ((float3*)thrust::raw_pointer_cast(da.data()), (float3*)thrust::raw_pointer_cast(db.data()), (float3*)thrust::raw_pointer_cast(doutput.data()));
    //ha = doutput;
    //for (int i = 0; i < 27; i++) {
    //    std::cout << ha[i].x << " " << ha[i].y << " " << ha[i].z << " \n";
    //}

    //auto val = F_grid_maker(256, 256, 1, 1, 1, 1);

    //TEST_F_grid_maker();


    //cudaEventRecord(stop, 0);
    //cudaEventSynchronize(stop);
    //cudaEventElapsedTime(&timerValueGPU, start, stop);
    //printf("\n GPU calculation time %f msec\n", timerValueGPU);

    grid_dimX = 4096;
    grid_dimY = 128;
    grid_dimZ = 3;

    const long long int ext_dimX = 2 * grid_dimX;
    const long long int ext_dimY = 2 * grid_dimY;
    const long long int ext_dimZ = 2 * grid_dimZ;
    const long long int FULLSIZE = ext_dimX * ext_dimY * ext_dimZ * 6;

    int error_value;

    //input[b * idist + ((z * inembed[1] + y) * inembed[2] + x) * istride]

    //output[b * odist + ((z * onembed[1] + y) * onembed[2] + x) * ostride]

    thrust::host_vector<cufftComplex> host_grid(FULLSIZE);
    for (int z = 0; z < ext_dimZ; z++) {
        for (int y = 0; y < ext_dimY; y++) {
            for (int x = 0; x < ext_dimX; x++) {
                for (int b = 0; b < 6; b++) {
                    host_grid[z * ext_dimX * ext_dimY*6 + y * ext_dimX*6 + x*6 + b].x = (b + 1.f);
                }
            }
        }
    }


    long long int n[3] = { ext_dimZ, ext_dimY, ext_dimX };
    long long int inembed[3] = { 1, ext_dimY, ext_dimX };
    long long int idist = 1;
    long long int istride = 6;

    cudaSetDevice(0);

    float timerValueGPU, timerValueCPU;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    thrust::device_vector<cufftComplex> input_device_grid = host_grid;
    thrust::device_vector<cufftComplex> output_device_grid(FULLSIZE);

    cufftHandle plan_adv;
    size_t workSize;
    cufftCreate(&plan_adv);
    cufftXtMakePlanMany(plan_adv, 3, n, inembed, istride, idist, CUDA_C_32F, inembed, istride, idist, CUDA_C_32F, 6, &workSize, CUDA_C_32F);
    printf("Temporary buffer size %li bytes\n", workSize);

    cudaEventRecord(start, 0);

    error_value = cufftExecC2C(plan_adv, (cufftComplex*)thrust::raw_pointer_cast(input_device_grid.data()), (cufftComplex*)thrust::raw_pointer_cast(input_device_grid.data()), CUFFT_FORWARD);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timerValueGPU, start, stop);
    printf("\n GPU calculation time %f msec\n", timerValueGPU);

    return 0;
}
