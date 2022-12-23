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

static __device__ __host__ inline cufftComplex operator+(cufftComplex, cufftComplex);
static __device__ __host__ inline cufftComplex operator*(cufftComplex, float);
static __device__ __host__ inline cufftComplex operator*(cufftComplex, cufftComplex);

static __global__ void Convolution_calc(cufftComplex*, cufftComplex* );

int main() {
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
////////////////////////////////////////////////////////////////////////////////
// Complex operations
////////////////////////////////////////////////////////////////////////////////

// Complex addition
static __device__ __host__ inline cufftComplex operator+(cufftComplex a, cufftComplex b) {
    cufftComplex c;
    c.x = a.x + b.x;
    c.y = a.y + b.y;
    return c;
}

// Complex scale
static __device__ __host__ inline cufftComplex operator*(cufftComplex a, float s) {
    cufftComplex c;
    c.x = s * a.x;
    c.y = s * a.y;
    return c;
}

// Complex multiplication
static __device__ __host__ inline cufftComplex operator*(cufftComplex a, cufftComplex b) {
    cufftComplex c;
    c.x = a.x * b.x - a.y * b.y;
    c.y = a.x * b.y + a.y * b.x;
    return c;
}
