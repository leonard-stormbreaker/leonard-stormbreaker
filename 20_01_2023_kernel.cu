
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

using Tensor = std::vector<std::vector<std::vector<std::vector<float>>>>;
using FTensor = std::vector<std::vector<std::vector<std::vector<std::complex<float>>>>>;

float f(const float& x, const float& y, const float& z) {
    float x2 = std::pow(x, 2.0);
    float y2 = std::pow(y, 2.0);
    float z2 = std::pow(z, 2.0);

    float R = std::sqrt(std::pow(x, 2.0) + std::pow(y, 2.0) + std::pow(z, 2.0));

    float A = (x == 0 && z == 0) ? 0 : y / 2.0 * (z2 - x2) * std::asinh(y / std::sqrt(x2 + z2));
    float B = (x == 0 && y == 0) ? 0 : z / 2.0 * (y2 - x2) * std::asinh(z / std::sqrt(x2 + y2));
    float C = (x == 0) ? 0 : -x * y * z * std::atan((y * z) / (x * R));
    float D = 1.0 / 6.0 * (2.0 * x2 - y2 - z2) * R;

    return A + B + C + D;
}

float g(const float& x, const float& y, const float& z) {
    float x2 = std::pow(x, 2.0);
    float y2 = std::pow(y, 2.0);
    float z2 = std::pow(z, 2.0);

    float R = std::sqrt(std::pow(x, 2.0) + std::pow(y, 2.0) + std::pow(z, 2.0));

    float A = (x == 0 && y == 0) ? 0 : x * y * z * std::asinh(z / std::sqrt(x2 + y2));
    float B = (y == 0 && z == 0) ? 0 : y / 6 * (3.0 * z2 - y2) * std::asinh(x / std::sqrt(y2 + z2));
    float C = (x == 0 && z == 0) ? 0 : x / 6 * (3.0 * z2 - x2) * std::asinh(y / std::sqrt(x2 + z2));
    float D = (z == 0) ? 0 : std::pow(z, 3.0) / 6 * std::atan((x * y) / (z * R));
    float E = (y == 0) ? 0 : z * y2 / 2 * std::atan((x * z) / (y * R));
    float F = (x == 0) ? 0 : z * x2 / 2 * std::atan((y * z) / (x * R));
    float G = x * y * R / 3.0;

    return A + B + C - D - E - F - G;
}

float Nxx_calc(const float& x, const float& y, const float& z, const float& deltaX, const float& deltaY, const float& deltaZ, std::function<float(float, float, float)> f) {
    float set_0 = 8.0 * f(x, y, z);

    float set_A = -4.0 * (+f(x - deltaX, y, z) + f(x + deltaX, y, z) + f(x, y - deltaY, z) + f(x, y + deltaY, z) + f(x, y, z - deltaZ) + f(x, y, z + deltaZ));

    float set_B = +2.0 * (+f(x - deltaX, y - deltaY, z) + f(x - deltaX, y + deltaY, z) + f(x + deltaX, y - deltaY, z) + f(x + deltaX, y + deltaY, z)
        + f(x - deltaX, y, z - deltaZ) + f(x - deltaX, y, z + deltaZ) + f(x + deltaX, y, z - deltaZ) + f(x + deltaX, y, z + deltaZ)
        + f(x, y - deltaY, z - deltaZ) + f(x, y - deltaY, z + deltaZ) + f(x, y + deltaY, z - deltaZ) + f(x, y + deltaY, z + deltaZ));

    float set_C = -1.0 * (+f(x - deltaX, y - deltaY, z - deltaZ) + f(x - deltaX, y - deltaY, z + deltaZ) + f(x - deltaX, y + deltaY, z - deltaZ) + f(x - deltaX, y + deltaY, z + deltaZ)
        + f(x + deltaX, y - deltaY, z - deltaZ) + f(x + deltaX, y - deltaY, z + deltaZ) + f(x + deltaX, y + deltaY, z - deltaZ) + f(x + deltaX, y + deltaY, z + deltaZ));

    return set_0 + set_A + set_B + set_C;
}

void grid_printer(std::vector<std::vector<std::vector<float>>> grid) {
    int plane_cnt = 0;
    for (auto plane : grid) {
        std::cout << plane_cnt++ << "\n";
        for (auto row : plane) {
            std::cout << "\t";
            for (auto elem : row) {
                std::cout << elem << " ";
            }
            std::cout << "\n";
        }
    }
    return;
}
void host_printer_cufft(thrust::host_vector<thrust::host_vector<thrust::host_vector<cufftComplex>>> grid) {
    int plane_cnt = 0;
    for (auto plane : grid) {
        std::cout << plane_cnt++ << "\n";
        for (auto row : plane) {
            std::cout << "\t";
            for (auto elem : row) {
                std::cout << "(" << elem.x << ", " << elem.y << ") ";
            }
            std::cout << "\n";
        }
    }
    return;
}
void flat_host_printer(thrust::host_vector<thrust::complex<float>> grid, int dimz, int dimx, int dimy) {
    for (int k = 0; k < dimz; k++) {
        std::cout << k << "\n";
        for (int i = 0; i < dimx; i++) {
            std::cout << "\t";
            for (int j = 0; j < dimy; j++) {
                std::cout << grid[k * dimx * dimy + i * dimx + j] << " ";

            }
            std::cout << "\n";
        }
    }
}
void flat_host_printer_cufft(thrust::host_vector<cufftComplex> grid, int dimz, int dimx, int dimy) {
    for (int k = 0; k < dimz; k++) {
        std::cout << k << "\n";
        for (int i = 0; i < dimx; i++) {
            std::cout << "\t";
            for (int j = 0; j < dimy; j++) {
                std::cout << "(" << grid[k * dimx * dimy + i * dimx + j].x << ", " << grid[k * dimx * dimy + i * dimx + j].y << ") ";

            }
            std::cout << "\n";
        }
    }
}


thrust::host_vector<cufftComplex> Tensor_calc(const int& grid_dimX, const int& grid_dimY, const int& grid_dimZ, const float& deltaX, const float& deltaY, const float& deltaZ) {
    const int ext_dimX = 2 * grid_dimX;
    const int ext_dimY = 2 * grid_dimY;
    const int ext_dimZ = 2 * grid_dimZ;

    thrust::host_vector<cufftComplex> flat_host_grid(ext_dimZ * ext_dimX * ext_dimY * CELLSIZE);

    for (int cell = 0; cell < CELLSIZE; cell++) {
        for (int k = -grid_dimZ + 1; k < grid_dimZ; k++) {
            for (int i = -grid_dimX + 1; i < grid_dimX; i++) {
                for (int j = -grid_dimY + 1; j < grid_dimY; j++) { //[-dimY, dimY-1]

                    int kIdx = (k >= 0) ? k : 2 * grid_dimZ - std::abs(k);
                    int iIdx = (i >= 0) ? i : 2 * grid_dimX - std::abs(i);
                    int jIdx = (j >= 0) ? j : 2 * grid_dimY - std::abs(j);

                    float x = i * deltaX;
                    float y = j * deltaY;
                    float z = k * deltaZ;


                    switch (cell) { // Nxx - 0, Nxy - 1, Nxz - 2, Nyy - 3, Nyz - 4, Nzz = 5
                    case 0: // Nxx
                        //host_grid[cell][kIdx][iIdx][jIdx].x = Nxx_calc(x, y, z, deltaX, deltaY, deltaZ, &f);
                        flat_host_grid[cell * ext_dimZ * ext_dimX * ext_dimY + kIdx * ext_dimX * ext_dimY + iIdx * ext_dimY + jIdx].x = Nxx_calc(x, y, z, deltaX, deltaY, deltaZ, &f);
                        break;
                    case 1: // Nyx
                        //host_grid[cell][kIdx][iIdx][jIdx].x = Nxx_calc(x, y, z, deltaX, deltaY, deltaZ, &g);
                        flat_host_grid[cell * ext_dimZ * ext_dimX * ext_dimY + kIdx * ext_dimX * ext_dimY + iIdx * ext_dimY + jIdx].x = Nxx_calc(x, y, z, deltaX, deltaY, deltaZ, &g);
                        break;
                    case 2: // Nyz
                        //host_grid[cell][kIdx][iIdx][jIdx].x = Nxx_calc(x, z, y, deltaX, deltaZ, deltaY, &g);
                        flat_host_grid[cell * ext_dimZ * ext_dimX * ext_dimY + kIdx * ext_dimX * ext_dimY + iIdx * ext_dimY + jIdx].x = Nxx_calc(x, z, y, deltaX, deltaZ, deltaY, &g);
                        break;
                    case 3: // Nyy
                        //host_grid[cell][kIdx][iIdx][jIdx].x = Nxx_calc(y, z, x, deltaY, deltaZ, deltaX, &f);
                        flat_host_grid[cell * ext_dimZ * ext_dimX * ext_dimY + kIdx * ext_dimX * ext_dimY + iIdx * ext_dimY + jIdx].x = Nxx_calc(y, z, x, deltaY, deltaZ, deltaX, &f);
                        break;
                    case 4: // Nyz
                        //host_grid[cell][kIdx][iIdx][jIdx].x = Nxx_calc(y, z, x, deltaY, deltaZ, deltaX, &g);
                        flat_host_grid[cell * ext_dimZ * ext_dimX * ext_dimY + kIdx * ext_dimX * ext_dimY + iIdx * ext_dimY + jIdx].x = Nxx_calc(y, z, x, deltaY, deltaZ, deltaX, &g);
                        break;
                    case 5: // Nzz
                        //host_grid[cell][kIdx][iIdx][jIdx].x = Nxx_calc(z, x, y, deltaZ, deltaX, deltaY, &f);
                        flat_host_grid[cell * ext_dimZ * ext_dimX * ext_dimY + kIdx * ext_dimX * ext_dimY + iIdx * ext_dimY + jIdx].x = Nxx_calc(z, x, y, deltaZ, deltaX, deltaY, &f);
                        break;
                    default:
                        break;
                    }

                }
            }
        }
    }

    //for (int k = 0; k < ext_dimZ; k++) {
    //    std::cout << k << "\n";
    //    for (int i = 0; i < ext_dimX; i++) {
    //        std::cout << "\t";
    //        for (int j = 0; j < ext_dimY; j++) {
    //            std::cout << "(" << flat_host_grid[k*ext_dimX*ext_dimY + i*ext_dimY + j].x << ", " << flat_host_grid[k * ext_dimX * ext_dimY + i * ext_dimY + j].y << ") ";

    //        }
    //        std::cout << "\n";
    //    }
    //}

    return flat_host_grid;
}

thrust::host_vector<cufftComplex> F_grid_maker(const int& grid_dimX, const int& grid_dimY, const int& grid_dimZ, const float& deltaX, const float& deltaY, const float& deltaZ) {
    const int ext_dimX = 2 * grid_dimX;
    const int ext_dimY = 2 * grid_dimY;
    const int ext_dimZ = 2 * grid_dimZ;

    int error_value;

    cudaSetDevice(0);

    std::cout << "TEMSOR_CALC STARTED\n";

    thrust::host_vector<cufftComplex> flat_host_grid = Tensor_calc(grid_dimX, grid_dimY, grid_dimZ, deltaX, deltaY, deltaZ);

    std::cout << "TENSOR_CALC DONE\n";

    thrust::device_vector<cufftComplex> device_grid(ext_dimZ * ext_dimX * ext_dimY * CELLSIZE);
    thrust::device_vector<cufftComplex> device_output(ext_dimZ * ext_dimX * ext_dimY * CELLSIZE);

    std::cout << "ALLOCATED\n";

    device_grid = flat_host_grid; //thrust::copy(flat_host_grid.begin(), flat_host_grid.end(), device_grid.begin());

    std::cout << "COPIED\n";

     /*flat_host_grid[b][z][x][y] = flast_host_grid[b*ext_dimZ*ext_dimX*ext_dimY + z*ext_dimX*ext_dimY + x*ext_dimY + y]
                                               ->[z*ext_dimX*ext_dimY*CELLSIZE + x*ext_dimY*CELLSIZE + y + b]*/

    int dims[3] = { ext_dimZ, ext_dimX, ext_dimY }; // n

    int inembed[3] = { 1, ext_dimX, ext_dimY };
    int istride = 1;
    int idist = ext_dimZ * ext_dimX * ext_dimY;

    int onembed[3] = { 1, ext_dimX, ext_dimY * CELLSIZE };
    int ostride = 1;
    int odist = ext_dimX * ext_dimY * CELLSIZE;

    cudaDeviceSynchronize();
    std::cout << "STARTED PLAN\n";
    cufftHandle plan;
    error_value = cufftPlanMany(&plan, 3, dims, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, CELLSIZE);

    std::cout << "HERE\n";
    if (error_value != CUFFT_SUCCESS) {
        fprintf(stderr, "CUFFT error: Plan creation failed -- F_grid_maker()");
    }
    std::cout << "PASSED\n";
    //float timerValueGPU, timerValueCPU;
    //cudaEvent_t start, stop;
    //cudaEventCreate(&start);
    //cudaEventCreate(&stop);

    //cudaEventRecord(start, 0);

    error_value = cufftExecC2C(plan, (cufftComplex*)thrust::raw_pointer_cast(device_grid.data()), (cufftComplex*)thrust::raw_pointer_cast(device_output.data()), CUFFT_FORWARD);
    if (error_value != CUFFT_SUCCESS) {
        fprintf(stderr, "CUFFT error: Plan creation failed -- F_grid_maker()");
        return {};
    }
    std::cout << "EXEC DONE\n";
    cudaDeviceSynchronize();
    device_grid.clear(); device_grid.shrink_to_fit();
    device_output.clear(); device_output.shrink_to_fit();
    cufftDestroy(plan);
    return {};
    //cudaEventRecord(stop, 0);
    //cudaEventSynchronize(stop);
    //cudaEventElapsedTime(&timerValueGPU, start, stop);
    //printf("\n GPU calculation time %f msec\n", timerValueGPU);

    //flat_host_grid = device_grid;
    //thrust::host_vector<cufftComplex> func_result(ext_dimZ * ext_dimX * ext_dimY * CELLSIZE);
    //thrust::copy(device_output.begin(), device_output.end(), func_result.begin());
    ////func_result = device_grid;
    //std::cout << "ALMOST DONE\n";
    //cufftDestroy(plan);
    //return {};
}

void TEST_F_grid_maker() {
    int grid_dimX = 2, grid_dimY = 2, grid_dimZ = 1;
    float deltaX = 1, deltaY = 1, deltaZ = 1;
    thrust::host_vector<cufftComplex> F_grid = F_grid_maker(grid_dimX, grid_dimY, grid_dimZ, deltaX, deltaY, deltaZ);
    thrust::device_vector<cufftComplex> test(8 * grid_dimZ * grid_dimX * grid_dimY);
    for (int k = 0; k < 2 * grid_dimZ; k++) {
        std::cout << k << "\n";
        for (int i = 0; i < 2 * grid_dimX; i++) {
            std::cout << "\t";
            for (int j = 0; j < 2 * grid_dimY; j++) {
                test[k * 4 * grid_dimX * grid_dimY + i * 2 * grid_dimY + j] = F_grid[k * 2 * grid_dimX * 2 * grid_dimY * 6 + i * 2 * grid_dimY * 6 + j + 0];
                std::cout << "(" << F_grid[k * 2*grid_dimX * 2*grid_dimY * 6 + i * 2*grid_dimY*6 + j + 0].x << ", " << F_grid[k * 2 * grid_dimX * 2 * grid_dimY * 6 + i * 2 * grid_dimY * 6 + j + 0].y << ") ";

            }
            std::cout << "\n";
        }
    }

    cufftHandle plan;
    cufftPlan3d(&plan, 2 * grid_dimZ, 2 * grid_dimX, 2 * grid_dimY, CUFFT_C2C);
    cufftExecC2C(plan, (cufftComplex*)thrust::raw_pointer_cast(test.data()), (cufftComplex*)thrust::raw_pointer_cast(test.data()), CUFFT_INVERSE);

    thrust::host_vector<cufftComplex> dummy = test;

    for (int k = 0; k < 2 * grid_dimZ; k++) {
        std::cout << k << "\n";
        for (int i = 0; i < 2 * grid_dimX; i++) {
            std::cout << "\t";
            for (int j = 0; j < 2 * grid_dimY; j++) {
                std::cout << "(" << dummy[k * 4 * grid_dimX * grid_dimY + i * 2 * grid_dimY + j].x / (32) << ", " << dummy[k * 4 * grid_dimX * grid_dimY + i * 2 * grid_dimY + j].y / (32) << ") ";
            }
            std::cout << "\n";
        }
    }


 /*   0 -- AFTER FFT
        (1.79917, 0) (0.795436, 2.5034e-06) (-0.208294, 0) (0.795436, -2.5034e-06)
        (5.88547, 2.23517e-06) (4.18879, 2.38419e-06) (2.49211, 6.25849e-07) (4.18879, 4.76837e-07)
        (9.97177, 0) (7.58214, -5.96046e-07) (5.19252, 0) (7.58214, 5.96046e-07)
        (5.88547, -2.23517e-06) (4.18879, -4.76837e-07) (2.49211, -6.25849e-07) (4.18879, -2.38419e-06)
    1
        (1.79917, 0) (0.795436, 2.5034e-06) (-0.208294, 0) (0.795436, -2.5034e-06)
        (5.88547, 2.23517e-06) (4.18879, 2.38419e-06) (2.49211, 6.25849e-07) (4.18879, 4.76837e-07)
        (9.97177, 0) (7.58214, -5.96046e-07) (5.19252, 0) (7.58214, 5.96046e-07)
        (5.88547, -2.23517e-06) (4.18879, -4.76837e-07) (2.49211, -6.25849e-07) (4.18879, -2.38419e-06)*/

    //0 AFTER INVERSE FFT
    //    (4.18879, 0) (0.848339, 0) (0, 0) (0.84834, 0)
    //    (-1.69668, 0) (-0.173238, 0) (0, 0) (-0.173237, 0)
    //    (0, 0) (0, 0) (0, 0) (0, 0)
    //    (-1.69668, 0) (-0.173237, 0) (0, 0) (-0.173237, 0)
    //    1
    //    (0, 0) (0, 0) (0, 0) (0, 0)
    //    (0, 0) (0, 0) (0, 0) (0, 0)
    //    (0, 0) (0, 0) (0, 0) (0, 0)
    //    (0, 0) (0, 0) (0, 0) (0, 0)
}

//-------------------------------------------------------------------- EXCHANGE FIELD

__global__ void exch_field_calc(float* exch_field, int grid_dimX, int grid_dimY, int grid_dimZ, float deltaX, float deltaY, float deltaZ) {

}

//static __device__ __host__ inline float3 addfloat3(float3 a, float3 b) {
//    float3 res;
//    res.x = a.x + b.x;
//    res.y = a.y + b.y;
//    res.z = a.z + b.z;
//    return res;
//}
//
//__global__ void add(float3* dA, float3* dB, float3* doutput) {
//    int idX = threadIdx.x + blockIdx.x * blockDim.x;
//    int idY = threadIdx.y + blockIdx.y * blockDim.y;
//    int idZ = threadIdx.z + blockIdx.z * blockDim.z;
//
//    doutput[idZ * 9 + idY * 3 + idX] = addfloat3(dA[idZ * 9 + idY * 3 + idX], dB[idZ * 9 + idY * 3 + idX]);
//    //doutput[idZ * 9 + idY * 3 + idX].y = dA[idZ * 9 + idY * 3 + idX].y + dB[idZ * 9 + idY * 3 + idX].y;
//    //doutput[idZ * 9 + idY * 3 + idX].z = dA[idZ * 9 + idY * 3 + idX].z + dB[idZ * 9 + idY * 3 + idX].z;
//}


static __device__ __host__ inline cufftComplex ComplexAdd(cufftComplex, cufftComplex);
static __device__ __host__ inline cufftComplex ComplexScale(cufftComplex, float);
static __device__ __host__ inline cufftComplex ComplexMul(cufftComplex, cufftComplex);
static __global__ void ComplexPointwiseMulAndScale(cufftComplex*, const cufftComplex*, int, float);

static __global__ void Convolution_calc(cufftComplex*, cufftComplex* );

int main() {

    int grid_dimX, grid_dimY, grid_dimZ; // actual number of cells is 

    grid_dimX = 128, grid_dimY = 128, grid_dimZ = 1;

    float deltaX, deltaY, deltaZ;

    deltaX = 1, deltaY = 1, deltaZ = 1;

    grid_dimX = 4096;
    grid_dimY = 128;
    grid_dimZ = 3;

    const long long int ext_dimX = 2LL * grid_dimX;
    const long long int ext_dimY = 2LL * grid_dimY;
    const long long int ext_dimZ = 2LL * grid_dimZ;
    const long long int FULLSIZE = ext_dimX * ext_dimY * ext_dimZ * 6LL;

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

    thrust::host_vector<cufftComplex> mag_grid(ext_dimZ*ext_dimY*ext_dimX*3);


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
    thrust::device_vector<cufftComplex>  F_mag_grid = mag_grid;
 


    cufftHandle plan_adv, plan;
    size_t workSize;
    cufftCreate(&plan_adv);
    cufftXtMakePlanMany(plan_adv, 3, n, inembed, istride, idist, CUDA_C_32F, inembed, istride, idist, CUDA_C_32F, 6, &workSize, CUDA_C_32F);
    printf("Temporary buffer size %li bytes\n", workSize);

    //error_value = cufftPlanMany(&plan, 3, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, CELLSIZE);

    //std::cout << "HERE\n";
    //if (error_value != CUFFT_SUCCESS) {
    //    fprintf(stderr, "CUFFT error: Plan creation failed -- F_grid_maker()");
    //}
    //std::cout << "PASSED\n";

    cudaEventRecord(start, 0);

    error_value = cufftExecC2C(plan_adv, (cufftComplex*)thrust::raw_pointer_cast(input_device_grid.data()), (cufftComplex*)thrust::raw_pointer_cast(input_device_grid.data()), CUFFT_FORWARD); // ~20 ms

    //host_grid = input_device_grid; // ~80 ms

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timerValueGPU, start, stop);
    printf("\n GPU calculation time %f msec\n", timerValueGPU);

    dim3 blockSize(16, 16); //256
    dim3 gridSize(ext_dimX / 16, ext_dimY / 16, ext_dimZ);

    return 0;
}

////////////////////////////////////////////////////////////////////////////////
// Complex operations
////////////////////////////////////////////////////////////////////////////////

// Complex addition
static __device__ __host__ inline cufftComplex ComplexAdd(cufftComplex a, cufftComplex b) {
    cufftComplex c;
    c.x = a.x + b.x;
    c.y = a.y + b.y;
    return c;
}

// Complex scale
static __device__ __host__ inline cufftComplex ComplexScale(cufftComplex a, float s) {
    cufftComplex c;
    c.x = s * a.x;
    c.y = s * a.y;
    return c;
}

// Complex multiplication
static __device__ __host__ inline cufftComplex ComplexMul(cufftComplex a, cufftComplex b) {
    cufftComplex c;
    c.x = a.x * b.x - a.y * b.y;
    c.y = a.x * b.y + a.y * b.x;
    return c;
}
// Complex pointwise multiplication
static __global__ void ComplexPointwiseMulAndScale(cufftComplex* a, const cufftComplex* b, int size, float scale) {
    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = threadID; i < size; i += numThreads) {
        a[i] = ComplexScale(ComplexMul(a[i], b[i]), scale);
    }
}


