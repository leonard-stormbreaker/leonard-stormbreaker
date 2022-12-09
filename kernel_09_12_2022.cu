
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cufft.h>
#include <cufftXt.h>

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
void flat_device_printer(thrust::device_vector<thrust::complex<float>> grid, int dimz, int dimx, int dimy) {
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

    thrust::host_vector<cufftComplex> flat_host_grid = Tensor_calc(grid_dimX, grid_dimY, grid_dimZ, deltaX, deltaY, deltaZ);

    thrust::device_vector<cufftComplex> device_grid(ext_dimZ * ext_dimX * ext_dimY * CELLSIZE);

    device_grid = flat_host_grid; //thrust::copy(flat_host_grid.begin(), flat_host_grid.end(), device_grid.begin());

     /*flat_host_grid[b][z][x][y] = flast_host_grid[b*ext_dimZ*ext_dimX*ext_dimY + z*ext_dimX*ext_dimY + x*ext_dimY + y]
                                               ->[z*ext_dimX*ext_dimY*CELLSIZE + x*ext_dimY*CELLSIZE + y + b]*/

    int dims[3] = { ext_dimZ, ext_dimX, ext_dimY }; // n

    int inembed[3] = { 1, ext_dimX, ext_dimY };
    int istride = 1;
    int idist = ext_dimZ * ext_dimX * ext_dimY;

    int onembed[3] = { 1, ext_dimX, ext_dimY * CELLSIZE };
    int ostride = 1;
    int odist = ext_dimX * ext_dimY * CELLSIZE;

    cufftHandle plan;
    error_value = cufftPlanMany(&plan, 3, dims, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, CELLSIZE);
    if (error_value != CUFFT_SUCCESS) {
        fprintf(stderr, "CUFFT error: Plan creation failed -- F_grid_maker()");
    }

    error_value = cufftExecC2C(plan, (cufftComplex*)thrust::raw_pointer_cast(device_grid.data()), (cufftComplex*)thrust::raw_pointer_cast(device_grid.data()), CUFFT_FORWARD);
    if (error_value != CUFFT_SUCCESS) {
        fprintf(stderr, "CUFFT error: Plan creation failed -- F_grid_maker()");
    }

    flat_host_grid = device_grid;

    return flat_host_grid;
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


int main() {

    //gpu_params();

    int grid_dimX, grid_dimY, grid_dimZ; // actual number of cells is 

    grid_dimX = 2, grid_dimY = 2, grid_dimZ = 1;

    float deltaX, deltaY, deltaZ;

    deltaX = 1, deltaY = 1, deltaZ = 1;

    //TEST_F_grid_maker();

    //--------------- EXCHANGE FIELD

    thrust::host_vector<float> mag_grid(grid_dimZ * grid_dimX * grid_dimY*VECDIM);

    thrust::device_vector<float> exch_field(grid_dimZ * grid_dimX * grid_dimY * VECDIM);

    //cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    //cudaArray_t cuArray;
    ////cudaMallocArray(&cuArray, &channelDesc, width, height);
    cudaTextureObject_t tex_ref;


    

    




    return 0;

}
