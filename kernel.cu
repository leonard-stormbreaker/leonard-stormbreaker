
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

#include "test_runner.h"

#define CELLSIZE 6

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

void fill_magnetic_grid(Tensor& mag_grid, int gridDimX, int gridDimY, int gridDimZ, float deltaX, float deltaY, float deltaZ) {

    for (int cell = 0; cell < CELLSIZE; cell++) {
        for (int k = -gridDimZ + 1; k < gridDimZ; k++) {
            for (int i = -gridDimX + 1; i < gridDimX; i++) {
                for (int j = -gridDimY + 1; j < gridDimY; j++) {

                    int kIdx = (k >= 0) ? k : 2 * gridDimZ - 1 - std::abs(k);
                    int iIdx = (i >= 0) ? i : 2 * gridDimX - 1 - std::abs(i);
                    int jIdx = (j >= 0) ? j : 2 * gridDimY - 1 - std::abs(j);

                    float x = i * deltaX;
                    float y = j * deltaY;
                    float z = k * deltaZ;

                    switch (cell) { // Nxx - 0, Nxy - 1, Nxz - 2, Nyy - 3, Nyz - 4, Nzz = 5
                    case 0: // Nxx
                        mag_grid[cell][kIdx][iIdx][jIdx] = Nxx_calc(x, y, z, deltaX, deltaY, deltaZ, &f);
                        break;
                    case 1: // Nyx
                        mag_grid[cell][kIdx][iIdx][jIdx] = Nxx_calc(x, y, z, deltaX, deltaY, deltaZ, &g);
                        break;
                    case 2: // Nyz
                        mag_grid[cell][kIdx][iIdx][jIdx] = Nxx_calc(x, z, y, deltaX, deltaZ, deltaY, &g);
                        break;
                    case 3: // Nyy
                        mag_grid[cell][kIdx][iIdx][jIdx] = Nxx_calc(y, z, x, deltaY, deltaZ, deltaX, &f);
                        break;
                    case 4: // Nyz
                        mag_grid[cell][kIdx][iIdx][jIdx] = Nxx_calc(y, z, x, deltaY, deltaZ, deltaX, &g);
                        break;
                    case 5: // Nzz
                        mag_grid[cell][kIdx][iIdx][jIdx] = Nxx_calc(z, x, y, deltaZ, deltaX, deltaY, &f);
                        break;
                    default:
                        break;
                    }

                }
            }
        }
    }
    return;
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

thrust::host_vector<cufftComplex> F_grid_maker(const int& grid_dimX, const int& grid_dimY, const int& grid_dimZ, const float& deltaX, const float& deltaY, const float& deltaZ) {
    const int ext_dimX = 2 * grid_dimX;
    const int ext_dimY = 2 * grid_dimY;
    const int ext_dimZ = 2 * grid_dimZ;

    thrust::host_vector<thrust::host_vector<thrust::host_vector<thrust::host_vector<cufftComplex>>>> host_grid(6,
        thrust::host_vector<thrust::host_vector<thrust::host_vector<cufftComplex>>>(ext_dimZ, thrust::host_vector<thrust::host_vector<cufftComplex>>(ext_dimX,
            thrust::host_vector<cufftComplex>(ext_dimY))));
    for (int cell = 0; cell < CELLSIZE; cell++) {
        for (int k = -grid_dimZ + 1; k < grid_dimZ; k++) {
            for (int i = -grid_dimX + 1; i < grid_dimX; i++) {
                for (int j = -grid_dimY + 1; j < grid_dimY; j++) {

                    int kIdx = (k >= 0) ? k : 2 * grid_dimZ - 1 - std::abs(k);
                    int iIdx = (i >= 0) ? i : 2 * grid_dimX - 1 - std::abs(i);
                    int jIdx = (j >= 0) ? j : 2 * grid_dimY - 1 - std::abs(j);

                    float x = i * deltaX;
                    float y = j * deltaY;
                    float z = k * deltaZ;


                    switch (cell) { // Nxx - 0, Nxy - 1, Nxz - 2, Nyy - 3, Nyz - 4, Nzz = 5
                    case 0: // Nxx
                        host_grid[cell][kIdx][iIdx][jIdx].x = Nxx_calc(x, y, z, deltaX, deltaY, deltaZ, &f);
                        break;
                    case 1: // Nyx
                        host_grid[cell][kIdx][iIdx][jIdx].x = Nxx_calc(x, y, z, deltaX, deltaY, deltaZ, &g);
                        break;
                    case 2: // Nyz
                        host_grid[cell][kIdx][iIdx][jIdx].x = Nxx_calc(x, z, y, deltaX, deltaZ, deltaY, &g);
                        break;
                    case 3: // Nyy
                        host_grid[cell][kIdx][iIdx][jIdx].x = Nxx_calc(y, z, x, deltaY, deltaZ, deltaX, &f);
                        break;
                    case 4: // Nyz
                        host_grid[cell][kIdx][iIdx][jIdx].x = Nxx_calc(y, z, x, deltaY, deltaZ, deltaX, &g);
                        break;
                    case 5: // Nzz
                        host_grid[cell][kIdx][iIdx][jIdx].x = Nxx_calc(z, x, y, deltaZ, deltaX, deltaY, &f);
                        break;
                    default:
                        break;
                    }

                }
            }
        }
    }


    cufftHandle plan;
    cufftPlan3d(&plan, ext_dimZ, ext_dimX, ext_dimY, CUFFT_C2C);

    thrust::device_vector<cufftComplex> device_grid(ext_dimZ * ext_dimX * ext_dimY);
    for (int k = 0; k < ext_dimZ; k++) {
        for (int i = 0; i < ext_dimX; i++) {
            for (int j = 0; j < ext_dimY; j++) {
                device_grid[k * ext_dimX * ext_dimY + i * ext_dimY + j] = host_grid[0][k][i][j];
            }
        }
    }

    if (cufftExecC2C(plan, (cufftComplex*)thrust::raw_pointer_cast(device_grid.data()), (cufftComplex*)thrust::raw_pointer_cast(device_grid.data()), CUFFT_FORWARD) != CUFFT_SUCCESS) {
        fprintf(stderr, "CUFFT error: ExecC2C Forward failed -- F_grid_maker()");
    }

    thrust::host_vector<cufftComplex> F_host_grid(ext_dimZ*ext_dimX*ext_dimY);
    thrust::copy(device_grid.begin(), device_grid.end(), F_host_grid.begin());


    cufftDestroy(plan);
    device_grid.clear();
    host_grid.clear();
    flat_host_printer_cufft(F_host_grid, ext_dimZ, ext_dimX, ext_dimY);

    return F_host_grid;
}

int main() {

    //gpu_params();

    int grid_dimX, grid_dimY, grid_dimZ; // actual number of cells is 

    grid_dimX = 2, grid_dimY = 2, grid_dimZ = 1;

    float deltaX, deltaY, deltaZ;

    deltaX = 1, deltaY = 1, deltaZ = 1;

    thrust::host_vector<cufftComplex> F_grid = F_grid_maker(grid_dimX, grid_dimY, grid_dimZ, deltaX, deltaY, deltaZ);
    flat_host_printer_cufft(F_grid, 2 * grid_dimZ, 2 * grid_dimX, 2 * grid_dimY);

    return 0;

}
