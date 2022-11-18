//
// Created by devilox on 11/2/21.
//
//-----------------------------//
#include <iomanip>
#include "DemagTensor.h"
//-----------------------------//
DemagTensor::DemagTensor(uint32_t tDimX, uint32_t tDimY, uint32_t tDimZ, double tDeltaX, double tDeltaY) {
    mDimX           = tDimX;
    mDimY           = tDimY;
    mDimZ           = tDimZ;

    mExtendedDimX   = tDimX * 2;
    mExtendedDimY   = tDimY * 2;
    mExtendedDimZ   = tDimZ * tDimZ;

    mDeltaX         = tDeltaX;
    mDeltaY         = tDeltaY;

    //----------//

    mExtendedTensor.resize(mExtendedDimZ);
    mExtendedTensorImage.resize(mExtendedDimZ);
    mTestImage.resize(mExtendedDimZ);
    mTestImage.resize(mExtendedDimZ);

    for (int iLayer = 0; iLayer < mExtendedDimZ; iLayer++) {
        mExtendedTensor[iLayer].resize(mExtendedDimX);
        mExtendedTensorImage[iLayer].resize(mExtendedDimX);
        mTestImage[iLayer].resize(mExtendedDimX);

        for (int ix = 0; ix < mExtendedDimX; ix++) {
            mExtendedTensor[iLayer][ix].resize(mExtendedDimY);
            mExtendedTensorImage[iLayer][ix].resize(mExtendedDimY / 2 + 1);
            mTestImage[iLayer][ix].resize(mExtendedDimY);
        }
    }
}
//-----------------------------//
void DemagTensor::calcTensor(const std::vector <double>& tDeltaZ, const std::vector <std::vector <std::vector <int8_t>>>& tMask) {
    ///---TODO: check for size mismatch---///

    int StartX = -static_cast <int> (mDimX);
    int StartY = -static_cast <int> (mDimY);

    int StopX = static_cast <int> (mDimX);
    int StopY = static_cast <int> (mDimY);

    double NormDeltaZ = -1.0 / (mDeltaX * mDeltaY * 4.0 * M_PI);

#pragma omp parallel for default(none) shared(StartX, StartY, StopX, StopY, tMask, tDeltaZ, NormDeltaZ) num_threads(4)
    for (int kDest = 0; kDest < mDimZ; kDest++) {
        for (int i = StartX; i < StopX; i++) {
            for (int j = StartY; j < StopY; j++) {
                int IndexX = (i < 0) ? i + StopX * 2 : i;
                int IndexY = (j < 0) ? j + StopY * 2 : j;

                for (int kSource = 0; kSource < mDimZ; kSource++) {
                    double x = i * mDeltaX;
                    double y = j * mDeltaY;
                    double z = 0;

                    if (kSource > kDest) {
                        for (int iDelta = kSource; iDelta > kDest; iDelta--) {
                            z -= tDeltaZ[iDelta];
                        }
                    } else {
                        for (int iDelta = kSource; iDelta < kDest; iDelta++) {
                            z += tDeltaZ[iDelta];
                        }
                    }

                    double NxxVal = Nxx(x, y, z, mDeltaX, mDeltaX, mDeltaY, mDeltaY, tDeltaZ[kDest], tDeltaZ[kSource]);
                    double NyyVal = Nyy(x, y, z, mDeltaX, mDeltaX, mDeltaY, mDeltaY, tDeltaZ[kDest], tDeltaZ[kSource]);
                    double NzzVal = Nzz(x, y, z, mDeltaX, mDeltaX, mDeltaY, mDeltaY, tDeltaZ[kDest], tDeltaZ[kSource]);
                    double NxyVal = Nxy(x, y, z, mDeltaX, mDeltaX, mDeltaY, mDeltaY, tDeltaZ[kDest], tDeltaZ[kSource]);
                    double NxzVal = Nxz(x, y, z, mDeltaX, mDeltaX, mDeltaY, mDeltaY, tDeltaZ[kDest], tDeltaZ[kSource]);
                    double NyzVal = Nyz(x, y, z, mDeltaX, mDeltaX, mDeltaY, mDeltaY, tDeltaZ[kDest], tDeltaZ[kSource]);

                    double Norm = NormDeltaZ / tDeltaZ[kDest];
                    size_t NewZ = kSource * mDimZ + kDest;

                    mExtendedTensor[NewZ][IndexX][IndexY][0] = NxxVal * Norm;
                    mExtendedTensor[NewZ][IndexX][IndexY][4] = NyyVal * Norm;
                    mExtendedTensor[NewZ][IndexX][IndexY][8] = NzzVal * Norm;

                    mExtendedTensor[NewZ][IndexX][IndexY][1] = NxyVal * Norm;
                    mExtendedTensor[NewZ][IndexX][IndexY][2] = NxzVal * Norm;
                    mExtendedTensor[NewZ][IndexX][IndexY][3] = NxyVal * Norm;
                    mExtendedTensor[NewZ][IndexX][IndexY][5] = NyzVal * Norm;
                    mExtendedTensor[NewZ][IndexX][IndexY][6] = NxzVal * Norm;
                    mExtendedTensor[NewZ][IndexX][IndexY][7] = NyzVal * Norm;
                }
            }
        }
    }
}
void DemagTensor::calcImage() {
    ///---TODO: check if the tensor was calculated---///

    auto In     = fftw_alloc_real(mExtendedDimX * mExtendedDimY);
    auto Out    = fftw_alloc_complex(mExtendedDimX * mExtendedDimY);

    uint32_t ImageDimY = mExtendedDimY / 2 + 1;

    for (int iTensor = 0; iTensor < 9; iTensor++) {
        for (int iz = 0; iz < mDimZ * mDimZ; iz++) {
            for (int i = 0; i < mExtendedDimX; i++) {
                for (int j = 0; j < mExtendedDimY; j++) {
                    In[i * mExtendedDimY + j] = mExtendedTensor[iz][i][j][iTensor];
                }
            }

            auto Plan = fftw_plan_dft_r2c_2d(
                    static_cast <int>(mExtendedDimX),
                    static_cast <int>(mExtendedDimY),
                    In,
                    Out,
                    FFTW_ESTIMATE);
            fftw_execute(Plan);
            fftw_destroy_plan(Plan);

            //----------//

            for (int i = 0; i < mExtendedDimX; i++) {
                for (int j = 0; j < ImageDimY; j++) {
                    mExtendedTensorImage[iz][i][j][iTensor].real(Out[i * ImageDimY + j][0]);
                    mExtendedTensorImage[iz][i][j][iTensor].imag(Out[i * ImageDimY + j][1]);

                    mTestImage[iz][i][j][iTensor][0] = Out[i * ImageDimY + j][0];
                    mTestImage[iz][i][j][iTensor][1] = Out[i * ImageDimY + j][1];
                }
            }
        }
    }

    //----------//

    fftw_free(In);
    fftw_free(Out);
}

const DemagTensor::TensorGrid& DemagTensor::getTensor() const {
    return mExtendedTensor;
}
const DemagTensor::ImageGrid& DemagTensor::getImage() const {
    return mExtendedTensorImage;
}
//-----------------------------//
double DemagTensor::fxx(double x, double y, double z) {
    double x2 = std::pow(x, 2.0);
    double y2 = std::pow(y, 2.0);
    double z2 = std::pow(z, 2.0);

    double R = std::sqrt(std::pow(x, 2.0) + std::pow(y, 2.0) + std::pow(z, 2.0));

    double A = (x == 0 && z == 0) ? 0 : y / 2.0 * (z2 - x2) * std::asinh(y / std::sqrt(x2 + z2));
    double B = (x == 0 && y == 0) ? 0 : z / 2.0 * (y2 - x2) * std::asinh(z / std::sqrt(x2 + y2));
    double C = (x == 0) ? 0 : -x * y * z * std::atan((y * z) / (x * R));
    double D = 1.0 / 6.0 * (2.0 * x2 - y2 - z2) * R;

    return A + B + C + D;
}
double DemagTensor::F1xx(double x, double y, double z, double tDeltaY, double tDeltaYSource) {
    return  fxx(x, y + tDeltaY, z) -
            fxx(x, y, z) -
            fxx(x, y - tDeltaYSource + tDeltaY, z) +
            fxx(x, y - tDeltaYSource, z);
}
double DemagTensor::Fxx(double x, double y, double z, double tDeltaY, double tDeltaYSource, double tDeltaZ, double tDeltaZSource) {
    return  F1xx(x, y, z + tDeltaZ, tDeltaY, tDeltaYSource) -
            F1xx(x, y, z, tDeltaY, tDeltaYSource) -
            F1xx(x, y, z + tDeltaZ - tDeltaZSource, tDeltaY, tDeltaYSource) +
            F1xx(x, y, z - tDeltaZSource, tDeltaY, tDeltaYSource);
}
double DemagTensor::Nxx(double x, double y, double z, double tDeltaX, double tDeltaXSource, double tDeltaY, double tDeltaYSource, double tDeltaZ, double tDeltaZSource) {
    return  Fxx(x, y, z, tDeltaY, tDeltaYSource, tDeltaZ, tDeltaZSource) -
            Fxx(x + tDeltaX, y, z, tDeltaY, tDeltaYSource, tDeltaZ, tDeltaZSource) -
            Fxx(x - tDeltaXSource, y, z, tDeltaY, tDeltaYSource, tDeltaZ, tDeltaZSource) +
            Fxx(x + tDeltaX - tDeltaXSource, y, z, tDeltaY, tDeltaYSource, tDeltaZ, tDeltaZSource);
}
//-----------------------------//
double DemagTensor::Nyy(double x, double y, double z, double tDeltaX, double tDeltaXSource, double tDeltaY, double tDeltaYSource, double tDeltaZ, double tDeltaZSource) {
    return Nxx(y, z, x, tDeltaY, tDeltaYSource, tDeltaZ, tDeltaZSource, tDeltaX, tDeltaXSource);
}
//-----------------------------//
double DemagTensor::Nzz(double x, double y, double z, double tDeltaX, double tDeltaXSource, double tDeltaY, double tDeltaYSource, double tDeltaZ, double tDeltaZSource) {
    return Nxx(z, x, y, tDeltaZ, tDeltaZSource, tDeltaX, tDeltaXSource, tDeltaY, tDeltaYSource);
}
//-----------------------------//
double DemagTensor::gxy(double x, double y, double z) {
    double x2 = std::pow(x, 2.0);
    double y2 = std::pow(y, 2.0);
    double z2 = std::pow(z, 2.0);

    double R = std::sqrt(std::pow(x, 2.0) + std::pow(y, 2.0) + std::pow(z, 2.0));

    double A = (x == 0 && y == 0) ? 0 : x * y * z * std::asinh(z / std::sqrt(x2 + y2));
    double B = (y == 0 && z == 0) ? 0 : y / 6 * (3.0 * z2 - y2) * std::asinh(x / std::sqrt(y2 + z2));
    double C = (x == 0 && z == 0) ? 0 : x / 6 * (3.0 * z2 - x2) * std::asinh(y / std::sqrt(x2 + z2));
    double D = (z == 0) ? 0 : std::pow(z, 3.0) / 6 * std::atan((x * y) / (z * R));
    double E = (y == 0) ? 0 : z * y2 / 2 * std::atan((x * z) / (y * R));
    double F = (x == 0) ? 0 : z * x2 / 2 * std::atan((y * z) / (x * R));
    double G = x * y * R / 3.0;

    return A + B + C - D - E - F - G;
}
double DemagTensor::G1xy(double x, double y, double z, double deltaZ, double deltaZSource) {
    return  gxy(x, y, z + deltaZ) -
            gxy(x, y, z) -
            gxy(x, y, z + deltaZ - deltaZSource) +
            gxy(x, y, z - deltaZSource);
}
double DemagTensor::Gxy(double x, double y, double z, double tDeltaX, double tDeltaYSource, double deltaZ, double deltaZSource) {
    return  G1xy(x + tDeltaX, y, z, deltaZ, deltaZSource) -
            G1xy(x + tDeltaX, y - tDeltaYSource, z, deltaZ, deltaZSource) -
            G1xy(x, y, z, deltaZ, deltaZSource) +
            G1xy(x, y - tDeltaYSource, z, deltaZ, deltaZSource);
}
double DemagTensor::Nxy(double x, double y, double z, double tDeltaX, double tDeltaXSource, double tDeltaY, double tDeltaYSource, double deltaZ, double deltaZSource) {
    return  Gxy(x, y, z, tDeltaX, tDeltaYSource, deltaZ, deltaZSource) -
            Gxy(x - tDeltaXSource, y, z, tDeltaX, tDeltaYSource, deltaZ, deltaZSource) -
            Gxy(x, y + tDeltaY, z, tDeltaX, tDeltaYSource, deltaZ, deltaZSource) +
            Gxy(x - tDeltaXSource, y + tDeltaY, z, tDeltaX, tDeltaYSource, deltaZ, deltaZSource);
}
//-----------------------------//
double DemagTensor::Nxz(double x, double y, double z, double tDeltaX, double tDeltaXSource, double tDeltaY, double tDeltaYSource, double deltaZ, double deltaZSource) {
    return Nxy(x, z, y, tDeltaX, tDeltaXSource, deltaZ, deltaZSource, tDeltaY, tDeltaYSource);
}
//-----------------------------//
double DemagTensor::Nyz(double x, double y, double z, double tDeltaX, double tDeltaXSource, double tDeltaY, double tDeltaYSource, double deltaZ, double deltaZSource) {
    return Nxy(y, z, x, tDeltaY, tDeltaYSource, deltaZ, deltaZSource, tDeltaX, tDeltaXSource);
}