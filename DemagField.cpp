//
// Created by devilox on 11/20/21.
//
//-----------------------------//
#include "DemagField.h"
//-----------------------------//
DemagField::DemagField(uint32_t tDimX, uint32_t tDimY, uint32_t tDimZ) {
    mDimX           = tDimX;
    mDimY           = tDimY;
    mDimZ           = tDimZ;

    mExtendedDimX   = tDimX * 2;
    mExtendedDimY   = tDimY * 2;

    //----------//

    mExtendedMag.resize(tDimZ);
    mExtendedMagImage.resize(tDimZ);
    mResult.resize(tDimZ);

    mTestMult.resize(tDimZ * tDimZ);
    mTestMultInv.resize(tDimZ * tDimZ);

    for (int i = 0; i < tDimZ; i++) {
        mExtendedMag[i].resize(mExtendedDimX);
        mExtendedMagImage[i].resize(mExtendedDimX);
        mResult[i].resize(mExtendedDimX);

        for (int j = 0; j < mExtendedDimX; j++) {
            mExtendedMag[i][j].resize(mExtendedDimY, {0.0, 0.0, 0.0});
            mExtendedMagImage[i][j].resize(mExtendedDimY / 2 + 1);
            mResult[i][j].resize(mExtendedDimY, {0.0, 0.0, 0.0});
        }
    }

    for (int i = 0; i < tDimZ * tDimZ; i++) {
        mTestMult[i].resize(mExtendedDimX);
        mTestMultInv[i].resize(mExtendedDimX);

        for (int j = 0; j < mExtendedDimX; j++) {
            mTestMult[i][j].resize(mExtendedDimY);
            mTestMultInv[i][j].resize(mExtendedDimY);
        }
    }
}
//-----------------------------//
void DemagField::calcField(const DemagField::MagGrid& tMag, const DemagField::ImageTensorGrid& tDemagTensorImage) {
    for (int iz = 0; iz < mDimZ; iz++) {
        for (int i = 0; i < mDimX; i++) {
            for (int j = 0; j < mDimY; j++) {
                mExtendedMag[iz][i][j] = tMag[iz][i][j];
            }
        }
    }

    //----------//

    auto In     = fftw_alloc_real(mExtendedDimX * mExtendedDimY);
    auto Out    = fftw_alloc_complex(mExtendedDimX * mExtendedDimY);

    uint32_t ImageDimY = mExtendedDimY / 2 + 1;

    for (int iDim = 0; iDim < 3; iDim++) {
        for (int iz = 0; iz < mDimZ; iz++) {
            for (int i = 0; i < mExtendedDimX; i++) {
                for (int j = 0; j < mExtendedDimY; j++) {
                    In[i * mExtendedDimY + j] = mExtendedMag[iz][i][j][iDim];
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
                    mExtendedMagImage[iz][i][j][iDim].real(Out[i * ImageDimY + j][0]);
                    mExtendedMagImage[iz][i][j][iDim].imag(Out[i * ImageDimY + j][1]);
                }
            }
        }
    }

//    for (int iDim = 0; iDim < 3; iDim++) {
//        for (int k = 0; k < mDimZ; k++) {
//            for (int i = 0; i < mDimX * 2; i++) {
//                for (int j = 0; j < mDimY + 1; j++) {
//                    std::cout << mExtendedMagImage[k][i][j][iDim].imag() << "\t";
//                }
//                std::cout << std::endl;
//            }
//            std::cout << std::endl;
//        }
//        std::cout << std::endl;
//    }
//    std::cout << std::endl;

    //----------//

    for (int kSource = 0; kSource < mDimZ; kSource++) {
        for (int kDest = 0; kDest < mDimZ; kDest++) {
            for (int i = 0; i < mExtendedDimX ; i++) {
                for (int j = 0; j < ImageDimY; j++) {
                    size_t zNew = kSource * mDimZ + kDest;

                    mTestMult[zNew][i][j][0] =
                            tDemagTensorImage[zNew][i][j][0] * mExtendedMagImage[kDest][i][j][0] +
                            tDemagTensorImage[zNew][i][j][1] * mExtendedMagImage[kDest][i][j][1] +
                            tDemagTensorImage[zNew][i][j][2] * mExtendedMagImage[kDest][i][j][2];
                    mTestMult[zNew][i][j][1] =
                            tDemagTensorImage[zNew][i][j][3] * mExtendedMagImage[kDest][i][j][0] +
                            tDemagTensorImage[zNew][i][j][4] * mExtendedMagImage[kDest][i][j][1] +
                            tDemagTensorImage[zNew][i][j][5] * mExtendedMagImage[kDest][i][j][2];
                    mTestMult[zNew][i][j][2] =
                            tDemagTensorImage[zNew][i][j][6] * mExtendedMagImage[kDest][i][j][0] +
                            tDemagTensorImage[zNew][i][j][7] * mExtendedMagImage[kDest][i][j][1] +
                            tDemagTensorImage[zNew][i][j][8] * mExtendedMagImage[kDest][i][j][2];
                }
            }
        }
    }

    auto In2    = fftw_alloc_complex(mExtendedDimX * ImageDimY);
    auto Out2   = fftw_alloc_real(mExtendedDimX * mExtendedDimY);

    for (int iDim = 0; iDim < 3; iDim++) {
        for (int iz = 0; iz < mDimZ * mDimZ; iz++) {
            for (int i = 0; i < mExtendedDimX; i++) {
                for (int j = 0; j < ImageDimY; j++) {
                    In2[i * ImageDimY + j][0] = mTestMult[iz][i][j][iDim].real();
                    In2[i * ImageDimY + j][1] = mTestMult[iz][i][j][iDim].imag();
                }
            }

            auto Plan = fftw_plan_dft_c2r_2d(
                    static_cast <int>(mExtendedDimX),
                    static_cast <int>(mExtendedDimY),
                    In2,
                    Out2,
                    FFTW_ESTIMATE);
            fftw_execute(Plan);
            fftw_destroy_plan(Plan);

            //----------//

            for (int i = 0; i < mExtendedDimX; i++) {
                for (int j = 0; j < mExtendedDimY; j++) {
                    mTestMultInv[iz][i][j][iDim].real(Out2[i * mExtendedDimY + j]);
                }
            }
        }
    }

    double Norm = 1.0 / (mExtendedDimX * mExtendedDimY);
//    double Norm = 1.0 / (4 * 196);

    for (int i = 0; i < mDimX; i++) {
        for (int j = 0; j < mDimY; j++) {
            for (int kSource = 0; kSource < mDimZ; kSource++) {
                mResult[kSource][i][j][0] = 0.0;
                mResult[kSource][i][j][1] = 0.0;
                mResult[kSource][i][j][2] = 0.0;

                for (int kDest = 0; kDest < mDimZ; kDest++) {
                    mResult[kSource][i][j][0] += mTestMultInv[kSource * mDimZ + kDest][i][j][0].real();
                    mResult[kSource][i][j][1] += mTestMultInv[kSource * mDimZ + kDest][i][j][1].real();
                    mResult[kSource][i][j][2] += mTestMultInv[kSource * mDimZ + kDest][i][j][2].real();
                }

                mResult[kSource][i][j][0] *= Norm;
                mResult[kSource][i][j][1] *= Norm;
                mResult[kSource][i][j][2] *= Norm;
            }
        }
    }

    //----------//

    fftw_free(In2);
    fftw_free(Out2);

    fftw_free(In);
    fftw_free(Out);
}
[[nodiscard]] const DemagField::MagGrid& DemagField::getField() const {
    return mResult;
}