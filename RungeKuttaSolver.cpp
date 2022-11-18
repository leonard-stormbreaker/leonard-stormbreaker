//
// Created by devilox on 11/27/21.
//
//-----------------------------//
#include "RungeKuttaSolver.h"

#include <iomanip>
//-----------------------------//
RungeKuttaSolver::RungeKuttaSolver( uint32_t tDimX,     uint32_t tDimY,     uint32_t tDimZ,
                                    double tDeltaX,     double tDeltaY,     double tDeltaT,
                                    const std::vector <Material*>& tMaterials,
                                    const MaskGrid& tMasks,
                                    const ThicknessVector& tDeltaZ) {
    mHeff       = new EffectiveField(tDimX, tDimY, tDimZ, tDeltaX, tDeltaY, tMaterials, tDeltaZ, tMasks);

    mDimX       = tDimX;
    mDimY       = tDimY;
    mDimZ       = tDimZ;

    mDeltaT     = tDeltaT;

    mMagn.resize(tDimZ);
    mNewMagn.resize(tDimZ);

    mMCrossHeff.resize(tDimZ);
    mMCrossMCrossHeff.resize(tDimZ);
    mkVec.resize(tDimZ);

    for (int k = 0; k < tDimZ; k++) {
        mMagn[k].resize(tDimX);
        mNewMagn[k].resize(tDimX);

        mMCrossHeff[k].resize(tDimX);
        mMCrossMCrossHeff[k].resize(tDimX);
        mkVec[k].resize(tDimX);

        for (int i = 0; i < tDimX; i++) {
            mMagn[k][i].resize(tDimY, {0.0, 0.0, 0.0});
            mNewMagn[k][i].resize(tDimY, {0.0, 0.0, 0.0});

            mMCrossHeff[k][i].resize(tDimY, {0.0, 0.0, 0.0});
            mMCrossMCrossHeff[k][i].resize(tDimY, {0.0, 0.0, 0.0});
            mkVec[k][i].resize(tDimY, {0.0, 0.0, 0.0});
        }
    }

    mMaterials = tMaterials;
}

RungeKuttaSolver::~RungeKuttaSolver() {
    delete(mHeff);
}
//-----------------------------//
void RungeKuttaSolver::calc(const VecGrid& tExternalField, VecGrid& tMagn) {
    if (mHeff == nullptr) {
        throw std::runtime_error("Error");
    }

    double HalfTimeStep = mDeltaT / 2.0;
    double TimeKCoeff = mDeltaT / 6.0;

    int NumThreads = 4;

    //---Heun---//
//    mHeff -> calc(tExternalField, tMagn, tDeltaZ);
//    auto HeffField = mHeff -> getField();
//
//    for (int i = 0; i < mDimX; i++) {
//        for (int j = 0; j < mDimY; j++) {
//            for (int k = 0; k < mDimZ; k++) {
//                MCrossHeff[0] = tMagn[i][j][k][1] * HeffField[i][j][k][2] - tMagn[i][j][k][2] * HeffField[i][j][k][1];
//                MCrossHeff[1] = tMagn[i][j][k][2] * HeffField[i][j][k][0] - tMagn[i][j][k][0] * HeffField[i][j][k][2];
//                MCrossHeff[2] = tMagn[i][j][k][0] * HeffField[i][j][k][1] - tMagn[i][j][k][1] * HeffField[i][j][k][0];
//
//                MCrossMCrossHeff[0] = tMagn[i][j][k][1] * MCrossHeff[2] - tMagn[i][j][k][2] * MCrossHeff[1];
//                MCrossMCrossHeff[1] = tMagn[i][j][k][2] * MCrossHeff[0] - tMagn[i][j][k][0] * MCrossHeff[2];
//                MCrossMCrossHeff[2] = tMagn[i][j][k][0] * MCrossHeff[1] - tMagn[i][j][k][1] * MCrossHeff[0];
//
//                mMagn[i][j][k][0] = tMagn[i][j][k][0] + mDeltaT * GammaAlphaCoeff * (MCrossHeff[0] + mAlpha * MCrossMCrossHeff[0]);
//                mMagn[i][j][k][1] = tMagn[i][j][k][1] + mDeltaT * GammaAlphaCoeff * (MCrossHeff[1] + mAlpha * MCrossMCrossHeff[1]);
//                mMagn[i][j][k][2] = tMagn[i][j][k][2] + mDeltaT * GammaAlphaCoeff * (MCrossHeff[2] + mAlpha * MCrossMCrossHeff[2]);
//
//                mNewMagn[i][j][k][0] = tMagn[i][j][k][0] + HalfTimeStep * GammaAlphaCoeff * (MCrossHeff[0] + mAlpha * MCrossMCrossHeff[0]);
//                mNewMagn[i][j][k][1] = tMagn[i][j][k][1] + HalfTimeStep * GammaAlphaCoeff * (MCrossHeff[1] + mAlpha * MCrossMCrossHeff[1]);
//                mNewMagn[i][j][k][2] = tMagn[i][j][k][2] + HalfTimeStep * GammaAlphaCoeff * (MCrossHeff[2] + mAlpha * MCrossMCrossHeff[2]);
//            }
//        }
//    }
//
//    //----------//
//
//    mHeff -> calc(tExternalField, mMagn, tDeltaZ);
//    HeffField = mHeff -> getField();
//
//    for (int i = 0; i < mDimX; i++) {
//        for (int j = 0; j < mDimY; j++) {
//            for (int k = 0; k < mDimZ; k++) {
//                MCrossHeff[0] = mMagn[i][j][k][1] * HeffField[i][j][k][2] - mMagn[i][j][k][2] * HeffField[i][j][k][1];
//                MCrossHeff[1] = mMagn[i][j][k][2] * HeffField[i][j][k][0] - mMagn[i][j][k][0] * HeffField[i][j][k][2];
//                MCrossHeff[2] = mMagn[i][j][k][0] * HeffField[i][j][k][1] - mMagn[i][j][k][1] * HeffField[i][j][k][0];
//
//                MCrossMCrossHeff[0] = mMagn[i][j][k][1] * MCrossHeff[2] - mMagn[i][j][k][2] * MCrossHeff[1];
//                MCrossMCrossHeff[1] = mMagn[i][j][k][2] * MCrossHeff[0] - mMagn[i][j][k][0] * MCrossHeff[2];
//                MCrossMCrossHeff[2] = mMagn[i][j][k][0] * MCrossHeff[1] - mMagn[i][j][k][1] * MCrossHeff[0];
//
//                mNewMagn[i][j][k][0] += HalfTimeStep * GammaAlphaCoeff * (MCrossHeff[0] + mAlpha * MCrossMCrossHeff[0]);
//                mNewMagn[i][j][k][1] += HalfTimeStep * GammaAlphaCoeff * (MCrossHeff[1] + mAlpha * MCrossMCrossHeff[1]);
//                mNewMagn[i][j][k][2] += HalfTimeStep * GammaAlphaCoeff * (MCrossHeff[2] + mAlpha * MCrossMCrossHeff[2]);
//            }
//        }
//    }
    //---Heun---//

    //---Runge-Kutta---//
    mHeff -> calc(tExternalField, tMagn);
    auto HeffField = mHeff -> getField();

//    for (int k = 0; k < mDimZ; k++) {
//        for (int iDim = 0; iDim < 3; iDim++) {
//            for (int i = 0; i < mDimX; i++) {
//                for (int j = 0; j < mDimY; j++) {
//                    auto val = HeffField[k][i][j][iDim];
//                    std::cout << val << "\t";
//                }
//                std::cout << std::endl;
//            }
//            std::cout << std::endl;
//        }
//
//        std::cout << std::endl;
//    }

#pragma omp parallel for default(none) shared(tMagn, HeffField, HalfTimeStep) num_threads(NumThreads)
    for (int k = 0; k < mDimZ; k++) {
        auto Material = dynamic_cast <MagneticMaterial*>(mMaterials[k]);
        if (Material == nullptr) {
            continue;
        }

        double GammaAlphaCoeff = -Material -> getGamma() / (1.0 + std::pow(Material -> getAlpha(), 2.0));

        for (int i = 0; i < mDimX; i++) {
            for (int j = 0; j < mDimY; j++) {
                mMCrossHeff[k][i][j][0] = tMagn[k][i][j][1] * HeffField[k][i][j][2] - tMagn[k][i][j][2] * HeffField[k][i][j][1];
                mMCrossHeff[k][i][j][1] = tMagn[k][i][j][2] * HeffField[k][i][j][0] - tMagn[k][i][j][0] * HeffField[k][i][j][2];
                mMCrossHeff[k][i][j][2] = tMagn[k][i][j][0] * HeffField[k][i][j][1] - tMagn[k][i][j][1] * HeffField[k][i][j][0];

                mMCrossMCrossHeff[k][i][j][0] = tMagn[k][i][j][1] * mMCrossHeff[k][i][j][2] - tMagn[k][i][j][2] * mMCrossHeff[k][i][j][1];
                mMCrossMCrossHeff[k][i][j][1] = tMagn[k][i][j][2] * mMCrossHeff[k][i][j][0] - tMagn[k][i][j][0] * mMCrossHeff[k][i][j][2];
                mMCrossMCrossHeff[k][i][j][2] = tMagn[k][i][j][0] * mMCrossHeff[k][i][j][1] - tMagn[k][i][j][1] * mMCrossHeff[k][i][j][0];

                mkVec[k][i][j][0] = GammaAlphaCoeff * (mMCrossHeff[k][i][j][0] + Material -> getAlpha() * mMCrossMCrossHeff[k][i][j][0]);
                mkVec[k][i][j][1] = GammaAlphaCoeff * (mMCrossHeff[k][i][j][1] + Material -> getAlpha() * mMCrossMCrossHeff[k][i][j][1]);
                mkVec[k][i][j][2] = GammaAlphaCoeff * (mMCrossHeff[k][i][j][2] + Material -> getAlpha() * mMCrossMCrossHeff[k][i][j][2]);

                mMagn[k][i][j][0] = tMagn[k][i][j][0] + HalfTimeStep * mkVec[k][i][j][0];
                mMagn[k][i][j][1] = tMagn[k][i][j][1] + HalfTimeStep * mkVec[k][i][j][1];
                mMagn[k][i][j][2] = tMagn[k][i][j][2] + HalfTimeStep * mkVec[k][i][j][2];

                mNewMagn[k][i][j][0] = mkVec[k][i][j][0];
                mNewMagn[k][i][j][1] = mkVec[k][i][j][1];
                mNewMagn[k][i][j][2] = mkVec[k][i][j][2];
            }
        }
    }

    //----------//

    mHeff -> calc(tExternalField, mMagn);
    HeffField = mHeff -> getField();

#pragma omp parallel for default(none) shared(tMagn, HeffField, HalfTimeStep) num_threads(NumThreads)
    for (int k = 0; k < mDimZ; k++) {
        auto Material = dynamic_cast <MagneticMaterial*>(mMaterials[k]);
        if (Material == nullptr) {
            continue;
        }

        double GammaAlphaCoeff = -Material -> getGamma() / (1.0 + std::pow(Material -> getAlpha(), 2.0));

        for (int i = 0; i < mDimX; i++) {
            for (int j = 0; j < mDimY; j++) {
                mMCrossHeff[k][i][j][0] = mMagn[k][i][j][1] * HeffField[k][i][j][2] - mMagn[k][i][j][2] * HeffField[k][i][j][1];
                mMCrossHeff[k][i][j][1] = mMagn[k][i][j][2] * HeffField[k][i][j][0] - mMagn[k][i][j][0] * HeffField[k][i][j][2];
                mMCrossHeff[k][i][j][2] = mMagn[k][i][j][0] * HeffField[k][i][j][1] - mMagn[k][i][j][1] * HeffField[k][i][j][0];

                mMCrossMCrossHeff[k][i][j][0] = mMagn[k][i][j][1] * mMCrossHeff[k][i][j][2] - mMagn[k][i][j][2] * mMCrossHeff[k][i][j][1];
                mMCrossMCrossHeff[k][i][j][1] = mMagn[k][i][j][2] * mMCrossHeff[k][i][j][0] - mMagn[k][i][j][0] * mMCrossHeff[k][i][j][2];
                mMCrossMCrossHeff[k][i][j][2] = mMagn[k][i][j][0] * mMCrossHeff[k][i][j][1] - mMagn[k][i][j][1] * mMCrossHeff[k][i][j][0];

                mkVec[k][i][j][0] = GammaAlphaCoeff * (mMCrossHeff[k][i][j][0] + Material -> getAlpha() * mMCrossMCrossHeff[k][i][j][0]);
                mkVec[k][i][j][1] = GammaAlphaCoeff * (mMCrossHeff[k][i][j][1] + Material -> getAlpha() * mMCrossMCrossHeff[k][i][j][1]);
                mkVec[k][i][j][2] = GammaAlphaCoeff * (mMCrossHeff[k][i][j][2] + Material -> getAlpha() * mMCrossMCrossHeff[k][i][j][2]);

                mMagn[k][i][j][0] = tMagn[k][i][j][0] + HalfTimeStep * mkVec[k][i][j][0];
                mMagn[k][i][j][1] = tMagn[k][i][j][1] + HalfTimeStep * mkVec[k][i][j][1];
                mMagn[k][i][j][2] = tMagn[k][i][j][2] + HalfTimeStep * mkVec[k][i][j][2];

                mNewMagn[k][i][j][0] += mkVec[k][i][j][0] * 2.0;
                mNewMagn[k][i][j][1] += mkVec[k][i][j][1] * 2.0;
                mNewMagn[k][i][j][2] += mkVec[k][i][j][2] * 2.0;
            }
        }
    }

    //----------//

    mHeff -> calc(tExternalField, mMagn);
    HeffField = mHeff -> getField();

#pragma omp parallel for default(none) shared(tMagn, HeffField) num_threads(NumThreads)
    for (int k = 0; k < mDimZ; k++) {
        auto Material = dynamic_cast <MagneticMaterial*>(mMaterials[k]);
        if (Material == nullptr) {
            continue;
        }

        double GammaAlphaCoeff = -Material -> getGamma() / (1.0 + std::pow(Material -> getAlpha(), 2.0));

        for (int i = 0; i < mDimX; i++) {
            for (int j = 0; j < mDimY; j++) {
                mMCrossHeff[k][i][j][0] = mMagn[k][i][j][1] * HeffField[k][i][j][2] - mMagn[k][i][j][2] * HeffField[k][i][j][1];
                mMCrossHeff[k][i][j][1] = mMagn[k][i][j][2] * HeffField[k][i][j][0] - mMagn[k][i][j][0] * HeffField[k][i][j][2];
                mMCrossHeff[k][i][j][2] = mMagn[k][i][j][0] * HeffField[k][i][j][1] - mMagn[k][i][j][1] * HeffField[k][i][j][0];

                mMCrossMCrossHeff[k][i][j][0] = mMagn[k][i][j][1] * mMCrossHeff[k][i][j][2] - mMagn[k][i][j][2] * mMCrossHeff[k][i][j][1];
                mMCrossMCrossHeff[k][i][j][1] = mMagn[k][i][j][2] * mMCrossHeff[k][i][j][0] - mMagn[k][i][j][0] * mMCrossHeff[k][i][j][2];
                mMCrossMCrossHeff[k][i][j][2] = mMagn[k][i][j][0] * mMCrossHeff[k][i][j][1] - mMagn[k][i][j][1] * mMCrossHeff[k][i][j][0];

                mkVec[k][i][j][0] = GammaAlphaCoeff * (mMCrossHeff[k][i][j][0] + Material -> getAlpha() * mMCrossMCrossHeff[k][i][j][0]);
                mkVec[k][i][j][1] = GammaAlphaCoeff * (mMCrossHeff[k][i][j][1] + Material -> getAlpha() * mMCrossMCrossHeff[k][i][j][1]);
                mkVec[k][i][j][2] = GammaAlphaCoeff * (mMCrossHeff[k][i][j][2] + Material -> getAlpha() * mMCrossMCrossHeff[k][i][j][2]);

                mMagn[k][i][j][0] = tMagn[k][i][j][0] + mDeltaT * mkVec[k][i][j][0];
                mMagn[k][i][j][1] = tMagn[k][i][j][1] + mDeltaT * mkVec[k][i][j][1];
                mMagn[k][i][j][2] = tMagn[k][i][j][2] + mDeltaT * mkVec[k][i][j][2];

                mNewMagn[k][i][j][0] += mkVec[k][i][j][0] * 2.0;
                mNewMagn[k][i][j][1] += mkVec[k][i][j][1] * 2.0;
                mNewMagn[k][i][j][2] += mkVec[k][i][j][2] * 2.0;
            }
        }
    }

    //----------//

    mHeff -> calc(tExternalField, mMagn);
    HeffField = mHeff -> getField();

#pragma omp parallel for default(none) shared(tMagn, HeffField) num_threads(NumThreads)
    for (int k = 0; k < mDimZ; k++) {
        auto Material = dynamic_cast <MagneticMaterial*>(mMaterials[k]);
        if (Material == nullptr) {
            continue;
        }

        double GammaAlphaCoeff = -Material -> getGamma() / (1.0 + std::pow(Material -> getAlpha(), 2.0));

        for (int i = 0; i < mDimX; i++) {
            for (int j = 0; j < mDimY; j++) {
                mMCrossHeff[k][i][j][0] = mMagn[k][i][j][1] * HeffField[k][i][j][2] - mMagn[k][i][j][2] * HeffField[k][i][j][1];
                mMCrossHeff[k][i][j][1] = mMagn[k][i][j][2] * HeffField[k][i][j][0] - mMagn[k][i][j][0] * HeffField[k][i][j][2];
                mMCrossHeff[k][i][j][2] = mMagn[k][i][j][0] * HeffField[k][i][j][1] - mMagn[k][i][j][1] * HeffField[k][i][j][0];

                mMCrossMCrossHeff[k][i][j][0] = mMagn[k][i][j][1] * mMCrossHeff[k][i][j][2] - mMagn[k][i][j][2] * mMCrossHeff[k][i][j][1];
                mMCrossMCrossHeff[k][i][j][1] = mMagn[k][i][j][2] * mMCrossHeff[k][i][j][0] - mMagn[k][i][j][0] * mMCrossHeff[k][i][j][2];
                mMCrossMCrossHeff[k][i][j][2] = mMagn[k][i][j][0] * mMCrossHeff[k][i][j][1] - mMagn[k][i][j][1] * mMCrossHeff[k][i][j][0];

                mkVec[k][i][j][0] = GammaAlphaCoeff * (mMCrossHeff[k][i][j][0] + Material -> getAlpha() * mMCrossMCrossHeff[k][i][j][0]);
                mkVec[k][i][j][1] = GammaAlphaCoeff * (mMCrossHeff[k][i][j][1] + Material -> getAlpha() * mMCrossMCrossHeff[k][i][j][1]);
                mkVec[k][i][j][2] = GammaAlphaCoeff * (mMCrossHeff[k][i][j][2] + Material -> getAlpha() * mMCrossMCrossHeff[k][i][j][2]);

                mNewMagn[k][i][j][0] += mkVec[k][i][j][0];
                mNewMagn[k][i][j][1] += mkVec[k][i][j][1];
                mNewMagn[k][i][j][2] += mkVec[k][i][j][2];
            }
        }
    }

    //----------//

#pragma omp parallel for default(none) shared(tMagn, TimeKCoeff) num_threads(NumThreads)
    for (int k = 0; k < mDimZ; k++) {
        for (int i = 0; i < mDimX; i++) {
            for (int j = 0; j < mDimY; j++) {
                tMagn[k][i][j][0] += TimeKCoeff * mNewMagn[k][i][j][0];
                tMagn[k][i][j][1] += TimeKCoeff * mNewMagn[k][i][j][1];
                tMagn[k][i][j][2] += TimeKCoeff * mNewMagn[k][i][j][2];

                double norm = std::sqrt(std::pow(tMagn[k][i][j][0], 2.0) + std::pow(tMagn[k][i][j][1], 2.0) + std::pow(tMagn[k][i][j][2], 2.0));

                if (norm != 0) {
                    tMagn[k][i][j][0] /= norm;
                    tMagn[k][i][j][1] /= norm;
                    tMagn[k][i][j][2] /= norm;
                }
            }
        }
    }
}