//
// Created by devilox on 11/25/21.
//
//-----------------------------//
#include "EffectiveField.h"
//-----------------------------//
EffectiveField::EffectiveField(uint32_t tDimX,  uint32_t tDimY,     uint32_t tDimZ,
                               double tDeltaX,  double tDeltaY,
                               const std::vector <Material*>& tMaterials,
                               const ThicknessVector& tDeltaZ,
                               const MaskGrid& tMasks) : mMasks(tMasks), mDeltaZ(tDeltaZ) {
    mDimX       = tDimX;
    mDimY       = tDimY;
    mDimZ       = tDimZ;

    mDeltaX     = tDeltaX;
    mDeltaY     = tDeltaY;

    mDemag      = new DemagField(tDimX, tDimY, tDimZ);
    mTensor     = new DemagTensor(tDimX, tDimY, tDimZ, tDeltaX, tDeltaY);

    mTensor -> calcTensor(tDeltaZ, tMasks);
    mTensor -> calcImage();

    mField.resize(tDimZ);
    Test.resize(tDimZ);

    for (int i = 0; i < tDimZ; i++) {
        mField[i].resize(tDimX);
        Test[i].resize(tDimX);

        for (int j = 0; j < tDimX; j++) {
            mField[i][j].resize(tDimY, {0.0, 0.0, 0.0});
            Test[i][j].resize(tDimY, {0.0, 0.0, 0.0});
        }
    }

    mMaterials = tMaterials;
}
//-----------------------------//
void EffectiveField::calc(const VecGrid& tExternalField, const VecGrid& tMagn) {
    norml();

    mDemag -> calcField(tMagn, mTensor -> getImage());
    mDemagField = mDemag -> getField();

//    double AnisCoeff = 2.0 * mCoeffK / std::pow(mMagnSatur, 2.0);

    double mu0 = 4.0e-07 * M_PI;

#pragma omp parallel for default(none) shared(tMagn, mMasks, tExternalField, mu0) num_threads(4)
    for (int k = 0; k < mDimZ; k++) {
        auto Material = dynamic_cast <MagneticMaterial*>(mMaterials[k]);
        if (Material == nullptr) {
            continue;
        }

        double ExchCoeff = 2.0 * Material -> getA() / (Material -> getMs() * mu0);

        for (int i = 0; i < mDimX; i++) {
            for (int j = 0; j < mDimY; j++) {
                if (mMasks[k][i][j] < 1) {
                    continue;
                }

//                double AnisX = AnisCoeff * (tMagn[k][i][j][0] * ml[0] + tMagn[k][i][j][1] * ml[1] + tMagn[k][i][j][2] * ml[2]) * ml[0];
//                double AnisY = AnisCoeff * (tMagn[k][i][j][0] * ml[0] + tMagn[k][i][j][1] * ml[1] + tMagn[k][i][j][2] * ml[2]) * ml[1];
//                double AnisZ = AnisCoeff * (tMagn[k][i][j][0] * ml[0] + tMagn[k][i][j][1] * ml[1] + tMagn[k][i][j][2] * ml[2]) * ml[2];

                //----------//

                double d2MagnXdx2;
                double d2MagnYdx2;
                double d2MagnZdx2;

                double d2MagnXdy2;
                double d2MagnYdy2;
                double d2MagnZdy2;

                double d2MagnXdz2;
                double d2MagnYdz2;
                double d2MagnZdz2;

                if (mDimX > 1) {
//                    if (i == 0) {
                    if (i == 0 || mMasks[k][i - 1][j] < 1) {
                        d2MagnXdx2 = tMagn[k][i][j][0] - 2.0 * tMagn[k][i][j][0] + tMagn[k][i + 1][j][0];
                        d2MagnYdx2 = tMagn[k][i][j][1] - 2.0 * tMagn[k][i][j][1] + tMagn[k][i + 1][j][1];
                        d2MagnZdx2 = tMagn[k][i][j][2] - 2.0 * tMagn[k][i][j][2] + tMagn[k][i + 1][j][2];
//                    } else if (i == mDimX - 1) {
                    } else if (i == mDimX - 1 || mMasks[k][i + 1][j] < 1) {
                        d2MagnXdx2 = tMagn[k][i - 1][j][0] - 2.0 * tMagn[k][i][j][0] + tMagn[k][i][j][0];
                        d2MagnYdx2 = tMagn[k][i - 1][j][1] - 2.0 * tMagn[k][i][j][1] + tMagn[k][i][j][1];
                        d2MagnZdx2 = tMagn[k][i - 1][j][2] - 2.0 * tMagn[k][i][j][2] + tMagn[k][i][j][2];
                    } else {
                        d2MagnXdx2 = tMagn[k][i - 1][j][0] - 2.0 * tMagn[k][i][j][0] + tMagn[k][i + 1][j][0];
                        d2MagnYdx2 = tMagn[k][i - 1][j][1] - 2.0 * tMagn[k][i][j][1] + tMagn[k][i + 1][j][1];
                        d2MagnZdx2 = tMagn[k][i - 1][j][2] - 2.0 * tMagn[k][i][j][2] + tMagn[k][i + 1][j][2];
                    }
                } else {
                    d2MagnXdx2 = 0.0;
                    d2MagnYdx2 = 0.0;
                    d2MagnZdx2 = 0.0;
                }

                if (mDimY > 1) {
//                    if (j == 0) {
                    if (j == 0 || mMasks[k][i][j - 1] < 1) {
                        d2MagnXdy2 = tMagn[k][i][j][0] - 2.0 * tMagn[k][i][j][0] + tMagn[k][i][j + 1][0];
                        d2MagnYdy2 = tMagn[k][i][j][1] - 2.0 * tMagn[k][i][j][1] + tMagn[k][i][j + 1][1];
                        d2MagnZdy2 = tMagn[k][i][j][2] - 2.0 * tMagn[k][i][j][2] + tMagn[k][i][j + 1][2];
//                    } else if (j == mDimY - 1) {
                    } else if (j == mDimY - 1 || mMasks[k][i][j + 1] < 1) {
                        d2MagnXdy2 = tMagn[k][i][j - 1][0] - 2.0 * tMagn[k][i][j][0] + tMagn[k][i][j][0];
                        d2MagnYdy2 = tMagn[k][i][j - 1][1] - 2.0 * tMagn[k][i][j][1] + tMagn[k][i][j][1];
                        d2MagnZdy2 = tMagn[k][i][j - 1][2] - 2.0 * tMagn[k][i][j][2] + tMagn[k][i][j][2];
                    } else {
                        d2MagnXdy2 = tMagn[k][i][j - 1][0] - 2.0 * tMagn[k][i][j][0] + tMagn[k][i][j + 1][0];
                        d2MagnYdy2 = tMagn[k][i][j - 1][1] - 2.0 * tMagn[k][i][j][1] + tMagn[k][i][j + 1][1];
                        d2MagnZdy2 = tMagn[k][i][j - 1][2] - 2.0 * tMagn[k][i][j][2] + tMagn[k][i][j + 1][2];
                    }
                } else {
                    d2MagnXdy2 = 0.0;
                    d2MagnYdy2 = 0.0;
                    d2MagnZdy2 = 0.0;
                }

                if (mDimZ > 1) {
//                    if (k == 0) {
                    if (k == 0 || mMasks[k - 1][i][j] < 1) {
                        d2MagnXdz2 = tMagn[k][i][j][0] - 2.0 * tMagn[k][i][j][0] + tMagn[k + 1][i][j][0];
                        d2MagnYdz2 = tMagn[k][i][j][1] - 2.0 * tMagn[k][i][j][1] + tMagn[k + 1][i][j][1];
                        d2MagnZdz2 = tMagn[k][i][j][2] - 2.0 * tMagn[k][i][j][2] + tMagn[k + 1][i][j][2];
//                    } else if (k == mDimZ - 1) {
                    } else if (k == mDimZ - 1 || mMasks[k + 1][i][j] < 1) {
                        d2MagnXdz2 = tMagn[k - 1][i][j][0] - 2.0 * tMagn[k][i][j][0] + tMagn[k][i][j][0];
                        d2MagnYdz2 = tMagn[k - 1][i][j][1] - 2.0 * tMagn[k][i][j][1] + tMagn[k][i][j][1];
                        d2MagnZdz2 = tMagn[k - 1][i][j][2] - 2.0 * tMagn[k][i][j][2] + tMagn[k][i][j][2];
                    } else {
                        d2MagnXdz2 = tMagn[k - 1][i][j][0] - 2.0 * tMagn[k][i][j][0] + tMagn[k + 1][i][j][0];
                        d2MagnYdz2 = tMagn[k - 1][i][j][1] - 2.0 * tMagn[k][i][j][1] + tMagn[k + 1][i][j][1];
                        d2MagnZdz2 = tMagn[k - 1][i][j][2] - 2.0 * tMagn[k][i][j][2] + tMagn[k + 1][i][j][2];
                    }
                } else {
                    d2MagnXdz2 = 0.0;
                    d2MagnYdz2 = 0.0;
                    d2MagnZdz2 = 0.0;
                }

//                mField[k][i][j][0] = mDemagField[k][i][j][0];// * Material -> getMs();
//                mField[k][i][j][1] = mDemagField[k][i][j][1];// * Material -> getMs();
//                mField[k][i][j][2] = mDemagField[k][i][j][2];// * Material -> getMs();
//
//                Test[k][i][j][0] = d2MagnXdx2 / std::pow(mDeltaX, 2.0) +
//                                   d2MagnXdy2 / std::pow(mDeltaY, 2.0) +
//                                   d2MagnXdz2 / std::pow(mDeltaZ[k], 2.0);
//                Test[k][i][j][1] = d2MagnYdx2 / std::pow(mDeltaX, 2.0) +
//                                   d2MagnYdy2 / std::pow(mDeltaY, 2.0) +
//                                   d2MagnYdz2 / std::pow(mDeltaZ[k], 2.0);
//                Test[k][i][j][2] = d2MagnZdx2 / std::pow(mDeltaX, 2.0) +
//                                   d2MagnZdy2 / std::pow(mDeltaY, 2.0) +
//                                   d2MagnZdz2 / std::pow(mDeltaZ[k], 2.0);

                mField[k][i][j][0] = ExchCoeff *
                                    (d2MagnXdx2 / std::pow(mDeltaX, 2.0) +
                                    d2MagnXdy2 / std::pow(mDeltaY, 2.0) +
                                    d2MagnXdz2 / std::pow(mDeltaZ[k], 2.0)) +
                                    tExternalField[k][i][j][0] + mDemagField[k][i][j][0] * Material -> getMs();
                mField[k][i][j][1] = ExchCoeff *
                                    (d2MagnYdx2 / std::pow(mDeltaX, 2.0) +
                                    d2MagnYdy2 / std::pow(mDeltaY, 2.0) +
                                    d2MagnYdz2 / std::pow(mDeltaZ[k], 2.0)) +
                                    tExternalField[k][i][j][1] + mDemagField[k][i][j][1] * Material -> getMs();
                mField[k][i][j][2] = ExchCoeff *
                                    (d2MagnZdx2 / std::pow(mDeltaX, 2.0) +
                                    d2MagnZdy2 / std::pow(mDeltaY, 2.0) +
                                    d2MagnZdz2 / std::pow(mDeltaZ[k], 2.0)) +
                                    tExternalField[k][i][j][2] + mDemagField[k][i][j][2] * Material -> getMs();
            }
        }
    }

//    for (int iDim = 0; iDim < 3; iDim++) {
//        for (int i = 0; i < mDimX; i++) {
//            for (int j = 0; j < mDimY; j++) {
//                std::cout << mField[0][i][j][iDim] << "\t";
//            }
//
//            std::cout << std::endl;
//        }
//
//        std::cout << std::endl;
//    }
}

[[nodiscard]] const EffectiveField::VecGrid& EffectiveField::getField() const {
    return mField;
}
//-----------------------------//
void EffectiveField::norml() {
//    double Module = std::sqrt(std::pow(ml[0], 2.0) + std::pow(ml[1], 2.0) + std::pow(ml[2], 2.0));
//
//    ml[0] /= Module;
//    ml[1] /= Module;
//    ml[2] /= Module;
}