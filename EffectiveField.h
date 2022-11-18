//
// Created by devilox on 11/25/21.
//
//-----------------------------//
#ifndef SPINPMCALC_EFFECTIVEFIELD_H
#define SPINPMCALC_EFFECTIVEFIELD_H
//-----------------------------//
#include <vector>
#include <array>
#include <cmath>
#include <iostream>
//-----------------------------//
#include "DemagField.h"
#include "DemagTensor.h"
#include "../Shared.h"
//-----------------------------//
class EffectiveField {
private:
    using VecGrid = std::vector <std::vector <std::vector <std::array <double, 3>>>>;
    using MaskGrid = std::vector <std::vector <std::vector <int8_t>>>;
    using ThicknessVector = std::vector <double>;
public:
    ///---TODO: fix these shitty namings---//
    EffectiveField(uint32_t tDimX,  uint32_t tDimY,     uint32_t tDimZ,
                   double tDeltaX,  double tDeltaY,
                   const std::vector <Material*>& tMaterials,
                   const ThicknessVector& tDeltaZ,
                   const MaskGrid& tMasks);

    void calc(const VecGrid& tExternalField, const VecGrid& tMagn);

    [[nodiscard]] const VecGrid& getField() const;
private:
    uint32_t                    mDimX;
    uint32_t                    mDimY;
    uint32_t                    mDimZ;

    double                      mDeltaX;
    double                      mDeltaY;

//    double                      mCoeffA;
//    double                      mCoeffK;
//    double                      mMagnSatur;

    ///---TODO: rename this shit---///
//    std::array <double, 3>      ml              = {};

    VecGrid                     mField;
    VecGrid                     Test;
    VecGrid                     mDemagField;

    DemagField*                 mDemag          = nullptr;
    DemagTensor*                mTensor         = nullptr;

    std::vector <Material*>     mMaterials;

    const MaskGrid&             mMasks;
    const ThicknessVector&      mDeltaZ;

    //----------//

    void norml();
};
//-----------------------------//
#endif //SPINPMCALC_EFFECTIVEFIELD_H
