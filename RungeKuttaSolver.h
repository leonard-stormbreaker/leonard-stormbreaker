//
// Created by devilox on 11/27/21.
//
//-----------------------------//
#ifndef SPINPMCALC_RUNGEKUTTASOLVER_H
#define SPINPMCALC_RUNGEKUTTASOLVER_H
//-----------------------------//
#include <vector>
#include <array>
#include <cmath>
#include <iostream>
//-----------------------------//
#include "EffectiveField.h"
#include "DemagTensor.h"
#include "DemagField.h"
//-----------------------------//
class RungeKuttaSolver {
private:
    using VecGrid = std::vector <std::vector <std::vector <std::array <double, 3>>>>;
    using MaskGrid = std::vector <std::vector <std::vector <int8_t>>>;
    using ThicknessVector = std::vector <double>;
public:
    ///---TODO: fix these shitty namings---//
    ///---TODO: make tDeltaT adjustable---//
    RungeKuttaSolver(   uint32_t tDimX,     uint32_t tDimY,     uint32_t tDimZ,
                        double tDeltaX,     double tDeltaY,     double tDeltaT,
                        const std::vector <Material*>& tMaterials,
                        const MaskGrid& tMasks,
                        const ThicknessVector& tDeltaZ);
    ~RungeKuttaSolver();

    void calc(const VecGrid& tExternalField, VecGrid& tMagn);
private:
    uint32_t                    mDimX;
    uint32_t                    mDimY;
    uint32_t                    mDimZ;

    ///---TODO: fix this later---///
    double                      mDeltaT;

    EffectiveField*             mHeff           = nullptr;

    VecGrid                     mMagn;
    VecGrid                     mNewMagn;

    VecGrid                     mMCrossHeff;
    VecGrid                     mMCrossMCrossHeff;
    VecGrid                     mkVec;

    std::vector <Material*>     mMaterials;

    bool                        mFirstCalc = true;
};
//-----------------------------//
#endif //SPINPMCALC_RUNGEKUTTASOLVER_H
