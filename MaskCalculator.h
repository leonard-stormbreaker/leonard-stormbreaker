//
// Created by devilox on 2/13/22.
//
//-----------------------------//
#ifndef SPINPMCALC_MASKCALCULATOR_H
#define SPINPMCALC_MASKCALCULATOR_H
//-----------------------------//
#include <vector>
#include <cstdint>
#include <cmath>
//-----------------------------//
#include "../Shared.h"
//-----------------------------//
struct MaskLayer {
    double      mCenterX    = 0.0;
    double      mCenterY    = 0.0;

    double      mWidth      = 50.0;
    double      mHeight     = 50.0;

    Mask::Type  mType       = Mask::Type::ELLIPSE;

    int32_t     mIndex      = -1;
};
//-----------------------------//
class MaskCalculator {
    using MaskGrid = std::vector <std::vector <int8_t>>;
public:
    MaskCalculator( double tGridLeft,   double tGridRight,
                    double tGridUp,     double tGridDown,
                    double tGridStepX,  double tGridStepY);

    [[nodiscard]] MaskGrid calculateMask(   Mask::Type tType,
                                            double tMaskCenterX,    double tMaskCenterY,
                                            double tMaskWidth,      double tMaskHeight,
                                            size_t& tMagnCellNum) const;

    [[nodiscard]] size_t getDimX() const;
    [[nodiscard]] size_t getDimY() const;
private:
    double mGridLeft    = 0;
    double mGridRight   = 0;
    double mGridUp      = 0;
    double mGridDown    = 0;

    double mGridStepX   = 0;
    double mGridStepY   = 0;

    size_t mDimX        = 0;
    size_t mDimY        = 0;
};
//-----------------------------//
#endif //SPINPMCALC_MASKCALCULATOR_H
