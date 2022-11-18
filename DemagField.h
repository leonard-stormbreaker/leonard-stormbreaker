//
// Created by devilox on 11/20/21.
//
//-----------------------------//
#ifndef SPINPMCALC_DEMAGFIELD_H
#define SPINPMCALC_DEMAGFIELD_H
//-----------------------------//
#include <vector>
#include <array>
#include <iostream>
#include <complex>
//-----------------------------//
#include <fftw3.h>
//-----------------------------//
class DemagField {
private:
    using MagGrid = std::vector <std::vector <std::vector <std::array <double, 3>>>>;
    using ImageMagGrid = std::vector <std::vector <std::vector <std::array <std::complex <double>, 3>>>>;
    using ImageTensorGrid = std::vector <std::vector <std::vector <std::array <std::complex <double>, 9>>>>;
public:
    DemagField(uint32_t tDimX, uint32_t tDimY, uint32_t tDimZ);

    void calcField(const MagGrid& tMag, const ImageTensorGrid& tDemagTensorImage);
    [[nodiscard]] const MagGrid& getField() const;
private:
    uint32_t        mDimX;
    uint32_t        mDimY;
    uint32_t        mDimZ;

    uint32_t        mExtendedDimX;
    uint32_t        mExtendedDimY;

    MagGrid         mExtendedMag;
    ImageMagGrid    mExtendedMagImage;
    MagGrid         mResult;

    ImageMagGrid    mTestMult;
    ImageMagGrid    mTestMultInv;
};
//-----------------------------//
#endif //SPINPMCALC_DEMAGFIELD_H
