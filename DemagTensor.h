//
// Created by devilox on 11/2/21.
//
//-----------------------------//
#ifndef SPINPMCALC_DEMAGTENSOR_H
#define SPINPMCALC_DEMAGTENSOR_H
//-----------------------------//
#include <cmath>
#include <vector>
#include <array>
#include <iostream>
#include <future>
#include <complex>
//-----------------------------//
#include <fftw3.h>
//-----------------------------//
class DemagTensor {
private:
    using TensorGrid = std::vector <std::vector <std::vector <std::array <double, 9>>>>;
    using ImageGrid = std::vector <std::vector <std::vector <std::array <std::complex <double>, 9>>>>;
public:
    DemagTensor(uint32_t tDimX, uint32_t tDimY, uint32_t tDimZ, double tDeltaX, double tDeltaY);

    void calcTensor(const std::vector <double>& tDeltaZ, const std::vector <std::vector <std::vector <int8_t>>>& tMask);
    void calcImage();

    [[nodiscard]] const TensorGrid& getTensor() const;
    [[nodiscard]] const ImageGrid& getImage() const;
public:
    uint32_t        mDimX;
    uint32_t        mDimY;
    uint32_t        mDimZ;

    uint32_t        mExtendedDimX;
    uint32_t        mExtendedDimY;
    uint32_t        mExtendedDimZ;

    double          mDeltaX;
    double          mDeltaY;

    TensorGrid      mExtendedTensor;
    ImageGrid       mExtendedTensorImage;

    std::vector <std::vector <std::vector <std::array <fftw_complex, 9>>>>      mTestImage;

    //----------//

    [[nodiscard]] static double fxx(    double x,           double y,               double z);
    [[nodiscard]] static double F1xx(   double x,           double y,               double z,
                                        double tDeltaY,     double tDeltaYSource);
    [[nodiscard]] static double Fxx(    double x,           double y,               double z,
                                        double tDeltaY,     double tDeltaYSource,
                                        double tDeltaZ,     double tDeltaZSource);
    [[nodiscard]] static double Nxx(    double x,           double y,               double z,
                                        double tDeltaX,     double tDeltaXSource,
                                        double tDeltaY,     double tDeltaYSource,
                                        double tDeltaZ,     double tDeltaZSource);
    [[nodiscard]] static double Nyy(    double x,           double y,               double z,
                                        double tDeltaX,     double tDeltaXSource,
                                        double tDeltaY,     double tDeltaYSource,
                                        double tDeltaZ,     double tDeltaZSource);
    [[nodiscard]] static double Nzz(    double x,           double y,               double z,
                                        double tDeltaX,     double tDeltaXSource,
                                        double tDeltaY,     double tDeltaYSource,
                                        double tDeltaZ,     double tDeltaZSource);

    //----------//

    [[nodiscard]] static double gxy(    double x,           double y,               double z);
    [[nodiscard]] static double G1xy(   double x,           double y,               double z,
                                        double deltaZ,      double deltaZSource);
    [[nodiscard]] static double Gxy(    double x,           double y,               double z,
                                        double tDeltaX,     double tDeltaYSource,
                                        double deltaZ,      double deltaZSource);
    [[nodiscard]] static double Nxy(    double x,           double y,               double z,
                                        double tDeltaX,     double tDeltaXSource,
                                        double tDeltaY,     double tDeltaYSource,
                                        double deltaZ,      double deltaZSource);
    [[nodiscard]] static double Nxz(    double x,           double y,               double z,
                                        double tDeltaX,     double tDeltaXSource,
                                        double tDeltaY,     double tDeltaYSource,
                                        double deltaZ,      double deltaZSource);
    [[nodiscard]] static double Nyz(    double x,           double y,               double z,
                                        double tDeltaX,     double tDeltaXSource,
                                        double tDeltaY,     double tDeltaYSource,
                                        double deltaZ,      double deltaZSource);
};
//-----------------------------//
#endif //SPINPMCALC_DEMAGTENSOR_H
