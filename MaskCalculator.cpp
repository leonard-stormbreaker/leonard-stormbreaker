//
// Created by devilox on 2/13/22.
//

#include "MaskCalculator.h"

MaskCalculator::MaskCalculator( double tGridLeft,   double tGridRight,
                                double tGridUp,     double tGridDown,
                                double tGridStepX,  double tGridStepY) :
        mGridLeft(tGridLeft), mGridRight(tGridRight),
        mGridUp(tGridUp), mGridDown(tGridDown),
        mGridStepX(tGridStepX), mGridStepY(tGridStepY) {
    mDimX = std::ceil((tGridRight - tGridLeft) / mGridStepX);
    mDimY = std::ceil((tGridDown - tGridUp) / mGridStepY);
}

MaskCalculator::MaskGrid MaskCalculator::calculateMask( Mask::Type tType,
                                                        double tMaskCenterX,    double tMaskCenterY,
                                                        double tMaskWidth,      double tMaskHeight,
                                                        size_t& tMagnCellNum) const {
    MaskGrid Result(mDimX);

    for (auto& iRow : Result) {
        iRow.resize(mDimY, 0);
    }

    double x;
    double y;

    if (tType == Mask::Type::ELLIPSE) {
        double Ellipse;

        for (int i = 0; i < mDimX; i++) {
            for (int j = 0; j < mDimY; j++) {
                x = i * mGridStepX + mGridLeft;
                y = j * mGridStepY + mGridUp;

                Ellipse =
                        std::pow(x - tMaskCenterX, 2.0) / std::pow(tMaskWidth, 2.0) * 4.0 +
                        std::pow(y - tMaskCenterY, 2.0) / std::pow(tMaskHeight, 2.0) * 4.0;

                if (Ellipse <= 1) {
                    Result[i][j] = 1;
                    tMagnCellNum++;
                }
            }
        }
    } else {
        for (int i = 0; i < mDimX; i++) {
            for (int j = 0; j < mDimY; j++) {
                x = i * mGridStepX + mGridLeft;
                y = j * mGridStepY + mGridUp;

                if (x - tMaskCenterX > tMaskCenterX - tMaskWidth * 0.5 &&
                    x - tMaskCenterX < tMaskCenterX + tMaskWidth * 0.5 &&
                    y - tMaskCenterY > tMaskCenterY - tMaskHeight * 0.5 &&
                    y - tMaskCenterY < tMaskCenterY + tMaskHeight * 0.5) {
                    Result[i][j] = 1;
                    tMagnCellNum++;
                }
            }
        }
    }

    return Result;
}

size_t MaskCalculator::getDimX() const {
    return mDimX;
}
size_t MaskCalculator::getDimY() const {
    return mDimY;
}