#pragma once

#include <ceres/autodiff_cost_function.h>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

struct StampResidual
{
  StampResidual(double source_stamp, double target_stamp) : source_stamp(source_stamp), target_stamp(target_stamp) {}

  template <typename T>
  bool operator()(
    const T * const source_offset, const T * const target_offset, T * residuals) const
  {
    // (target + target_offset) - (source + source_offset)
    // (target - source) + target_offset - source_offset
    residuals[0] = T(target_stamp - source_stamp) + target_offset[0] - source_offset[0];
    return true;
  }

  static ceres::CostFunction * createResidual(
    double source_stamp, double target_stamp)
  {
    return new ceres::AutoDiffCostFunction<StampResidual, 1, 1, 1>(new StampResidual(source_stamp, target_stamp));
  }

  double source_stamp;
  double target_stamp;
};
