#pragma once

#include <pcl/point_types.h>

struct EIGEN_ALIGN16 PointXYZRelStamp
{
  PCL_ADD_POINT4D;
  std::uint32_t time_stamp;
};

struct EIGEN_ALIGN16 PointXYZAbsStamp
{
  PCL_ADD_POINT4D;
  double time_stamp;
};

POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZRelStamp,
    (float, x, x)(float, y, y)(float, z, z)(std::uint32_t, time_stamp, time_stamp))

POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZAbsStamp,
    (float, x, x)(float, y, y)(float, z, z)(double, time_stamp, time_stamp))