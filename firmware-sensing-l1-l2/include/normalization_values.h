#ifndef NORM_H
#define NORM_H
// NOTE: Thermal images must be Zero-Centered PER FRAME in C++ code.
// formula: pixel = pixel - frame_average

const float SCALAR_MEAN[]={129046.0391,0.0378,-0.0052,965244.5000,3.7879,0.0003,0.0012,0.0015};
const float SCALAR_STD[]={995249.6875,0.4591,0.1536,1411383.0000,5.1649,0.5735,0.0476,0.0477};
#endif