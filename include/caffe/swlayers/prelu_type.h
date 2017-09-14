#ifndef RELU_TYPE_H_
#define RELU_TYPE_H_
// define buffer size for relu data
// 1: double buffer for in and out
//      buffersize < 64KB / 4 = 16KB
//    single buffer for in and out
//      buffersize < 64KB / 2 = 32KB
// 2: double buffer for in, diff and out
//      buffersize < 64KB / 6 < 10.67KB
//    single buffer for in, diff and out
//      buffersize < 64KB / 3 = 21.33KB
// now single buffer
#define __PRELU_BUFFSIZE_1 2*1024
#define __PRELU_BUFFSIZE_2 2*1024

typedef struct PReluData_st {
  // 0
  void* in;
  // 8
  void* out;
  // 16
  void* slope_data;
  // 24
  int count;
  int dim;
  int channels;
  int div_factor;

}PReluData;

typedef struct PReluDiffData_st {
  // 0
  void* in;
  // 8
  void* diff;
  // 16
  void* out;
  // 24
  void* slope_data;
  // 32
  void* slope_diff;
  int count;
  int dim;
  int channels;
  int div_factor;

}PReluDiffData;
#endif

