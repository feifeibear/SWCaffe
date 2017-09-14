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
#define __RELU_BUFFSIZE_1 2*1024
#define __RELU_BUFFSIZE_2 2*1024

typedef union SlopeType_un {
  float flt;
  double dbl;
}SlopeType;

typedef struct ReluData_st {
  // 0
  void* in;
  // 8
  void* out;
  // 16
  SlopeType negative_slope;
  // 24
  int count;

}ReluData;

typedef struct ReluDiffData_st {
  // 0
  void* in;
  // 8
  void* diff;
  // 16
  void* out;
  // 24
  SlopeType negative_slope;
  // 32
  int count;

}ReluDiffData;
#endif

