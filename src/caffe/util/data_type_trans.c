#include "caffe/util/data_type_trans.h"
#include "athread.h"

extern SLAVE_FUN(sw_double2float)();
extern SLAVE_FUN(sw_float2double)();
typedef struct DataTransPara_st {
  void *src;
  void *dst;
  int count;
}DataTransPara;
// Precondition: already athread_init()
void double2float(double* src, float* dst, int count) {
  DataTransPara *para = (DataTransPara*)malloc(sizeof(DataTransPara));
  para->src = src;
  para->dst = dst;
  para->count = count;
  athread_spawn(sw_double2float,para);
  athread_join();
  free(para);
}
void float2double(float* src, double* dst, int count) {
  DataTransPara *para = (DataTransPara*)malloc(sizeof(DataTransPara));
  para->src = src;
  para->dst = dst;
  para->count = count;
  athread_spawn(sw_float2double,para);
  athread_join();
  free(para);
}
