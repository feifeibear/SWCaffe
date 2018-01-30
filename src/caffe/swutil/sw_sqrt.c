/******************************************
 * Created by Liandeng Li
 * Date: 2017/10/5
 * Sqrt functions: (in SPEs)
 * 1. double: sw_sqrt_d(double* src, double* dst, int count)
 * 2. float : sw_sqrt_f(float * src, float * dst, int count)
 * ***************************************/
#include "caffe/util/sw_dnn.h"
#include "athread.h"

extern SLAVE_FUN(sw_slave_sqrt_d)();
extern SLAVE_FUN(sw_slave_sqrt_f)();
typedef struct SqrtTransPara_st {
  void *src;
  void *dst;
  int count;
}SqrtPara;
// Precondition: already athread_init()
void sw_sqrt_d(const double* src, double* dst,const int count) {
  SqrtPara *para = (SqrtPara*)malloc(sizeof(SqrtPara));
  para->src = src;
  para->dst = dst;
  para->count = count;
  athread_spawn(sw_slave_sqrt_d,para);
  athread_join();
  free(para);
}
void sw_sqrt_f(const float* src, float* dst,const int count) {
  SqrtPara *para = (SqrtPara*)malloc(sizeof(SqrtPara));
  para->src = src;
  para->dst = dst;
  para->count = count;
  athread_spawn(sw_slave_sqrt_f,para);
  athread_join();
  free(para);
}
