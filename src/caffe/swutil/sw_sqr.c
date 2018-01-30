/******************************************
 * Created by Liandeng Li
 * Date: 2017/10/5
 * Sqrt functions: (in SPEs)
 * 1. double: sw_sqr_d(double* src, double* dst, int count)
 * 2. float : sw_sqr_f(float * src, float * dst, int count)
 * ***************************************/
#include "caffe/util/sw_dnn.h"
#include "athread.h"

extern SLAVE_FUN(sw_slave_sqr_d)();
extern SLAVE_FUN(sw_slave_sqr_f)();
typedef struct SqrTransPara_st {
  void *src;
  void *dst;
  int count;
}SqrPara;
// Precondition: already athread_init()
void sw_sqr_d(const double* src, double* dst,const int count) {
  SqrPara *para = (SqrPara*)malloc(sizeof(SqrPara));
  para->src = src;
  para->dst = dst;
  para->count = count;
  athread_spawn(sw_slave_sqr_d,para);
  athread_join();
  free(para);
}
void sw_sqr_f(const float* src, float* dst,const int count) {
  SqrPara *para = (SqrPara*)malloc(sizeof(SqrPara));
  para->src = src;
  para->dst = dst;
  para->count = count;
  athread_spawn(sw_slave_sqr_f,para);
  athread_join();
  free(para);
}
