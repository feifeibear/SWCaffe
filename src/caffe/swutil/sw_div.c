/******************************************
 * Created by Liandeng Li
 * Date: 2017/10/5
 * Division  functions: (in SPEs)
 * 1. double: sw_div_d(double* src, double* dst, int count)
 * 2. float : sw_div_f(float * src, float * dst, int count)
 * ***************************************/
#include "caffe/util/sw_dnn.h"
#include "athread.h"

extern SLAVE_FUN(sw_slave_div_d)();
extern SLAVE_FUN(sw_slave_div_f)();
typedef struct DivTransPara_st {
  void *src1;
  void *src2;
  void *dst;
  int count;
}DivPara;
// Precondition: already athread_init()
void sw_div_d(const double* src1,const double *src2, double* dst,const int count) {
  DivPara *para = (DivPara*)malloc(sizeof(DivPara));
  para->src1 = src1;
  para->src2 = src2;
  para->dst = dst;
  para->count = count;
  athread_spawn(sw_slave_div_d,para);
  athread_join();
  free(para);
}
void sw_div_f(const float* src1,const float *src2, float* dst,const int count) {
  DivPara *para = (DivPara*)malloc(sizeof(DivPara));
  para->src1 = src1;
  para->src2 = src2;
  para->dst = dst;
  para->count = count;
  athread_spawn(sw_slave_div_f,para);
  athread_join();
  free(para);
}
