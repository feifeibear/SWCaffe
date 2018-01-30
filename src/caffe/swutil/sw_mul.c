/******************************************
 * Created by Liandeng Li
 * Date: 2017/10/5
 * Multify  functions: (in SPEs)
 * 1. double: sw_mul_d(double* src1,double *src2, double* dst, int count)
 * 2. float : sw_mul_f(float * src1,float *src2, float * dst, int count)
 * ***************************************/
#include "caffe/util/sw_dnn.h"
#include "athread.h"

extern SLAVE_FUN(sw_slave_mul_d)();
extern SLAVE_FUN(sw_slave_mul_f)();
typedef struct MulTransPara_st {
  void *src1;
  void *src2;
  void *dst;
  int count;
}MulPara;
// Precondition: already athread_init()
void sw_mul_d(const double* src1,const double *src2, double* dst,const  int count) {
  MulPara *para = (MulPara*)malloc(sizeof(MulPara));
  para->src1 = src1;
  para->src2 = src2;
  para->dst = dst;
  para->count = count;
  athread_spawn(sw_slave_mul_d,para);
  athread_join();
  free(para);
}
void sw_mul_f(const float* src1,const float *src2, float* dst,const int count) {
  MulPara *para = (MulPara*)malloc(sizeof(MulPara));
  para->src1 = src1;
  para->src2 = src2;
  para->dst = dst;
  para->count = count;
  athread_spawn(sw_slave_mul_f,para);
  athread_join();
  free(para);
}
