/******************************************
 * Created by Liandeng Li
 * Date: 2017/10/5
 * addtify  functions: (in SPEs)
 * 1. double: sw_add_d(double* src1,double *src2, double* dst, int count)
 * 2. float : sw_add_f(float * src1,float *src2, float * dst, int count)
 * ***************************************/
#include "caffe/util/sw_dnn.h"
#include "athread.h"
#include "simd.h"
extern SLAVE_FUN(sw_slave_add_d)();
extern SLAVE_FUN(sw_slave_add_f)();
typedef struct addTransPara_st {
  void *src1;
  void *src2;
  void *dst;
  int count;
}addPara;
// Precondition: already athread_init()
void sw_add_d(const double* src1,const double *src2, double* dst, const int count) {
  addPara *para = (addPara*)malloc(sizeof(addPara));
  para->src1 = src1;
  para->src2 = src2;
  para->dst = dst;
  para->count = count;
  athread_spawn(sw_slave_add_d,para);
  athread_join();
  free(para);
}
void sw_add_f(const float* src1,const float *src2, float* dst,const int count) {
  int min_size = 8192;
  if(count < min_size)
  {
    int simdsize = 4,i=0;
    floatv4 vc1,vc2;
    for(i=0;i+simdsize-1<count;i+=simdsize)
    {
      simd_load(vc1,src1+i);
      simd_load(vc2,src2+i);
      vc1 = vc1 + vc2;
      simd_store(vc1,dst+i);
    }
    for(;i<count;i++)
      dst[i] = src1[i]+src2[i];
    return;
  }
  addPara *para = (addPara*)malloc(sizeof(addPara));
  para->src1 = src1;
  para->src2 = src2;
  para->dst = dst;
  para->count = count;
  athread_spawn(sw_slave_add_f,para);
  athread_join();
  free(para);
}
