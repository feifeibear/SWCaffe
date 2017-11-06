/******************************************
 * Created by Xin You
 * Date: 2017/8/7
 * Memory copy functions: (in SPEs)
 * 1. double: sw_memcpy_d(double* src, double* dst, int count)
 * 2. float : sw_memcpy_f(float * src, float * dst, int count)
 * ***************************************/
#include "caffe/util/sw_memcpy.h"
#include "athread.h"

extern SLAVE_FUN(sw_slave_memcpy_d)();
extern SLAVE_FUN(sw_slave_memcpy_f)();
typedef struct MemcpyTransPara_st {
  void *src;
  void *dst;
  int count;
}MemcpyPara;
// Precondition: already athread_init()
void sw_memcpy_d(double* src, double* dst, int count) {
  MemcpyPara *para = (MemcpyPara*)malloc(sizeof(MemcpyPara));
  para->src = src;
  para->dst = dst;
  para->count = count;
  athread_spawn(sw_slave_memcpy_d,para);
  athread_join();
  free(para);
}
void sw_memcpy_f(float* src, float* dst, int count) {
  MemcpyPara *para = (MemcpyPara*)malloc(sizeof(MemcpyPara));
  para->src = src;
  para->dst = dst;
  para->count = count;
  athread_spawn(sw_slave_memcpy_f,para);
  athread_join();
  free(para);
}
