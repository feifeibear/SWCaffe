/******************************************
 * Created by Xin You
 * Date: 2017/8/7
 * Memory copy functions: (in SPEs)
 * 1. double: sw_memcpy_d(double* src, double* dst, int count)
 * 2. float : sw_memcpy_f(float * src, float * dst, int count)
 * ***************************************/
#include "caffe/util/sw_dnn.h"
#include "athread.h"

extern SLAVE_FUN(sw_slave_weights_memcpy_f)();
extern SLAVE_FUN(sw_slave_weights_memcpy_d)();
extern SLAVE_FUN(sw_slave_memcpy_d)();
extern SLAVE_FUN(sw_slave_memcpy_f)();
extern SLAVE_FUN(sw_slave_memcpy_i)();
extern SLAVE_FUN(sw_slave_memcpy_ui)();
typedef struct MemcpyTransPara_st {
  void *src;
  void *dst;
  int count,num;
}MemcpyPara __attribute__((__aligned__(16)));
// Precondition: already athread_init()
void sw_repeat_memcpy_f(const float* src, float* dst,const int count,int num) {
  printf("Alert: Unchecked implementation of sw_repeat_memcpy_f()\n");
  MemcpyPara *para = (MemcpyPara*)malloc(sizeof(MemcpyPara));
  para->src = src;
  para->dst = dst;
  para->count = count;
  para->num = num;
  athread_spawn(sw_slave_weights_memcpy_f,para);
  athread_join();
  free(para);
}
void sw_repeat_memcpy_d(const double* src, double* dst,const int count,int num) {
  printf("Alert: Unchecked implementation of sw_repeat_memcpy_d()\n");
  MemcpyPara *para = (MemcpyPara*)malloc(sizeof(MemcpyPara));
  para->src = src;
  para->dst = dst;
  para->count = count;
  para->num = num;
  athread_spawn(sw_slave_weights_memcpy_f,para);
  athread_join();
  free(para);
}
void sw_memcpy_d(const double* src, double* dst,const int count) {
  printf("Alert: Unchecked implementation of sw_memcpy_d()\n");
  MemcpyPara *para = (MemcpyPara*)malloc(sizeof(MemcpyPara));
  para->src = src;
  para->dst = dst;
  para->count = count;
  athread_spawn(sw_slave_memcpy_d,para);
  athread_join();
  free(para);
}
void sw_memcpy_f(const float* src, float* dst,const int count) {
  //modified by zwl
  if(count < 256){
    memcpy(dst, src, (long)count*sizeof(float));
  }else{
    MemcpyPara *para = (MemcpyPara*)malloc(sizeof(MemcpyPara));
    para->src = src+count%256;
    para->dst = dst+count%256;
    para->count = count-count%256;
    athread_spawn(sw_slave_memcpy_f,para);
    if(count % 256 != 0){
      memcpy(dst, src, (long)(count%256)*sizeof(float));
    }
    athread_join();
    free(para); 
  } 
}
void sw_memcpy_i(const int* src, int* dst,const int count) {
  printf("Alert: Unchecked implementation of sw_memcpy_i()\n");
  MemcpyPara *para = (MemcpyPara*)malloc(sizeof(MemcpyPara));
  para->src = src;
  para->dst = dst;
  para->count = count;
  athread_spawn(sw_slave_memcpy_i,para);
  athread_join();
  free(para);
}
void sw_memcpy_ui(const unsigned int* src,unsigned int* dst,const int count) {
  printf("Alert: Unchecked implementation of sw_memcpy_ui()\n");
  MemcpyPara *para = (MemcpyPara*)malloc(sizeof(MemcpyPara));
  para->src = src;
  para->dst = dst;
  para->count = count;
  athread_spawn(sw_slave_memcpy_ui,para);
  athread_join();
  free(para);
}
