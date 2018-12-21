/******************************************
 * Created by Liandeng Li
 * Date: 2017/10/5
 * addtify  functions: (in SPEs)
 * 1. double: sw_add_d(double* src1,double *src2, double* dst, int count)
 * 2. float : sw_add_f(float * src1,float *src2, float * dst, int count)
 * ***************************************/
#include "athread.h"
#include "simd.h"
extern SLAVE_FUN(sw_slave_add_f_allreduce)();
typedef struct addTransPara_st {
  void *src1;
  void *src2;
  void *dst;
  int count;
} addPara;
// Precondition: already athread_init()
void thread_init(){
  athread_init();
}
void sw_add_f_allreduce(float* src1,float *src2, float* dst,const int count) {
  int min_size = 1024*32;
  /*float * tmpbuff = (float*)malloc(count*sizeof(float));
  int i= 0;
  for(i=0;i<count;i++)
      tmpbuff[i] = src1[i]+src2[i];
  */
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
  } else {
    /*addPara para;*/
    // modified by zhuchuanjia
    addPara *para = (addPara *)malloc(sizeof(addPara));
    para->src1 = src1;
    para->src2 = src2;
    para->dst = dst;
    para->count = count;
    /*para.src1 = src1;*/
    /*para.src2 = src2;*/
    /*para.dst = dst;*/
    /*para.count = count;*/
    athread_spawn(sw_slave_add_f_allreduce,para);
    athread_join();
    free(para);
  }
  /*int j=0;
  for(i=0;i<count;i++){
    if(fabs(tmpbuff[i] - dst[i]) > 1e-4)
      if(j++<5)
      {
        printf("count = %d i=%d host= %f slave=%f\n",count,i,tmpbuff[i],dst[i]);
      }
      else break;
  }
  free(tmpbuff);*/
}
