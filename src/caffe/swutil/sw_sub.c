/******************************************
 * Created by Liandeng Li
 * Date: 2017/10/5
 * subtify  functions: (in SPEs)
 * 1. double: sw_sub_d(double* src1,double *src2, double* dst, int count)
 * 2. float : sw_sub_f(float * src1,float *src2, float * dst, int count)
 * ***************************************/
#include "caffe/util/sw_dnn.h"
#include "athread.h"

//extern SLAVE_FUN(sw_slave_net_param_sub_d)();
extern SLAVE_FUN(sw_slave_net_param_sub_f)();
//extern SLAVE_FUN(sw_slave_sub_d)();
extern SLAVE_FUN(sw_slave_sub_f)();
typedef struct subTransPara_st {
  void *src1;
  void *src2;
  void *dst;
  int count,numprocs;
}subPara;
// Precondition: already athread_init()
void sw_sub_d(const double* src1,const double *src2, double* dst, const int count) {
  subPara *para = (subPara*)malloc(sizeof(subPara));
  para->src1 = src1;
  para->src2 = src2;
  para->dst = dst;
  para->count = count;
  //athread_spawn(sw_slave_sub_d,para);
  //athread_join();
  free(para);
}
void sw_sub_f(const float* src1,const float *src2, float* dst, const int count) {
  subPara *para = (subPara*)malloc(sizeof(subPara));
  para->src1 = src1;
  para->src2 = src2;
  para->dst = dst;
  para->count = count;
  athread_spawn(sw_slave_sub_f,para);
  athread_join();
  free(para);
}
void sw_net_param_sub_f(const int count,const int numprocs,float * data,float * diff) {
  subPara *para = (subPara*)malloc(sizeof(subPara));
  para->src1 = data;
  para->src2 = diff;
  para->numprocs = numprocs;
  para->count = count;
  athread_spawn(sw_slave_net_param_sub_f,para);
  athread_join();
  free(para);
}
void sw_net_param_sub_d(const int count,const int numprocs,double * data,double * diff) {
  subPara *para = (subPara*)malloc(sizeof(subPara));
  para->src1 = data;
  para->src2 = diff;
  para->numprocs = numprocs;
  para->count = count;
  //athread_spawn(sw_slave_net_param_sub_d,para);
  //athread_join();
  free(para);
}
