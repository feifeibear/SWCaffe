/******************************************
 * Created by Liandeng Li
 * Date: 2017/10/5
 * Scale  functions: (in SPEs)
 * 1. double: sw_scale_d(double* src,double *scale, double* dst, int outer_dim,int inner_dim,int scale_dim)
 * 2. float : sw_scale_f(float * src,float *scale, float * dst, int outer_dim,int inner_dim,int scale_dim)
 * ***************************************/
#include "caffe/util/sw_dnn.h"
#include "athread.h"

extern SLAVE_FUN(sw_slave_scale_d)();
extern SLAVE_FUN(sw_slave_scale_f)();
typedef struct ScaleTransPara_st {
  void *src;
  void *scale;
  void *dst;
  int outer_dim,inner_dim,scale_dim;
}ScalePara;
// Precondition: already athread_init()
void sw_scale_layer_d(const double* src,const double *scale, double* dst, const int outer_dim,const int inner_dim,const int scale_dim) {
  ScalePara *para = (ScalePara*)malloc(sizeof(ScalePara));
  para->src = src;
  para->scale = scale;
  para->dst = dst;
  para->outer_dim = outer_dim;
  para->inner_dim = inner_dim;
  para->scale_dim = scale_dim;
  athread_spawn(sw_slave_scale_d,para);
  athread_join();
  free(para);
}
void sw_scale_layer_f(const float* src,const float *scale, float* dst,const int outer_dim,const int inner_dim,const int scale_dim) {
  ScalePara *para = (ScalePara*)malloc(sizeof(ScalePara));
  para->src = src;
  para->scale = scale;
  para->dst = dst;
  para->outer_dim = outer_dim;
  para->inner_dim = inner_dim;
  para->scale_dim = scale_dim;
  athread_spawn(sw_slave_scale_f,para);
  athread_join();
  free(para);
}
