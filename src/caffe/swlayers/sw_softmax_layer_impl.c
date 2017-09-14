/********************************************
 * Created by Xin You
 * Date: 2017/8/24
 * softmax layer interface for acc version.
 * *****************************************/
#include <stdio.h>
#include <assert.h>
#include <athread.h>
#include <math.h>
#include "caffe/swlayers/sw_softmax_layer_impl.h"

//#define DEBUG_INFO
//#define MPE_TRANS

extern SLAVE_FUN(swsoftmax_trans_f)();
extern SLAVE_FUN(swsofmax_f)();
//extern SLAVE_FUN(swsofmax_d)();

typedef struct TransData_st {
  void* in;
  void* out;
  int tZ;
  int tX;
  int tY;
}TransData;

typedef struct SoftmaxData_st{
  void* bottom_data;
  void* sum_multiplier_;
  void* scale_data;
  void* top_data;
  int channels;
  int dim;
  int outer_num_;
  int inner_num_;
}SoftmaxData;

void sw_softmax_forward_impl_f(
    const float* bottom_data,
    const float* sum_multiplier_,
    float* scale_data,
    float* top_data,
    int channels,
    int dim,
    int outer_num_,
    int inner_num_) {
#ifdef DEBUG_INFO
  printf("channels = %d, dim = %d, outer_num_ = %d, inner_num_ = %d\n",channels,dim,outer_num_,inner_num_);
  /*int testArr[80];
  int testArr__[80];
  int tt, ii,ij,ik;
  for(tt = 0;tt<80;tt++) testArr[tt] = tt;
  TransData tdata;
  tdata.in=testArr;
  tdata.out=testArr__;
  tdata.tX = 4;
  tdata.tY = 5;
  tdata.tZ = 4;
  athread_spawn(swsoftmax_trans_f,&tdata);
  athread_join();
  for(ik=0;ik<4;++ik) {
  for(ii=0;ii<4;++ii){
    for(ij=0;ij<5;++ij) {
      printf("%d ",testArr__[ik*16+ii*4+ij]);
    }
    printf("\n\t");
  }
  printf("\n");
  }*/
#endif
  assert(dim==channels*inner_num_);
  int i,j,k;
  float* bottom_data_T = (float*)malloc(sizeof(float)*outer_num_*dim);
  float* top_data_T = (float*)malloc(sizeof(float)*outer_num_*dim);
  // matrix trans
#ifdef USE_SWSOFTMAX
  for(i=0; i < outer_num_;++i) {
    for(j=0;j < channels;++j) {
      for(k=0;k < inner_num_;++k) {
        bottom_data_T[i*dim+k*channels+j] = bottom_data[i*dim+j*inner_num_+k];
      }
    }
  }
#else
  TransData* tpara = (TransData*)malloc(sizeof(TransData));
  tpara->in = bottom_data;
  tpara->out= bottom_data_T;
  tpara->tZ = outer_num_;
  tpara->tY = channels;
  tpara->tX = inner_num_;
  athread_spawn(swsoftmax_trans_f,tpara);
  athread_join();
  free(tpara);
#endif
  SoftmaxData* param = (SoftmaxData*)malloc(sizeof(SoftmaxData));
  param->bottom_data = bottom_data_T;
  param->sum_multiplier_ = sum_multiplier_;
  param->scale_data = scale_data;
  param->top_data = top_data_T;
  param->channels = channels;
  param->dim = dim;
  param->outer_num_ = outer_num_;
  param->inner_num_ = inner_num_;
  assert(channels+2*channels*(inner_num_/64+1)<64*1024/sizeof(float));
  athread_spawn(swsofmax_f,param);
  athread_join();
  free(param);
  free(bottom_data_T);
 // printf("matrix trans back\n");
  // matrix trans back
#ifdef USE_SWSOFTMAX
  for(i=0; i < outer_num_;++i) {
    for(j=0;j < channels;++j) {
      for(k=0;k < inner_num_;++k) {
        top_data[i*dim+j*inner_num_+k] = top_data_T[i*dim+k*channels+j];
      }
    }
  }
#else
  tpara = (TransData*)malloc(sizeof(TransData));
  tpara->in = top_data_T;
  tpara->out= top_data;
  tpara->tZ = outer_num_;
  tpara->tY = inner_num_;
  tpara->tX = channels;
  athread_spawn(swsoftmax_trans_f,tpara);
  athread_join();
  free(tpara);
#endif
  free(top_data_T);
  //printf("fin\n");
}
