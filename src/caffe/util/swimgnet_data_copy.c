/***********************************************
 * Created by Xin You
 * Date: 2017/8/21
 * slave functions used by imagenet_data_layer
 * ********************************************/
#include <assert.h>
#include <stdio.h>
#include "caffe/util/swimgnet_data_copy.h"
#include <athread.h>

typedef struct DataParam_st {
  unsigned char* data;
  void* top0;
  void* mean;
  int n_cols;
  int n_rows;
  int batch_size;
}DataParam;

extern SLAVE_FUN(sw_slave_imgnet_f)();
extern SLAVE_FUN(sw_slave_imgnet_d)();

void swimgnet_data_copy_f(unsigned char* _data, char* _labels, float* _top0, float* _top1, float* _mean, int _batch_size, int n_cols, int n_rows) {
  // setting param
  DataParam* param = (DataParam*)malloc(sizeof(DataParam));
  param -> data = _data;
  param -> top0 = _top0;
  param -> mean = _mean;
  param -> n_cols = n_cols;
  param -> n_rows = n_rows;
  param -> batch_size = _batch_size;
  // spawn
  int local_data_size = n_rows*(3*n_cols/64+1);
  assert(local_data_size*(sizeof(char)+2*sizeof(float))<64*1024);

  athread_spawn(sw_slave_imgnet_f,param);
  // copy labels while slave cores are copying data
  // for the batch size is usually less than 256   
  int img;
  for( img = 0; img < _batch_size; ++img  ){
    _top1[img] = (float)(_labels[img]);
  }
  athread_join();
  free(param);
}

void swimgnet_data_copy_d(unsigned char* _data, char* _labels, double* _top0, double* _top1, double* _mean, int _batch_size, int n_cols, int n_rows) {
  // setting param
  DataParam* param = (DataParam*)malloc(sizeof(DataParam));
  param -> data = _data;
  param -> top0 = _top0;
  param -> mean = _mean;
  param -> n_cols = n_cols;
  param -> n_rows = n_rows;
  param -> batch_size = _batch_size;
  // spawn
  int local_data_size = n_rows*(3*n_cols/64+1);
  assert(local_data_size*(sizeof(char)+2*sizeof(double))<64*1024);

  athread_spawn(sw_slave_imgnet_d,param);
  // copy labels while slave cores are copying data
  // for the batch size is usually less than 256   
  int img;
  for( img = 0; img < _batch_size; ++img  ){
    _top1[img] = (double)(_labels[img]);
  }
  athread_join();
  free(param);
}
