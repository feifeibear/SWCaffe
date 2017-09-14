#include <stdio.h>
#include <assert.h>
#include "athread.h"
#include <math.h>
#include "caffe/swlayers/sw_relu_layer_impl.h"
#include "caffe/swlayers/relu_type.h"

#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)<(b)?(a):(b))

extern SLAVE_FUN(relu_slave_forward_f)();
extern SLAVE_FUN(relu_slave_forward_d)();
extern SLAVE_FUN(relu_slave_backward_f)();
extern SLAVE_FUN(relu_slave_backward_d)();

void sw_relu_forward_impl_f(
        const float* in,
        float* out,
        float negative_slope,
        int count) {
#ifdef NOSPE
  int i;
  for( i = 0; i < count; ++i) {
    out[i] = max(in[i], 0.0)
              + negative_slope * min(in[i], 0.0);
  }
#else
  ReluData* param = (ReluData*)malloc(sizeof(ReluData));
  param->in = in;
  param->out= out;
  param->negative_slope.flt=negative_slope;
  param->count=count;
  int ldm_consume = 4*__RELU_BUFFSIZE_1*2;
  assert(ldm_consume < 64*1024);
  athread_spawn(relu_slave_forward_f,param);
  athread_join();
  free(param);
#endif
}

void sw_relu_forward_impl_d(
        const double* in,
        double* out,
        double negative_slope,
        int count) {
//#define NOSPE
#ifdef NOSPE
  int i;
  for( i = 0; i < count; ++i) {
    out[i] = max(in[i], 0.0)
              + negative_slope * min(in[i], 0.0);
  }
#else
  ReluData* param = (ReluData*)malloc(sizeof(ReluData));
  param->in = in;
  param->out= out;
  param->negative_slope.dbl=negative_slope;
  param->count=count;
  int ldm_consume = 8*__RELU_BUFFSIZE_1*2;
  assert(ldm_consume < 64*1024);
  athread_spawn(relu_slave_forward_d,param);
  athread_join();
  free(param);
#endif
//#undef NOSPE
}

void sw_relu_backward_impl_f(
        const float* bottom_data,
        const float* top_diff,
        float* bottom_diff,
        float negative_slope,
        int count) {
#ifdef NOSPE
  int i;
  for( i = 0; i < count; ++i) {
    bottom_diff[i] = top_diff[i] * ((bottom_data[i] > 0)
          + negative_slope * (bottom_data[i] <= 0));
  }
#else
  ReluDiffData* param = (ReluDiffData*)malloc(sizeof(ReluDiffData));
  param->in = bottom_data;
  param->diff = top_diff;
  param->out= bottom_diff;
  param->negative_slope.flt=negative_slope;
  param->count=count;
  int ldm_consume = 4*__RELU_BUFFSIZE_2*3;
  assert(ldm_consume < 64*1024);
  athread_spawn(relu_slave_backward_f,param);
  athread_join();
  free(param);
#endif
}

void sw_relu_backward_impl_d(
        const double* bottom_data,
        const double* top_diff,
        double* bottom_diff,
        double negative_slope,
        int count) {
//#define NOSPE
#ifdef NOSPE
  int i;
  for( i = 0; i < count; ++i) {
    bottom_diff[i] = top_diff[i] * ((bottom_data[i] > 0)
          + negative_slope * (bottom_data[i] <= 0));
  }
#else
  ReluDiffData* param = (ReluDiffData*)malloc(sizeof(ReluDiffData));
  param->in = bottom_data;
  param->diff = top_diff;
  param->out= bottom_diff;
  param->negative_slope.dbl=negative_slope;
  param->count=count;
  int ldm_consume = 8*__RELU_BUFFSIZE_2*3;
  assert(ldm_consume < 64*1024);
  athread_spawn(relu_slave_backward_d,param);
  athread_join();
  free(param);
#endif
//#undef NOSPE
}
