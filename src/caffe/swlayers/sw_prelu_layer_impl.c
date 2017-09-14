#include <stdio.h>
#include <assert.h>
#include "athread.h"
#include <math.h>
#include "caffe/swlayers/sw_prelu_layer_impl.h"
#include "caffe/swlayers/prelu_type.h"

#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)<(b)?(a):(b))

extern SLAVE_FUN(prelu_slave_forward_f)();
extern SLAVE_FUN(prelu_slave_forward_d)();
extern SLAVE_FUN(prelu_slave_backward_f)();
extern SLAVE_FUN(prelu_slave_backward_d)();
//extern SLAVE_FUN(prelu_slave_backward_both_f)();
//extern SLAVE_FUN(prelu_slave_backward_both_d)();
extern SLAVE_FUN(prelu_slave_backward_para_f)();
extern SLAVE_FUN(prelu_slave_backward_para_d)();

void sw_prelu_forward_impl_f(
        const float* in,
        float* out,
        float* slope_data,
        int count,
        int dim,
        int channels,
        int div_factor) {
#ifdef NOSPE
  int i;
  for( i = 0; i < count; ++i) {
    int c = (i / dim) % channels / div_factor;
    out[i] = max(in[i], 0.0)
              + slope_data[c] * min(in[i], 0.0);
  }
#else
  PReluData* param = (PReluData*)malloc(sizeof(PReluData));
  param->in = in;
  param->out= out;
  param->slope_data=slope_data;
  param->count=count;
  param->dim = dim;
  param->channels = channels;
  param->div_factor = div_factor;
  //printf("%d %d %d\n",channels,dim,div_factor);
  int ldm_consume = 4*__PRELU_BUFFSIZE_1*2;
  assert(ldm_consume < 64*1024);
  athread_spawn(prelu_slave_forward_f,param);
  athread_join();
  free(param);
#endif
}

void sw_prelu_forward_impl_d(
        const double* in,
        double* out,
        double* slope_data,
        int count,
        int dim,
        int channels,
        int div_factor) {
//#define NOSPE
#ifdef NOSPE
  int i;
  for( i = 0; i < count; ++i) {
    int c = (i / dim) % channels / div_factor;
    out[i] = max(in[i], 0.0)
              + slope_data[c] * min(in[i], 0.0);
  }
#else
  PReluData* param = (PReluData*)malloc(sizeof(PReluData));
  param->in = in;
  param->out= out;
  param->slope_data=slope_data;
  param->count=count;
  param->dim = dim;
  param->channels = channels;
  param->div_factor = div_factor;
  int ldm_consume = 8*__PRELU_BUFFSIZE_1*2;
  assert(ldm_consume < 64*1024);
  athread_spawn(prelu_slave_forward_d,param);
  athread_join();
  free(param);
#endif
//#undef NOSPE
}

// TODO: divide the slave function into three functions: 
// 1. with both param_propagate_down and propagate_down
// 2. with only param_propagate_down
// 3. with only propagate_down

void sw_prelu_backward_impl_f(
        const float* bottom_data,
        const float* top_diff,
        float* bottom_diff,
        float* slope_data,
        float* slope_diff,
        int count,
        int dim,
        int channels,
        int div_factor,
        int param_propagate_down,
        int propagate_down) {
#ifdef NOSPE
  int i;
  for( i = 0; i < count; ++i) {
    int c = (i / dim) % channels / div_factor;
    bottom_diff[i] = top_diff[i] * ((bottom_data[i] > 0)
          + slope_data[c] * (bottom_data[i] <= 0));
  }
#else
  PReluDiffData* param = (PReluDiffData*)malloc(sizeof(PReluDiffData));
  param->in = bottom_data;
  param->diff = top_diff;
  param->out= bottom_diff;
  param->slope_data=slope_data;
  param->slope_diff=slope_diff;
  param->count=count;
  param->dim = dim;
  param->channels = channels;
  param->div_factor = div_factor;
  int ldm_consume = 4*__PRELU_BUFFSIZE_2*3;
  assert(ldm_consume < 64*1024);
  if(param_propagate_down) {
    athread_spawn(prelu_slave_backward_para_f,param);
    athread_join();
  }
  if(propagate_down) {
    athread_spawn(prelu_slave_backward_f,param);
    athread_join();
  }
  free(param);
#endif
}

void sw_prelu_backward_impl_d(
        const double* bottom_data,
        const double* top_diff,
        double* bottom_diff,
        double* slope_data,
        double* slope_diff,
        int count,
        int dim,
        int channels,
        int div_factor,
        int param_propagate_down,
        int propagate_down) {
//#define NOSPE
#ifdef NOSPE
  int i;
  for( i = 0; i < count; ++i) {
    int c = (i / dim) % channels / div_factor;
    bottom_diff[i] = top_diff[i] * ((bottom_data[i] > 0)
          + slope_data[c] * (bottom_data[i] <= 0));
  }
#else
  PReluDiffData* param = (PReluDiffData*)malloc(sizeof(PReluDiffData));
  param->in = bottom_data;
  param->diff = top_diff;
  param->out= bottom_diff;
  param->slope_data=slope_data;
  param->slope_diff=slope_diff;
  param->count=count;
  param->dim = dim;
  param->channels = channels;
  param->div_factor = div_factor;
  int ldm_consume = 8*__PRELU_BUFFSIZE_2*3;
  assert(ldm_consume < 64*1024);
  if(param_propagate_down) {
    athread_spawn(prelu_slave_backward_para_d,param);
    athread_join();
  }
  if(propagate_down) {
    athread_spawn(prelu_slave_backward_d,param);
    athread_join();
  }
  free(param);
#endif
//#undef NOSPE
}
