#include <assert.h>
#include <athread.h>
#include "caffe/util/swim2col.h"

#define LDM_MAX (64*1024)

extern SLAVE_FUN(sw_im2col_large_stride_f)();
extern SLAVE_FUN(sw_im2col_large_stride_d)();
extern SLAVE_FUN(sw_col2im_large_stride_f)();
extern SLAVE_FUN(sw_im2col_large_d)();
extern SLAVE_FUN(sw_im2col_large_f)();
extern SLAVE_FUN(sw_col2im_large_d)();
extern SLAVE_FUN(sw_col2im_large_f)();

typedef struct Im2colPara_st {
  void* data_im;
  void* data_col;
  int channels;
  int height;
  int width;
  int kernel_h;
  int kernel_w;
  int pad_h;
  int pad_w;
  int stride_h;
  int stride_w;
  int dilation_h;
  int dilation_w;
}Im2colPara;

// float version
void swim2col_f(const float* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    float* data_col) {
  Im2colPara* para = (Im2colPara*)malloc(sizeof(Im2colPara));
  para->data_im = data_im;
  para->data_col= data_col;
  para->channels= channels;
  para->height  = height;
  para->width   = width;
  para->kernel_h= kernel_h;
  para->kernel_w= kernel_w;
  para->pad_h   = pad_h;
  para->pad_w   = pad_w;
  para->stride_h= stride_h;
  para->stride_w= stride_w;
  para->dilation_h = dilation_h;
  para->dilation_w = dilation_w;
  // check parameter Precondition of sw_im2col_large_d
  assert(dilation_h==1);
  assert(dilation_w==1);
  if(stride_h==1 && stride_w==1) {
    assert((width+2*pad_w)*sizeof(float)<LDM_MAX);
    // spawn
    //printf("SPAWN sw_im2col_large_f\n");
    athread_spawn(sw_im2col_large_f,para);
    athread_join();
    //printf("sw_col2im_large_f end\n");
  } else {
    assert(((width+2*pad_w)+(width+2*pad_w-kernel_w)/stride_w+1)*sizeof(float)<LDM_MAX);
#ifdef PRINT_DEBUGINFO
    printf("SPAWN sw_im2col_large_stride_f\n");
#endif
    athread_spawn(sw_im2col_large_stride_f,para);
    athread_join();
#ifdef PRINT_DEBUGINFO
    printf("sw_im2col_large_stride_f end\n");
#endif
  }

  free(para);
}

// double version
void swim2col_d(const double* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    double* data_col) {
  Im2colPara* para = (Im2colPara*)malloc(sizeof(Im2colPara));
  para->data_im = data_im;
  para->data_col= data_col;
  para->channels= channels;
  para->height  = height;
  para->width   = width;
  para->kernel_h= kernel_h;
  para->kernel_w= kernel_w;
  para->pad_h   = pad_h;
  para->pad_w   = pad_w;
  para->stride_h= stride_h;
  para->stride_w= stride_w;
  para->dilation_h = dilation_h;
  para->dilation_w = dilation_w;
  // check parameter Precondition of sw_im2col_large_d
  assert(dilation_h==1);
  assert(dilation_w==1);
  if(stride_h==1 && stride_w==1) {
    assert((width+2*pad_w)*sizeof(double)<LDM_MAX);
    // spawn
    athread_spawn(sw_im2col_large_d,para);
    athread_join();
  } else {
    assert(((width+2*pad_w)+(width+2*pad_w-kernel_w)/stride_w+1)*sizeof(double)<LDM_MAX);
#ifdef PRINT_DEBUGINFO
    printf("SPAWN sw_im2col_large_stride_d\n");
#endif
    athread_spawn(sw_im2col_large_stride_d,para);
    athread_join();
#ifdef PRINT_DEBUGINFO
    printf("sw_im2col_large_stride_d end\n");
#endif
  }

  free(para);
}

// double version
void swcol2im_f(const float* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    float* data_im) {
  Im2colPara* para = (Im2colPara*)malloc(sizeof(Im2colPara));
  para->data_im = data_im;
  para->data_col= data_col;
  para->channels= channels;
  para->height  = height;
  para->width   = width;
  para->kernel_h= kernel_h;
  para->kernel_w= kernel_w;
  para->pad_h   = pad_h;
  para->pad_w   = pad_w;
  para->stride_h= stride_h;
  para->stride_w= stride_w;
  para->dilation_h = dilation_h;
  para->dilation_w = dilation_w;
  // check parameter Precondition of sw_im2col_large_d
  assert(dilation_h==1);
  assert(dilation_w==1);
  if(stride_h==1 && stride_w==1) {
    assert((width+2*pad_w-kernel_w+1+width)*sizeof(float)<LDM_MAX);
    // spawn
    athread_spawn(sw_col2im_large_f,para);
    athread_join();
  } else {
    assert(((width+2*pad_w-kernel_w)/stride_w+1+width)*sizeof(float)<LDM_MAX);
    athread_spawn(sw_col2im_large_stride_f,para);
    athread_join();
  }

  free(para);
}

// double version
void swcol2im_d(const double* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    double* data_im) {
  Im2colPara* para = (Im2colPara*)malloc(sizeof(Im2colPara));
  para->data_im = data_im;
  para->data_col= data_col;
  para->channels= channels;
  para->height  = height;
  para->width   = width;
  para->kernel_h= kernel_h;
  para->kernel_w= kernel_w;
  para->pad_h   = pad_h;
  para->pad_w   = pad_w;
  para->stride_h= stride_h;
  para->stride_w= stride_w;
  para->dilation_h = dilation_h;
  para->dilation_w = dilation_w;
  // check parameter Precondition of sw_im2col_large_d
  assert(stride_h==1);
  assert(stride_w==1);
  assert(dilation_h==1);
  assert(dilation_w==1);
  assert((width+2*pad_w-kernel_w+1+width)*sizeof(double)<LDM_MAX);
  // spawn
  athread_spawn(sw_col2im_large_d,para);
  athread_join();

  free(para);
}
