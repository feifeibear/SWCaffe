#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <slave.h>
#include <math.h>
#include "simd.h"
#include "dma.h"
#include "caffe/swlayers/prelu_type.h"

/*************
 * PRELU kernel for SPEs
 * Xin You
 * 2017 Aug.21
 *
 * input_data   is of dim(count)
 * input_diff   is of dim(count)
 * output_diff  is of dim(count)
 *
 * ***********/

// <double> slave functions

#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)<(b)?(a):(b))

#define SPNUM 64

#define Type double
#define SIMDSIZE 1
#define SIMDType double


__thread_local dma_desc dma_get_input, dma_get_diff, dma_put_output, dma_get_slope, dma_get_slope_diff, dma_put_slope_diff;

void prelu_slave_forward_d(PReluData* param)
{
  int count, start, local_count;
  int id = athread_get_id(-1);
  int dim = param->dim;
  int channels = param->channels;
  int div_factor = param->div_factor;
  count = param->count;
  local_count = count/64 + (id<(count%64));
  start = id*(count/64)+(id<(count%SPNUM)?id:(count%SPNUM));
  int local_inout_size = local_count/SIMDSIZE;
  SIMDType* local_input = (SIMDType*) ldm_malloc(sizeof(Type)*__PRELU_BUFFSIZE_1);
  SIMDType* local_output= (SIMDType*) ldm_malloc(sizeof(Type)*__PRELU_BUFFSIZE_1);
  Type* local_slope = (Type*) ldm_malloc(sizeof(Type)*((channels-1)/div_factor+1));
  Type* slope_ptr = (Type*)param->slope_data;
  Type* in_ptr = &((Type*)param->in )[start];
  Type* out_ptr= &((Type*)param->out)[start];
  volatile int input_replyget = 0, replyput = 0, slope_replyget = 0;
  int i,offset;

  dma_set_op(&dma_get_input, DMA_GET);
  dma_set_mode(&dma_get_input, PE_MODE);
  dma_set_reply(&dma_get_input, &input_replyget);

  dma_set_op(&dma_get_slope, DMA_GET);
  dma_set_mode(&dma_get_slope, PE_MODE);
  dma_set_reply(&dma_get_slope, &slope_replyget);

  dma_set_op(&dma_put_output, DMA_PUT);
  dma_set_mode(&dma_put_output, PE_MODE);
  dma_set_reply(&dma_put_output, &replyput);

  //DMA for local_input(local_count)
  dma_set_size(&dma_get_input, __PRELU_BUFFSIZE_1/SIMDSIZE*sizeof(SIMDType));

  //DMA for local_output(local_count)
  dma_set_size(&dma_put_output, __PRELU_BUFFSIZE_1/SIMDSIZE*sizeof(SIMDType));

  dma_set_size(&dma_get_slope, (1+(channels-1)/div_factor)*sizeof(Type));
  dma(dma_get_slope, (long)slope_ptr,(long)local_slope);
  dma_wait(&slope_replyget, 1); slope_replyget = 0;

  for(offset=0;offset+__PRELU_BUFFSIZE_1-1<local_inout_size;offset+=__PRELU_BUFFSIZE_1)
  {
    // begin DMA
    dma(dma_get_input, (long)(in_ptr+offset), (long)(local_input));
    dma_wait(&input_replyget, 1); input_replyget = 0;
    for(i=0;i<__PRELU_BUFFSIZE_1;++i) {
      int c = ((i+start+offset) / dim) % channels / div_factor;
      local_output[i] = max(local_input[i],0.0)
          + local_slope[c]*min(local_input[i],0.0);
    }

    dma(dma_put_output, (long)(out_ptr+offset), (long)(local_output));
    dma_wait(&replyput, 1); replyput = 0;
  }
  // calculate the rest
  if(offset<local_inout_size) {
    dma_set_size(&dma_get_input, (local_inout_size-offset)/SIMDSIZE*sizeof(SIMDType));
    dma_set_size(&dma_put_output,(local_inout_size-offset)/SIMDSIZE*sizeof(SIMDType));
    // begin DMA
    dma(dma_get_input, (long)(in_ptr+offset), (long)(local_input));
    dma_wait(&input_replyget, 1); input_replyget = 0;

    for(i=0;i<local_inout_size-offset;++i) {
      int c = ((i+start+offset) / dim) % channels / div_factor;
      local_output[i] = max(local_input[i],0)
          + local_slope[c]*min(local_input[i],0);
    }

    dma(dma_put_output, (long)(out_ptr+offset), (long)(local_output));
    dma_wait(&replyput, 1); replyput = 0;
  }
  ldm_free(local_input, sizeof(SIMDType)*__PRELU_BUFFSIZE_1);
  ldm_free(local_output,sizeof(SIMDType)*__PRELU_BUFFSIZE_1);
  ldm_free(local_slope,sizeof(Type)*(channels-1));

}

void prelu_slave_backward_d(PReluDiffData* param)
{
  int count, start, local_count;
  int id = athread_get_id(-1);
  int dim = param->dim;
  int channels = param->channels;
  int div_factor = param->div_factor;
  count = param->count;
  local_count = count/64 + (id<(count%64));
  start = id*(count/64)+(id<(count%64)?id:(count%64));
  int local_inout_size = local_count/SIMDSIZE;
  SIMDType* local_input = (SIMDType*) ldm_malloc(sizeof(Type)*__PRELU_BUFFSIZE_2);
  SIMDType* local_diff = (SIMDType*) ldm_malloc(sizeof(Type)*__PRELU_BUFFSIZE_2);
  SIMDType* local_output= (SIMDType*) ldm_malloc(sizeof(Type)*__PRELU_BUFFSIZE_2);
  Type* local_slope = (Type*) ldm_malloc(sizeof(Type)*(1+(channels-1)/div_factor));
  Type* slope_ptr = (Type*)param->slope_data;
  Type* in_ptr = param->in+start*sizeof(Type);
  Type* diff_ptr = param->diff+start*sizeof(Type);
  Type* out_ptr= param->out+start*sizeof(Type);
  volatile int input_replyget = 0, diff_replyget = 0, replyput = 0, slope_replyget = 0;
  int i,offset;

  dma_set_op(&dma_get_input, DMA_GET);
  dma_set_mode(&dma_get_input, PE_MODE);
  dma_set_reply(&dma_get_input, &input_replyget);

  dma_set_op(&dma_get_diff, DMA_GET);
  dma_set_mode(&dma_get_diff, PE_MODE);
  dma_set_reply(&dma_get_diff, &diff_replyget);

  dma_set_op(&dma_get_slope, DMA_GET);
  dma_set_mode(&dma_get_slope, PE_MODE);
  dma_set_reply(&dma_get_slope, &slope_replyget);

  dma_set_op(&dma_put_output, DMA_PUT);
  dma_set_mode(&dma_put_output, PE_MODE);
  dma_set_reply(&dma_put_output, &replyput);

  //DMA for local_input(local_count)
  dma_set_size(&dma_get_input, __PRELU_BUFFSIZE_2*sizeof(SIMDType));

  //DMA for local_diff(local_count)
  dma_set_size(&dma_get_diff, __PRELU_BUFFSIZE_2*sizeof(SIMDType));

  //DMA for local_output(local_count)
  dma_set_size(&dma_put_output, __PRELU_BUFFSIZE_2*sizeof(SIMDType));

  dma_set_size(&dma_get_slope, (1+(channels-1)/div_factor)*sizeof(Type));
  dma(dma_get_slope, (long)slope_ptr,(long)local_slope);
  dma_wait(&slope_replyget, 1); slope_replyget = 0;

  for(offset=0;offset+__PRELU_BUFFSIZE_2-1<local_inout_size;offset+=__PRELU_BUFFSIZE_2) {
    // begin DMA
    dma(dma_get_input, (long)(in_ptr+offset), (long)(local_input));
    dma_wait(&input_replyget, 1); input_replyget = 0;

    dma(dma_get_diff, (long)(diff_ptr+offset), (long)(local_diff));
    dma_wait(&diff_replyget, 1); diff_replyget = 0;

    for(i=0;i<__PRELU_BUFFSIZE_2;++i) {
      int c = ((i+start+offset) / dim) % channels / div_factor;
      local_output[i] = local_diff[i] * ((local_input[i]>0)
          + local_slope[c]*(local_input[i]<=0));
    }

    dma(dma_put_output, (long)(out_ptr+offset), (long)(local_output));
    dma_wait(&replyput, 1); replyput = 0;
  }
  // calculate the rest
  if(offset<local_inout_size) {
  dma_set_size(&dma_get_input,(local_inout_size-offset)*sizeof(SIMDType));
  dma_set_size(&dma_get_diff, (local_inout_size-offset)*sizeof(SIMDType));
  dma_set_size(&dma_put_output,(local_inout_size-offset)*sizeof(SIMDType));
    // begin DMA
    dma(dma_get_input, (long)(in_ptr+offset), (long)(local_input));
    dma_wait(&input_replyget, 1); input_replyget = 0;

    dma(dma_get_diff, (long)(diff_ptr+offset), (long)(local_diff));
    dma_wait(&diff_replyget, 1); diff_replyget = 0;

    for(i=0;i<local_inout_size-offset;++i) {
      int c = ((i+start+offset) / dim) % channels / div_factor;
      local_output[i] = local_diff[i] * ((local_input[i]>0)
          + local_slope[c]*(local_input[i]<=0));
    }

    dma(dma_put_output, (long)(out_ptr+offset), (long)(local_output));
    dma_wait(&replyput, 1); replyput = 0;
  }
  ldm_free(local_input ,sizeof(SIMDType)*__PRELU_BUFFSIZE_2);
  ldm_free(local_diff  ,sizeof(SIMDType)*__PRELU_BUFFSIZE_2);
  ldm_free(local_output,sizeof(SIMDType)*__PRELU_BUFFSIZE_2);
  ldm_free(local_slope,sizeof(Type)*(channels-1));

}
#undef Type
#undef SIMDSIZE
#undef SIMDType

// <float> slave functions
#define Type float
#define SIMDSIZE 1
#define SIMDType float
void prelu_slave_forward_f(PReluData* param)
{
  int count, start, local_count;
  int id = athread_get_id(-1);
  int dim = param->dim;
  int channels = param->channels;
  int div_factor = param->div_factor;
  count = param->count;
  local_count = count/64 + (id<(count%64));
  start = id*(count/64)+(id<(count%64)?id:(count%64));
  int local_inout_size = local_count/SIMDSIZE;
  SIMDType* local_input = (SIMDType*) ldm_malloc(sizeof(Type)*__PRELU_BUFFSIZE_1);
  SIMDType* local_output= (SIMDType*) ldm_malloc(sizeof(Type)*__PRELU_BUFFSIZE_1);
  Type* local_slope = (Type*) ldm_malloc(sizeof(Type)*(1+(channels-1)/div_factor));
  Type* slope_ptr = (Type*)param->slope_data;
  Type* in_ptr = param->in+start*sizeof(Type);
  Type* out_ptr= param->out+start*sizeof(Type);
  volatile int input_replyget = 0, replyput = 0, slope_replyget = 0;
  int i,offset;

  dma_set_op(&dma_get_input, DMA_GET);
  dma_set_mode(&dma_get_input, PE_MODE);
  dma_set_reply(&dma_get_input, &input_replyget);

  dma_set_op(&dma_get_slope, DMA_GET);
  dma_set_mode(&dma_get_slope, PE_MODE);
  dma_set_reply(&dma_get_slope, &slope_replyget);

  dma_set_op(&dma_put_output, DMA_PUT);
  dma_set_mode(&dma_put_output, PE_MODE);
  dma_set_reply(&dma_put_output, &replyput);

  //DMA for local_input(local_count)
  dma_set_size(&dma_get_input, __PRELU_BUFFSIZE_1/SIMDSIZE*sizeof(SIMDType));

  //DMA for local_output(local_count)
  dma_set_size(&dma_put_output, __PRELU_BUFFSIZE_1/SIMDSIZE*sizeof(SIMDType));

  dma_set_size(&dma_get_slope, (1+(channels-1)/div_factor)*sizeof(Type));
  dma(dma_get_slope, (long)slope_ptr,(long)local_slope);
  dma_wait(&slope_replyget, 1); slope_replyget = 0;

  for(offset=0;offset+__PRELU_BUFFSIZE_1-1<local_inout_size;offset+=__PRELU_BUFFSIZE_1)
  {
    // begin DMA
    dma(dma_get_input, (long)(in_ptr+offset), (long)(local_input));
    dma_wait(&input_replyget, 1); input_replyget = 0;

    for(i=0;i<__PRELU_BUFFSIZE_1;++i) {
      int c = ((i+offset+start) / dim) % channels / div_factor;
      local_output[i] = max(local_input[i],0)
          + local_slope[c]*min(local_input[i],0);
    }

    dma(dma_put_output, (long)(out_ptr+offset), (long)(local_output));
    dma_wait(&replyput, 1); replyput = 0;
  }
  // calculate the rest
  if(offset<local_inout_size) {
    dma_set_size(&dma_get_input, (local_inout_size-offset)/SIMDSIZE*sizeof(SIMDType));
    dma_set_size(&dma_put_output,(local_inout_size-offset)/SIMDSIZE*sizeof(SIMDType));
    // begin DMA
    dma(dma_get_input, (long)(in_ptr+offset), (long)(local_input));
    dma_wait(&input_replyget, 1); input_replyget = 0;

    for(i=0;i<local_inout_size-offset;++i) {
      int c = ((i+offset+start) / dim) % channels / div_factor;
      local_output[i] = max(local_input[i],0)
          + local_slope[c]*min(local_input[i],0);
    }

    dma(dma_put_output, (long)(out_ptr+offset), (long)(local_output));
    dma_wait(&replyput, 1); replyput = 0;
  }
  ldm_free(local_input, sizeof(SIMDType)*__PRELU_BUFFSIZE_1);
  ldm_free(local_output,sizeof(SIMDType)*__PRELU_BUFFSIZE_1);
  ldm_free(local_slope,sizeof(Type)*(channels-1));

}

void prelu_slave_backward_f(PReluDiffData* param)
{
  int count, start, end, local_count;
  int id = athread_get_id(-1);
  int dim = param->dim;
  int channels = param->channels;
  int div_factor = param->div_factor;
  count = param->count;
  local_count = count/64 + (id<(count%64));
  start = id*(count/64)+(id<(count%64)?id:(count%64));
  end = start+local_count;
  int local_inout_size = local_count/SIMDSIZE;
  SIMDType* local_input = (SIMDType*) ldm_malloc(sizeof(Type)*__PRELU_BUFFSIZE_2);
  SIMDType* local_diff = (SIMDType*) ldm_malloc(sizeof(Type)*__PRELU_BUFFSIZE_2);
  SIMDType* local_output= (SIMDType*) ldm_malloc(sizeof(Type)*__PRELU_BUFFSIZE_2);
  Type* local_slope = (Type*) ldm_malloc(sizeof(Type)*(1+(channels-1)/div_factor));
  Type* slope_ptr = (Type*)param->slope_data;
  Type* in_ptr = param->in+start*sizeof(Type);
  Type* diff_ptr = param->diff+start*sizeof(Type);
  Type* out_ptr= param->out+start*sizeof(Type);
  volatile int input_replyget = 0, diff_replyget = 0, replyput = 0, slope_replyget = 0;
  int i,offset;

  dma_set_op(&dma_get_input, DMA_GET);
  dma_set_mode(&dma_get_input, PE_MODE);
  dma_set_reply(&dma_get_input, &input_replyget);

  dma_set_op(&dma_get_diff, DMA_GET);
  dma_set_mode(&dma_get_diff, PE_MODE);
  dma_set_reply(&dma_get_diff, &diff_replyget);

  dma_set_op(&dma_get_slope, DMA_GET);
  dma_set_mode(&dma_get_slope, PE_MODE);
  dma_set_reply(&dma_get_slope, &slope_replyget);

  dma_set_op(&dma_put_output, DMA_PUT);
  dma_set_mode(&dma_put_output, PE_MODE);
  dma_set_reply(&dma_put_output, &replyput);

  //DMA for local_input(local_count)
  dma_set_size(&dma_get_input, __PRELU_BUFFSIZE_2*sizeof(SIMDType));

  //DMA for local_diff(local_count)
  dma_set_size(&dma_get_diff, __PRELU_BUFFSIZE_2*sizeof(SIMDType));

  //DMA for local_output(local_count)
  dma_set_size(&dma_put_output, __PRELU_BUFFSIZE_2*sizeof(SIMDType));

  dma_set_size(&dma_get_slope, (1+(channels-1)/div_factor)*sizeof(Type));
  dma(dma_get_slope, (long)slope_ptr,(long)local_slope);
  dma_wait(&slope_replyget, 1); slope_replyget = 0;

  for(offset=0;offset+__PRELU_BUFFSIZE_2-1<local_inout_size;offset+=__PRELU_BUFFSIZE_2) {
    // begin DMA
    dma(dma_get_input, (long)(in_ptr+offset), (long)(local_input));
    dma_wait(&input_replyget, 1); input_replyget = 0;

    dma(dma_get_diff, (long)(diff_ptr+offset), (long)(local_diff));
    dma_wait(&diff_replyget, 1); diff_replyget = 0;

    for(i=0;i<__PRELU_BUFFSIZE_2;++i) {
      int c = ((i+offset+start) / dim) % channels / div_factor;
      local_output[i] = local_diff[i] * ((local_input[i]>0)
          + local_slope[c]*(local_input[i]<=0));
    }

    dma(dma_put_output, (long)(out_ptr+offset), (long)(local_output));
    dma_wait(&replyput, 1); replyput = 0;
  }
  // calculate the rest
  if(offset<local_inout_size) {
    dma_set_size(&dma_get_input,(local_inout_size-offset)*sizeof(SIMDType));
    dma_set_size(&dma_get_diff, (local_inout_size-offset)*sizeof(SIMDType));
    dma_set_size(&dma_put_output,(local_inout_size-offset)*sizeof(SIMDType));
    // begin DMA
    dma(dma_get_input, (long)(in_ptr+offset), (long)(local_input));
    dma_wait(&input_replyget, 1); input_replyget = 0;

    dma(dma_get_diff, (long)(diff_ptr+offset), (long)(local_diff));
    dma_wait(&diff_replyget, 1); diff_replyget = 0;

    for(i=0;i<local_inout_size-offset;++i) {
      int c = ((i+start+offset) / dim) % channels / div_factor;
      local_output[i] = local_diff[i] * ((local_input[i]>0)
          + local_slope[c]*(local_input[i]<=0));
    }

    dma(dma_put_output, (long)(out_ptr+offset), (long)(local_output));
    dma_wait(&replyput, 1); replyput = 0;
  }
  ldm_free(local_input ,sizeof(SIMDType)*__PRELU_BUFFSIZE_2);
  ldm_free(local_diff  ,sizeof(SIMDType)*__PRELU_BUFFSIZE_2);
  ldm_free(local_output,sizeof(SIMDType)*__PRELU_BUFFSIZE_2);
  ldm_free(local_slope,sizeof(Type)*(channels-1));

}
#undef Type
#undef SIMDSIZE
#undef SIMDType

#define Type float
#define SIMDSIZE 4
#define SIMDType floatv4
// round up with SIMDSIZE
#define ROUNDUP(x) (((x)+SIMDSIZE-1)&(~(SIMDSIZE-1)))
#define REG_PUTR(var, dst) asm volatile ("putr %0,%1\n"::"r"(var),"r"(dst))
#define REG_PUTC(var, dst) asm volatile ("putc %0,%1\n"::"r"(var),"r"(dst))
#define REG_GETR(var) asm volatile ("getr %0\n":"=r"(var))
#define REG_GETC(var) asm volatile ("getc %0\n":"=r"(var))
// x is the code
#define EXE_ONEBYONE(x) do{ SIMDType __tmp; if(id==0) {x;REG_PUTR(__tmp,1);REG_GETR(__tmp);REG_PUTC(__tmp,1);} \
                            else if(id%8==0) {REG_GETC(__tmp);x;REG_PUTR(__tmp,1);REG_GETR(__tmp); \
                                              if(id/8!=7)REG_PUTC(__tmp,id/8+1);}\
                            else if(id%8==7) {REG_GETR(__tmp);x;REG_PUTR(__tmp,0);}\
                            else {REG_GETR(__tmp);x;REG_PUTR(__tmp,id%8+1);}\
                            }while(0)
// div_factor == 1 or channels
void prelu_slave_backward_para_f(PReluDiffData* param)
{
  int count, start, end, local_count;
  int id = athread_get_id(-1);
  int dim = param->dim;
  int channels = param->channels;
  int div_factor = param->div_factor;
  count = param->count;
  local_count = count/64 + (id<(count%64));
  start = id*(count/64)+(id<(count%64)?id:(count%64));
  end = start+local_count;
  int local_inout_size = local_count;//SIMDSIZE;
  Type* local_input = (Type*) ldm_malloc(sizeof(Type)*__PRELU_BUFFSIZE_2);
  Type* local_diff  = (Type*) ldm_malloc(sizeof(Type)*__PRELU_BUFFSIZE_2);
  Type* local_slope_diff = (Type*) ldm_malloc(sizeof(Type)*ROUNDUP(channels));
  Type* slope_ptr = (Type*)param->slope_data;
  Type* in_ptr = param->in+start*sizeof(Type);
  Type* diff_ptr = param->diff+start*sizeof(Type);
  Type* slope_diff_ptr = param->slope_diff;
  volatile int input_replyget = 0, diff_replyget = 0, replyput = 0;
  volatile int slope_replyget = 0, slope_diff_replyget = 0, slope_diff_replyput = 0;
  int i,offset;

  dma_set_op(&dma_get_input, DMA_GET);
  dma_set_mode(&dma_get_input, PE_MODE);
  dma_set_reply(&dma_get_input, &input_replyget);

  dma_set_op(&dma_get_diff, DMA_GET);
  dma_set_mode(&dma_get_diff, PE_MODE);
  dma_set_reply(&dma_get_diff, &diff_replyget);

  dma_set_op(&dma_get_slope_diff, DMA_GET);
  dma_set_mode(&dma_get_slope_diff, PE_MODE);
  dma_set_reply(&dma_get_slope_diff, &slope_diff_replyget);

  dma_set_op(&dma_put_slope_diff, DMA_PUT);
  dma_set_mode(&dma_put_slope_diff, PE_MODE);
  dma_set_reply(&dma_put_slope_diff, &slope_diff_replyput);

  //DMA for local_input(local_count)
  dma_set_size(&dma_get_input, __PRELU_BUFFSIZE_2*sizeof(Type));

  //DMA for local_diff(local_count)
  dma_set_size(&dma_get_diff, __PRELU_BUFFSIZE_2*sizeof(Type));
  if(div_factor==channels) {
    if(id==0)
      local_slope_diff[0] = slope_diff_ptr[0];
    else
      local_slope_diff[0] = 0;
  } else {
    if(id==0) {
      dma_set_size(&dma_get_slope_diff, channels*sizeof(Type));
      dma(dma_get_slope_diff, (long)slope_diff_ptr,(long)local_slope_diff);
      dma_wait(&slope_diff_replyget, 1); slope_diff_replyget = 0;
      for(i=channels;i<ROUNDUP(channels);++i) local_slope_diff[i] = 0;
    } else {
      for(i=0;i<ROUNDUP(channels);++i) local_slope_diff[i] = 0;
    }
  }

  //if(id==0) printf("SPE %d: start with local_slope_diff[0] = %lf\n",id,local_slope_diff[0]);

  for(offset=0;offset+__PRELU_BUFFSIZE_2-1<local_inout_size;offset+=__PRELU_BUFFSIZE_2) {
    // begin DMA
    dma(dma_get_input, (long)(in_ptr+offset), (long)(local_input));
    dma_wait(&input_replyget, 1); input_replyget = 0;

    dma(dma_get_diff, (long)(diff_ptr+offset), (long)(local_diff));
    dma_wait(&diff_replyget, 1); diff_replyget = 0;

    for(i=0;i<__PRELU_BUFFSIZE_2;++i) {
      int c = ((i+start+offset) / dim) % channels / div_factor;
      local_slope_diff[c] += local_diff[i] * local_input[i] * (local_input[i]<=0);
    }
  }
  // calculate the rest
  if(offset<local_inout_size) {
    dma_set_size(&dma_get_input,(local_inout_size-offset)*sizeof(Type));
    dma_set_size(&dma_get_diff, (local_inout_size-offset)*sizeof(Type));
    // begin DMA
    dma(dma_get_input, (long)(in_ptr+offset), (long)(local_input));
    dma_wait(&input_replyget, 1); input_replyget = 0;

    dma(dma_get_diff, (long)(diff_ptr+offset), (long)(local_diff));
    dma_wait(&diff_replyget, 1); diff_replyget = 0;

    for(i=0;i<local_inout_size-offset;++i) {
      int c = ((i+start+offset) / dim) % channels / div_factor;
      local_slope_diff[c] += local_diff[i] * local_input[i] * (local_input[i]<=0);
    }
  }
  //EXE_ONEBYONE(printf("SPE %d: slope_diff[0]=%lf\n",id,local_slope_diff[0]));
  //if(id==56) printf("SPE %d:now sum with register communication. now slope_diff[0]=%lf\n",id,local_slope_diff[0]);
  // sum
  int col = id % 8;
  int row = id / 8;
  int dst;
  int size = ROUNDUP(channels)/SIMDSIZE;
  SIMDType vsend, vrecv, vori;
  // sum with flow
    if(row % 2 == 0) {
      if(col == 7) {
        dst = row + 1;
        for(i=0;i<size;++i) {
          simd_load(vori,&local_slope_diff[SIMDSIZE*i]);
          REG_GETR(vrecv);
          vsend = vori + vrecv;
          REG_PUTC(vsend,dst);
        }
      } else if(col!=0) {
        dst = col + 1;
        for(i=0;i<size;++i) {
          simd_load(vori,&local_slope_diff[SIMDSIZE*i]);
          REG_GETR(vrecv);
          vsend = vori + vrecv;
          REG_PUTR(vsend,dst);
        }
      } else if(id==0) {
        dst = col + 1;
        for(i=0;i<size;++i) {
          simd_load(vsend,&local_slope_diff[SIMDSIZE*i]);
          REG_PUTR(vsend,dst);
        }
      } else {
        dst = col + 1;
        for(i=0;i<size;++i) {
          simd_load(vori,&local_slope_diff[SIMDSIZE*i]);
          REG_GETC(vrecv);
          vsend = vori + vrecv;
          REG_PUTR(vsend,dst);
        }
      }
    } else {
      if(col == 7) {
        dst = col - 1;
        for(i=0;i<size;++i) {
          simd_load(vori,&local_slope_diff[SIMDSIZE*i]);
          REG_GETC(vrecv);
          vsend = vori + vrecv;
          REG_PUTR(vsend,dst);
        }
      } else if(col!=0) {
        dst = col - 1;
        for(i=0;i<size;++i) {
          simd_load(vori,&local_slope_diff[SIMDSIZE*i]);
          REG_GETR(vrecv);
          vsend = vori + vrecv;
          REG_PUTR(vsend,dst);
        }
      } else if(row==7) {
        for(i=0;i<size;++i) {
          simd_load(vori,&local_slope_diff[SIMDSIZE*i]);
          REG_GETR(vrecv);
          vsend = vori + vrecv;
          simd_store(vsend,&local_slope_diff[SIMDSIZE*i]);
        }
      } else {
        dst = row + 1;
        for(i=0;i<size;++i) {
          simd_load(vori,&local_slope_diff[SIMDSIZE*i]);
          REG_GETR(vrecv);
          vsend = vori + vrecv;
          REG_PUTC(vsend,dst);
        }
      }
    }
    //printf("SPE %d: local_slope_diff[0]=%lf\n",id,local_slope_diff[0]);
    //if(id==56) printf("SPE %d:sum complete.\n",id);
    // put answer back: row = 7, col = 0 -> id = row*8+col = 56
    if(div_factor==channels && id==56)
      slope_diff_ptr[0] = local_slope_diff[0];
    else if(id==56) {
      dma_set_size(&dma_put_slope_diff, channels*sizeof(Type));
      dma(dma_put_slope_diff, (long)slope_diff_ptr,(long)local_slope_diff);
      dma_wait(&slope_diff_replyput, 1); slope_diff_replyput = 0;
    }
  //if(id==56) printf("SPE %d:slope diff is OK.\n",id);
  // free
  ldm_free(local_input ,sizeof(SIMDType)*__PRELU_BUFFSIZE_2);
  ldm_free(local_diff  ,sizeof(SIMDType)*__PRELU_BUFFSIZE_2);
  ldm_free(local_slope_diff,sizeof(Type)*(channels-1));
}
#undef Type
#undef SIMDSIZE
#undef SIMDType

#define Type double
#define SIMDSIZE 4
#define SIMDType doublev4
void prelu_slave_backward_para_d(PReluDiffData* param)
{
  int count, start, end, local_count;
  int id = athread_get_id(-1);
  int dim = param->dim;
  int channels = param->channels;
  int div_factor = param->div_factor;
  count = param->count;
  local_count = count/64 + (id<(count%64));
  start = id*(count/64)+(id<(count%64)?id:(count%64));
  end = start+local_count;
  int local_inout_size = local_count;//SIMDSIZE;
  Type* local_input = (Type*) (long) ldm_malloc(sizeof(Type)*__PRELU_BUFFSIZE_2);
  Type* local_diff  = (Type*) (long) ldm_malloc(sizeof(Type)*__PRELU_BUFFSIZE_2);
  Type* local_slope_diff = (Type*) (long) ldm_malloc(sizeof(Type)*ROUNDUP(channels));
  Type* in_ptr = param->in+start*sizeof(Type);
  Type* diff_ptr = param->diff+start*sizeof(Type);
  Type* slope_diff_ptr = param->slope_diff;
  volatile int input_replyget = 0, diff_replyget = 0, replyput = 0;
  volatile int slope_replyget = 0, slope_diff_replyget = 0, slope_diff_replyput = 0;
  int i,offset;

  dma_set_op(&dma_get_input, DMA_GET);
  dma_set_mode(&dma_get_input, PE_MODE);
  dma_set_reply(&dma_get_input, &input_replyget);

  dma_set_op(&dma_get_diff, DMA_GET);
  dma_set_mode(&dma_get_diff, PE_MODE);
  dma_set_reply(&dma_get_diff, &diff_replyget);

  dma_set_op(&dma_get_slope_diff, DMA_GET);
  dma_set_mode(&dma_get_slope_diff, PE_MODE);
  dma_set_reply(&dma_get_slope_diff, &slope_diff_replyget);

  dma_set_op(&dma_put_slope_diff, DMA_PUT);
  dma_set_mode(&dma_put_slope_diff, PE_MODE);
  dma_set_reply(&dma_put_slope_diff, &slope_diff_replyput);

  //DMA for local_input(local_count)
  dma_set_size(&dma_get_input, __PRELU_BUFFSIZE_2*sizeof(Type));

  //DMA for local_diff(local_count)
  dma_set_size(&dma_get_diff, __PRELU_BUFFSIZE_2*sizeof(Type));
  if(div_factor==channels) {
    if(id==0)
      local_slope_diff[0] = slope_diff_ptr[0];
    else
      local_slope_diff[0] = 0;
  } else {
    if(id==0) {
      dma_set_size(&dma_get_slope_diff, channels*sizeof(Type));
      dma(dma_get_slope_diff, (long)slope_diff_ptr,(long)local_slope_diff);
      dma_wait(&slope_diff_replyget, 1); slope_diff_replyget = 0;
      for(i=channels;i<ROUNDUP(channels);++i) local_slope_diff[i] = 0;
    } else {
      for(i=0;i<ROUNDUP(channels);++i) local_slope_diff[i] = 0;
    }
  }
  for(offset=0;offset+__PRELU_BUFFSIZE_2-1<local_inout_size;offset+=__PRELU_BUFFSIZE_2) {
    // begin DMA
    dma(dma_get_input, (long)(in_ptr+offset), (long)(local_input));
    dma_wait(&input_replyget, 1); input_replyget = 0;

    dma(dma_get_diff, (long)(diff_ptr+offset), (long)(local_diff));
    dma_wait(&diff_replyget, 1); diff_replyget = 0;

    for(i=0;i<__PRELU_BUFFSIZE_2;++i) {
      int c = ((i+start+offset) / dim) % channels / div_factor;
      local_slope_diff[c] += local_diff[i] * local_input[i] * (local_input[i]<=0);
    }
  }
  // calculate the rest
  if(offset<local_inout_size) {
    dma_set_size(&dma_get_input,(local_inout_size-offset)*sizeof(Type));
    dma_set_size(&dma_get_diff, (local_inout_size-offset)*sizeof(Type));
    // begin DMA
    dma(dma_get_input, (long)(in_ptr+offset), (long)(local_input));
    dma_wait(&input_replyget, 1); input_replyget = 0;

    dma(dma_get_diff, (long)(diff_ptr+offset), (long)(local_diff));
    dma_wait(&diff_replyget, 1); diff_replyget = 0;

    for(i=0;i<local_inout_size-offset;++i) {
      int c = ((i+start+offset) / dim) % channels / div_factor;
      local_slope_diff[c] += local_diff[i] * local_input[i] * (local_input[i]<=0);
    }
  }
  // sum
  int col = id % 8;
  int row = id / 8;
  int dst;
  int size = ROUNDUP(channels)/SIMDSIZE;
  SIMDType vsend, vrecv, vori;
    if(row % 2 == 0) {
      if(col == 7) {
        dst = row + 1;
        for(i=0;i<size;++i) {
          simd_load(vori,&local_slope_diff[SIMDSIZE*i]);
          REG_GETR(vrecv);
          vsend = vori + vrecv;
          REG_PUTC(vsend,dst);
        }
      } else if(col!=0) {
        dst = col + 1;
        for(i=0;i<size;++i) {
          simd_load(vori,&local_slope_diff[SIMDSIZE*i]);
          REG_GETR(vrecv);
          vsend = vori + vrecv;
          REG_PUTR(vsend,dst);
        }
      } else if(id==0) {
        dst = col + 1;
        for(i=0;i<size;++i) {
          simd_load(vsend,&local_slope_diff[SIMDSIZE*i]);
          REG_PUTR(vsend,dst);
        }
      } else {
        dst = col + 1;
        for(i=0;i<size;++i) {
          simd_load(vori,&local_slope_diff[SIMDSIZE*i]);
          REG_GETC(vrecv);
          vsend = vori + vrecv;
          REG_PUTR(vsend,dst);
        }
      }
    } else {
      if(col == 7) {
        dst = col - 1;
        for(i=0;i<size;++i) {
          simd_load(vori,&local_slope_diff[SIMDSIZE*i]);
          REG_GETC(vrecv);
          vsend = vori + vrecv;
          REG_PUTR(vsend,dst);
        }
      } else if(col!=0) {
        dst = col - 1;
        for(i=0;i<size;++i) {
          simd_load(vori,&local_slope_diff[SIMDSIZE*i]);
          REG_GETR(vrecv);
          vsend = vori + vrecv;
          REG_PUTR(vsend,dst);
        }
      } else if(row==7) {
        for(i=0;i<size;++i) {
          simd_load(vori,&local_slope_diff[SIMDSIZE*i]);
          REG_GETR(vrecv);
          vsend = vori + vrecv;
          simd_store(vsend,&local_slope_diff[SIMDSIZE*i]);
        }
      } else {
        dst = col - 1;
        for(i=0;i<size;++i) {
          simd_load(vori,&local_slope_diff[SIMDSIZE*i]);
          REG_GETR(vrecv);
          vsend = vori + vrecv;
          REG_PUTR(vsend,dst);
        }
      }
    }
  // put answer back
  if(div_factor==channels && id==63)
    slope_diff_ptr[0] = local_slope_diff[0];
  else if(id==63) {
      dma_set_size(&dma_put_slope_diff, channels*sizeof(Type));
      dma(dma_put_slope_diff, (long)slope_diff_ptr,(long)local_slope_diff);
      dma_wait(&slope_diff_replyput, 1); slope_diff_replyput = 0;
  }
  // free
  ldm_free(local_input ,sizeof(SIMDType)*__PRELU_BUFFSIZE_2);
  ldm_free(local_diff  ,sizeof(SIMDType)*__PRELU_BUFFSIZE_2);
  ldm_free(local_slope_diff,sizeof(Type)*(channels-1));
}
#undef Type
#undef SIMDSIZE
#undef SIMDType
