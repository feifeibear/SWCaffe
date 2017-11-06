#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <slave.h>
#include <math.h>
#include "simd.h"
#include "dma.h"
#include "caffe/swlayers/relu_type.h"

/*************
 * RELU kernel for SPEs
 * Xin You
 * 2017 Aug.2nd
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

__thread_local dma_desc dma_get_input, dma_get_diff, dma_put_output;

void relu_slave_forward_d(ReluData* param)
{
  int count, start, local_count;
  int id = athread_get_id(-1);
  count = param->count;
  local_count = count/64 + (id<(count%64));
  start = id*(count/64)+(id<(count%SPNUM)?id:(count%SPNUM));
  int local_inout_size = local_count/SIMDSIZE;
  SIMDType* local_input = (SIMDType*) ldm_malloc(sizeof(Type)*__RELU_BUFFSIZE_1);
  SIMDType* local_output= (SIMDType*) ldm_malloc(sizeof(Type)*__RELU_BUFFSIZE_1);
  Type local_negative_slope = param->negative_slope.dbl;
  Type* in_ptr = &((Type*)param->in )[start];
  Type* out_ptr= &((Type*)param->out)[start];
  volatile int input_replyget = 0, replyput = 0;
  int i,offset;

  //dma_desc dma_get_input, dma_put_output;

  dma_set_op(&dma_get_input, DMA_GET);
  dma_set_mode(&dma_get_input, PE_MODE);
  dma_set_reply(&dma_get_input, &input_replyget);

  dma_set_op(&dma_put_output, DMA_PUT);
  dma_set_mode(&dma_put_output, PE_MODE);
  dma_set_reply(&dma_put_output, &replyput);

  //DMA for local_input(local_count)
  dma_set_size(&dma_get_input, __RELU_BUFFSIZE_1/SIMDSIZE*sizeof(SIMDType));
  //dma_set_bsize(&dma_get_input,__RELU_BUFFSIZE_1/SIMDSIZE*sizeof(SIMDType));

  //DMA for local_output(local_count)
  dma_set_size(&dma_put_output, __RELU_BUFFSIZE_1/SIMDSIZE*sizeof(SIMDType));
  //dma_set_bsize(&dma_put_output,__RELU_BUFFSIZE_1/SIMDSIZE*sizeof(SIMDType));

  for(offset=0;offset+__RELU_BUFFSIZE_1-1<local_inout_size;offset+=__RELU_BUFFSIZE_1)
  {
    // begin DMA
    dma(dma_get_input, (long)(in_ptr+offset), (long)(local_input));
    dma_wait(&input_replyget, 1); input_replyget = 0;
    //athread_get(PE_MODE,&in_ptr[offset],local_input,__RELU_BUFFSIZE_1/SIMDSIZE*sizeof(SIMDType),&input_replyget,0,0,0);
    //while(input_replyget!=1); input_replyget=0;
    for(i=0;i<__RELU_BUFFSIZE_1;++i) {
      //local_input[i] = in_ptr[offset+i];
      local_output[i] = max(local_input[i],0.0)
          + local_negative_slope*min(local_input[i],0.0);
      //out_ptr[offset+i] = local_output[i];
    }

    dma(dma_put_output, (long)(out_ptr+offset), (long)(local_output));
    dma_wait(&replyput, 1); replyput = 0;
    //athread_put(PE_MODE,local_output,&out_ptr[offset],__RELU_BUFFSIZE_1/SIMDSIZE*sizeof(SIMDType),&replyput,0,0);
    //while(replyput!=1); replyput=0;
  }
  // calculate the rest
  if(offset<local_inout_size) {
    dma_set_size(&dma_get_input, (local_inout_size-offset)/SIMDSIZE*sizeof(SIMDType));
    dma_set_size(&dma_put_output,(local_inout_size-offset)/SIMDSIZE*sizeof(SIMDType));
    // begin DMA
    //athread_get(PE_MODE,&in_ptr[offset],local_input,(local_inout_size-offset)/SIMDSIZE*sizeof(SIMDType),&input_replyget,0,0,0);
    //while(input_replyget!=1); input_replyget=0;
    dma(dma_get_input, (long)(in_ptr+offset), (long)(local_input));
    dma_wait(&input_replyget, 1); input_replyget = 0;

    for(i=0;i<local_inout_size-offset;++i) {
      local_output[i] = max(local_input[i],0)
          + local_negative_slope*min(local_input[i],0);
    }

    dma(dma_put_output, (long)(out_ptr+offset), (long)(local_output));
    dma_wait(&replyput, 1); replyput = 0;
    //athread_put(PE_MODE,local_output,&out_ptr[offset],(local_inout_size-offset)/SIMDSIZE*sizeof(SIMDType),&replyput,0,0);
    //while(replyput!=1); replyput=0;
  }
  ldm_free(local_input, sizeof(SIMDType)*__RELU_BUFFSIZE_1);
  ldm_free(local_output,sizeof(SIMDType)*__RELU_BUFFSIZE_1);

}

void relu_slave_backward_d(ReluDiffData* param)
{
  int count, start, local_count;
  int id = athread_get_id(-1);
  count = param->count;
  local_count = count/64 + (id<(count%64));
  start = id*(count/64)+id*(id<(count%64));
  int local_inout_size = local_count/SIMDSIZE;
  SIMDType* local_input = (SIMDType*) ldm_malloc(sizeof(Type)*__RELU_BUFFSIZE_2);
  SIMDType* local_diff = (SIMDType*) ldm_malloc(sizeof(Type)*__RELU_BUFFSIZE_2);
  SIMDType* local_output= (SIMDType*) ldm_malloc(sizeof(Type)*__RELU_BUFFSIZE_2);
  Type local_negative_slope = param->negative_slope.dbl;
  Type* in_ptr = param->in+start*sizeof(Type);
  Type* diff_ptr = param->diff+start*sizeof(Type);
  Type* out_ptr= param->out+start*sizeof(Type);
  volatile int input_replyget = 0, diff_replyget = 0, replyput = 0;
  int i,offset;


  dma_set_op(&dma_get_input, DMA_GET);
  dma_set_mode(&dma_get_input, PE_MODE);
  dma_set_reply(&dma_get_input, &input_replyget);

  dma_set_op(&dma_get_diff, DMA_GET);
  dma_set_mode(&dma_get_diff, PE_MODE);
  dma_set_reply(&dma_get_diff, &diff_replyget);

  dma_set_op(&dma_put_output, DMA_PUT);
  dma_set_mode(&dma_put_output, PE_MODE);
  dma_set_reply(&dma_put_output, &replyput);

  //DMA for local_input(local_count)
  dma_set_size(&dma_get_input, __RELU_BUFFSIZE_2*sizeof(SIMDType));
  //dma_set_bsize(&dma_get_input,__RELU_BUFFSIZE_2*sizeof(SIMDType));

  //DMA for local_diff(local_count)
  dma_set_size(&dma_get_diff, __RELU_BUFFSIZE_2*sizeof(SIMDType));
  //dma_set_bsize(&dma_get_diff,__RELU_BUFFSIZE_2*sizeof(SIMDType));

  //DMA for local_output(local_count)
  dma_set_size(&dma_put_output, __RELU_BUFFSIZE_2*sizeof(SIMDType));
  //dma_set_bsize(&dma_put_output,__RELU_BUFFSIZE_2*sizeof(SIMDType));

  for(offset=0;offset+__RELU_BUFFSIZE_2-1<local_inout_size;offset+=__RELU_BUFFSIZE_2) {
    // begin DMA
    dma(dma_get_input, (long)(in_ptr+offset), (long)(local_input));
    dma_wait(&input_replyget, 1); input_replyget = 0;

    dma(dma_get_diff, (long)(diff_ptr+offset), (long)(local_diff));
    dma_wait(&diff_replyget, 1); diff_replyget = 0;
    //athread_get(PE_MODE,&in_ptr[offset],local_input,__RELU_BUFFSIZE_2/SIMDSIZE*sizeof(SIMDType),&input_replyget,0,0,0);
    //athread_get(PE_MODE,&diff_ptr[offset],local_diff,__RELU_BUFFSIZE_2/SIMDSIZE*sizeof(SIMDType),&input_replyget,0,0,0);
    //while(input_replyget!=2); input_replyget=0;

    for(i=0;i<__RELU_BUFFSIZE_2;++i) {
      //local_diff[i] = diff_ptr[offset+i];
      //local_input[i]= in_ptr[offset+i];
      local_output[i] = local_diff[i] * ((local_input[i]>0)
          + local_negative_slope*(local_input[i]<=0));
      //out_ptr[offset+i]=local_output[i];
    }

    dma(dma_put_output, (long)(out_ptr+offset), (long)(local_output));
    dma_wait(&replyput, 1); replyput = 0;
    //athread_put(PE_MODE,local_output,&out_ptr[offset],__RELU_BUFFSIZE_2/SIMDSIZE*sizeof(SIMDType),&replyput,0,0);
    //while(replyput!=1); replyput=0;
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
    //athread_get(PE_MODE,&in_ptr[offset],local_input,(local_inout_size-offset)*sizeof(SIMDType),&input_replyget,0,0,0);
    //athread_get(PE_MODE,&diff_ptr[offset],local_diff,(local_inout_size-offset)*sizeof(SIMDType),&input_replyget,0,0,0);
    //while(input_replyget!=2); input_replyget=0;

    for(i=0;i<local_inout_size-offset;++i) {
      local_output[i] = local_diff[i] * ((local_input[i]>0)
          + local_negative_slope*(local_input[i]<=0));
    }

    dma(dma_put_output, (long)(out_ptr+offset), (long)(local_output));
    dma_wait(&replyput, 1); replyput = 0;
    //athread_put(PE_MODE,local_output,&out_ptr[offset],(local_inout_size-offset)*sizeof(SIMDType),&replyput,0,0);
    //while(replyput!=1); replyput=0;
  }
  ldm_free(local_input ,sizeof(SIMDType)*__RELU_BUFFSIZE_2);
  ldm_free(local_diff  ,sizeof(SIMDType)*__RELU_BUFFSIZE_2);
  ldm_free(local_output,sizeof(SIMDType)*__RELU_BUFFSIZE_2);

}
#undef Type
#undef SIMDSIZE
#undef SIMDType

// <float> slave functions
#define Type float
#define SIMDSIZE 1
#define SIMDType float
void relu_slave_forward_f(ReluData* param)
{
  int count, start, local_count;
  int id = athread_get_id(-1);
  count = param->count;
  local_count = count/64 + (id<(count%64));
  start = id*(count/64)+id*(id<(count%64));
  int local_inout_size = local_count/SIMDSIZE;
  SIMDType* local_input = (SIMDType*) ldm_malloc(sizeof(Type)*__RELU_BUFFSIZE_1);
  SIMDType* local_output= (SIMDType*) ldm_malloc(sizeof(Type)*__RELU_BUFFSIZE_1);
  Type local_negative_slope = param->negative_slope.flt;
  Type* in_ptr = param->in+start*sizeof(Type);
  Type* out_ptr= param->out+start*sizeof(Type);
  volatile int input_replyget = 0, replyput = 0;
  int i,offset;

  //dma_desc dma_get_input, dma_put_output;

  dma_set_op(&dma_get_input, DMA_GET);
  dma_set_mode(&dma_get_input, PE_MODE);
  dma_set_reply(&dma_get_input, &input_replyget);

  dma_set_op(&dma_put_output, DMA_PUT);
  dma_set_mode(&dma_put_output, PE_MODE);
  dma_set_reply(&dma_put_output, &replyput);

  //DMA for local_input(local_count)
  dma_set_size(&dma_get_input, __RELU_BUFFSIZE_1/SIMDSIZE*sizeof(SIMDType));
  //dma_set_bsize(&dma_get_input,__RELU_BUFFSIZE_1/SIMDSIZE*sizeof(SIMDType));

  //DMA for local_output(local_count)
  dma_set_size(&dma_put_output, __RELU_BUFFSIZE_1/SIMDSIZE*sizeof(SIMDType));
  //dma_set_bsize(&dma_put_output,__RELU_BUFFSIZE_1/SIMDSIZE*sizeof(SIMDType));

  for(offset=0;offset+__RELU_BUFFSIZE_1-1<local_inout_size;offset+=__RELU_BUFFSIZE_1)
  {
    // begin DMA
    dma(dma_get_input, (long)(in_ptr+offset), (long)(local_input));
    dma_wait(&input_replyget, 1); input_replyget = 0;
    //athread_get(PE_MODE,&in_ptr[offset],local_input,__RELU_BUFFSIZE_1/SIMDSIZE*sizeof(SIMDType),&input_replyget,0,0,0);
    //while(input_replyget!=1); input_replyget=0;

    for(i=0;i<__RELU_BUFFSIZE_1;++i) {
      local_output[i] = max(local_input[i],0)
          + local_negative_slope*min(local_input[i],0);
    }

    dma(dma_put_output, (long)(out_ptr+offset), (long)(local_output));
    dma_wait(&replyput, 1); replyput = 0;
    //athread_put(PE_MODE,local_output,&out_ptr[offset],__RELU_BUFFSIZE_1/SIMDSIZE*sizeof(SIMDType),&replyput,0,0);
    //while(replyput!=1); replyput=0;
  }
  // calculate the rest
  if(offset<local_inout_size) {
    dma_set_size(&dma_get_input, (local_inout_size-offset)/SIMDSIZE*sizeof(SIMDType));
    dma_set_size(&dma_put_output,(local_inout_size-offset)/SIMDSIZE*sizeof(SIMDType));
    // begin DMA
    dma(dma_get_input, (long)(in_ptr+offset), (long)(local_input));
    dma_wait(&input_replyget, 1); input_replyget = 0;

    //athread_get(PE_MODE,&in_ptr[offset],local_input,(local_inout_size-offset)/SIMDSIZE*sizeof(SIMDType),&input_replyget,0,0,0);
    //while(input_replyget!=1); input_replyget=0;
    for(i=0;i<local_inout_size-offset;++i) {
      local_output[i] = max(local_input[i],0)
          + local_negative_slope*min(local_input[i],0);
    }

    dma(dma_put_output, (long)(out_ptr+offset), (long)(local_output));
    dma_wait(&replyput, 1); replyput = 0;
    //athread_put(PE_MODE,local_output,&out_ptr[offset],(local_inout_size-offset)/SIMDSIZE*sizeof(SIMDType),&replyput,0,0);
    //while(replyput!=1); replyput=0;
  }
  ldm_free(local_input, sizeof(SIMDType)*__RELU_BUFFSIZE_1);
  ldm_free(local_output,sizeof(SIMDType)*__RELU_BUFFSIZE_1);

}

void relu_slave_backward_f(ReluDiffData* param)
{
  int count, start, end, local_count;
  int id = athread_get_id(-1);
  count = param->count;
  local_count = count/64 + (id<(count%64));
  start = id*(count/64)+id*(id<(count%64));
  end = start+local_count;
  int local_inout_size = local_count/SIMDSIZE;
  SIMDType* local_input = (SIMDType*) ldm_malloc(sizeof(Type)*__RELU_BUFFSIZE_2);
  SIMDType* local_diff = (SIMDType*) ldm_malloc(sizeof(Type)*__RELU_BUFFSIZE_2);
  SIMDType* local_output= (SIMDType*) ldm_malloc(sizeof(Type)*__RELU_BUFFSIZE_2);
  Type local_negative_slope = param->negative_slope.flt;
  Type* in_ptr = param->in+start*sizeof(Type);
  Type* diff_ptr = param->diff+start*sizeof(Type);
  Type* out_ptr= param->out+start*sizeof(Type);
  volatile int input_replyget = 0, diff_replyget = 0, replyput = 0;
  int i,offset;

  //dma_desc dma_get_input, dma_get_diff, dma_put_output;

  dma_set_op(&dma_get_input, DMA_GET);
  dma_set_mode(&dma_get_input, PE_MODE);
  dma_set_reply(&dma_get_input, &input_replyget);

  dma_set_op(&dma_get_diff, DMA_GET);
  dma_set_mode(&dma_get_diff, PE_MODE);
  dma_set_reply(&dma_get_diff, &diff_replyget);

  dma_set_op(&dma_put_output, DMA_PUT);
  dma_set_mode(&dma_put_output, PE_MODE);
  dma_set_reply(&dma_put_output, &replyput);

  //DMA for local_input(local_count)
  dma_set_size(&dma_get_input, __RELU_BUFFSIZE_2*sizeof(SIMDType));
  //dma_set_bsize(&dma_get_input,__RELU_BUFFSIZE_2*sizeof(SIMDType));

  //DMA for local_diff(local_count)
  dma_set_size(&dma_get_diff, __RELU_BUFFSIZE_2*sizeof(SIMDType));
  //dma_set_bsize(&dma_get_diff,__RELU_BUFFSIZE_2*sizeof(SIMDType));

  //DMA for local_output(local_count)
  dma_set_size(&dma_put_output, __RELU_BUFFSIZE_2*sizeof(SIMDType));
  //dma_set_bsize(&dma_put_output,__RELU_BUFFSIZE_2*sizeof(SIMDType));

  for(offset=0;offset+__RELU_BUFFSIZE_2-1<local_inout_size;offset+=__RELU_BUFFSIZE_2) {
    // begin DMA
    dma(dma_get_input, (long)(in_ptr+offset), (long)(local_input));
    dma_wait(&input_replyget, 1); input_replyget = 0;

    dma(dma_get_diff, (long)(diff_ptr+offset), (long)(local_diff));
    dma_wait(&diff_replyget, 1); diff_replyget = 0;
    //athread_get(PE_MODE,&in_ptr[offset],local_input,__RELU_BUFFSIZE_2/SIMDSIZE*sizeof(SIMDType),&input_replyget,0,0,0);
    //athread_get(PE_MODE,&diff_ptr[offset],local_diff,__RELU_BUFFSIZE_2/SIMDSIZE*sizeof(SIMDType),&input_replyget,0,0,0);
    //while(input_replyget!=2); input_replyget=0;

    for(i=0;i<__RELU_BUFFSIZE_2;++i) {
      local_output[i] = local_diff[i] * ((local_input[i]>0)
          + local_negative_slope*(local_input[i]<=0));
    }

    dma(dma_put_output, (long)(out_ptr+offset), (long)(local_output));
    dma_wait(&replyput, 1); replyput = 0;
    //athread_put(PE_MODE,local_output,&out_ptr[offset],__RELU_BUFFSIZE_2/SIMDSIZE*sizeof(SIMDType),&replyput,0,0);
    //while(replyput!=1); replyput=0;
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

    //athread_get(PE_MODE,&in_ptr[offset],local_input,(local_inout_size-offset)*sizeof(SIMDType),&input_replyget,0,0,0);
    //athread_get(PE_MODE,&diff_ptr[offset],local_diff,(local_inout_size-offset)*sizeof(SIMDType),&input_replyget,0,0,0);
    //while(input_replyget!=2); input_replyget=0;
    for(i=0;i<local_inout_size-offset;++i) {
      local_output[i] = local_diff[i] * ((local_input[i]>0)
          + local_negative_slope*(local_input[i]<=0));
    }

    dma(dma_put_output, (long)(out_ptr+offset), (long)(local_output));
    dma_wait(&replyput, 1); replyput = 0;
    //athread_put(PE_MODE,local_output,&out_ptr[offset],(local_inout_size-offset)*sizeof(SIMDType),&replyput,0,0);
    //while(replyput!=1); replyput=0;
  }
  ldm_free(local_input ,sizeof(SIMDType)*__RELU_BUFFSIZE_2);
  ldm_free(local_diff  ,sizeof(SIMDType)*__RELU_BUFFSIZE_2);
  ldm_free(local_output,sizeof(SIMDType)*__RELU_BUFFSIZE_2);

}
#undef Type
#undef SIMDSIZE
#undef SIMDType
