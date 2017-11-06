/******************************************
 * Created by Xin You
 * Date: 2017/8/4
 * Data transfer functions: (in SPEs)
 * 1. double to float: sw_double2float(double* src, float* dst, int count)
 * 2. float to double: sw_float2double(float* src, double* dst, int count)
 * ***************************************/
#include <slave.h>
#include <simd.h>
#include <dma.h>
// BUFFSIZE: number of float/double numbers in LDM buffer
#define BUFFSIZE 4*1024
#define SIMDSIZE 4
#define SIMDTYPED doublev4
#define SIMDTYPEF floatv4
#define SPNUM 64

//__thread_local dma_desc dma_get_src, dma_put_dst;
__thread_local dma_desc dma_get_src, dma_put_dst, dma_get_src_ua, dma_put_dst_ua;

typedef struct DataTransPara_st {
  void *src;
  void *dst;
  int count;
}DataTransPara;

void sw_double2float(DataTransPara *para) {
  double* local_src = (double*)ldm_malloc(BUFFSIZE*sizeof(double));
  float*  local_dst = (float *)ldm_malloc(BUFFSIZE*sizeof(float));
  SIMDTYPED vsrc;
  SIMDTYPEF vdst;
  int id = athread_get_id(-1);
  int count = para->count;
  int local_count = count/SPNUM + (id<(count%SPNUM));
  int start = id*count/SPNUM + (id<(count%SPNUM)?id:(count%SPNUM));
  double* src_ptr = &((double*)para->src)[start];
  float*  dst_ptr = &((float *)para->dst)[start];
  int blockNum = local_count/BUFFSIZE;
  int restNum = local_count - blockNum*BUFFSIZE;
  volatile int input_replyget=0, replyput=0;
  int i,off;
  // DMA settings

  dma_set_op(&dma_get_src, DMA_GET);
  dma_set_mode(&dma_get_src, PE_MODE);
  dma_set_reply(&dma_get_src, &input_replyget);

  dma_set_op(&dma_put_dst, DMA_PUT);
  dma_set_mode(&dma_put_dst, PE_MODE);
  dma_set_reply(&dma_put_dst, &replyput);

  dma_set_size(&dma_get_src,BUFFSIZE*sizeof(double));
  dma_set_size(&dma_put_dst,BUFFSIZE*sizeof(float));

  for(off = 0; off+BUFFSIZE-1<local_count; off+=BUFFSIZE)
  {
    // DMA get a block
    dma(dma_get_src, (long)(src_ptr+off), (long)(local_src));
    dma_wait(&input_replyget, 1); input_replyget = 0;

    for(i=0; i<BUFFSIZE; i+=SIMDSIZE) {
      simd_load(vsrc,&local_src[i]);
      vdst = (SIMDTYPEF)vsrc; // convert double to float
      simd_store(vdst,&local_dst[i]);
    }

    // DMA put result
    dma(dma_put_dst, (long)(dst_ptr+off), (long)(local_dst));
    dma_wait(&replyput, 1); replyput = 0;
  }

  if(off<local_count) {
    /*
    dma_set_op(&dma_get_src_ua, DMA_GET);
    dma_set_mode(&dma_get_src_ua, PE_MODE);
    dma_set_reply(&dma_get_src_ua, &input_replyget);

    dma_set_op(&dma_put_dst_ua, DMA_PUT);
    dma_set_mode(&dma_put_dst_ua, PE_MODE);
    dma_set_reply(&dma_put_dst_ua, &replyput);

    dma_set_size(&dma_get_src_ua,restNum*sizeof(double));
    dma_set_size(&dma_put_dst_ua,restNum*sizeof(float));
    dma(dma_get_src_ua,(long)(src_ptr+blockNum*BUFFSIZE),(long)local_src);
    dma_wait(&input_replyget, 1); input_replyget = 0;*/
    athread_get(PE_MODE,(src_ptr+off),local_src,(local_count-off)*sizeof(double),&input_replyget,0,0,0);
    while(input_replyget!=1) ; input_replyget = 0;

    for(i=0; i+SIMDSIZE-1<local_count-off; i+=SIMDSIZE) {
      simd_load(vsrc,&local_src[i]);
      vdst = (SIMDTYPEF)vsrc; // convert double to float
      simd_store(vdst,&local_dst[i]);
    }
    for(;i<local_count-off;i++) {
      local_dst[i]=local_src[i];
    }
    athread_put(PE_MODE,local_dst,(dst_ptr+off),(local_count-off)*sizeof(float),&replyput,0,0);
    while(replyput!=1) ; replyput = 0;

    //dma(dma_put_dst_ua,(long)(dst_ptr+blockNum*BUFFSIZE),(long)local_dst);
    //dma_wait(&replyput, 1); replyput = 0;
  }

  ldm_free(local_src, BUFFSIZE*sizeof(double));
  ldm_free(local_dst, BUFFSIZE*sizeof(float));
}

void sw_float2double(DataTransPara *para) {
  float * local_src = (float *)ldm_malloc(BUFFSIZE*sizeof(float ));
  double* local_dst = (double*)ldm_malloc(BUFFSIZE*sizeof(double));
  SIMDTYPEF vsrc;
  SIMDTYPED vdst;
  int id = athread_get_id(-1);
  int count = para->count;
  int local_count = count/SPNUM + (id<(count%SPNUM));
  int start = id*count/SPNUM + (id<(count%SPNUM)?id:(count%SPNUM));
  float * src_ptr = &((float *)para->src)[start];
  double* dst_ptr = &((double*)para->dst)[start];
  int blockNum = local_count/BUFFSIZE;
  int restNum = local_count - blockNum*BUFFSIZE;
  volatile int input_replyget=0, replyput=0;
  int i,off;
  // DMA settings
  //dma_desc dma_get_src, dma_put_dst;
  //dma_desc dma_get_src_ua, dma_put_dst_ua;

  dma_set_op(&dma_get_src, DMA_GET);
  dma_set_mode(&dma_get_src, PE_MODE);
  dma_set_reply(&dma_get_src, &input_replyget);

  dma_set_op(&dma_put_dst, DMA_PUT);
  dma_set_mode(&dma_put_dst, PE_MODE);
  dma_set_reply(&dma_put_dst, &replyput);

  dma_set_size(&dma_get_src,BUFFSIZE*sizeof(float));
  dma_set_size(&dma_put_dst,BUFFSIZE*sizeof(double));

  for(off = 0; off+BUFFSIZE-1<local_count; off+=BUFFSIZE)
  {
    // DMA get a block
    dma(dma_get_src, (long)(src_ptr+off), (long)(local_src));
    dma_wait(&input_replyget, 1); input_replyget = 0;

    for(i=0; i<BUFFSIZE; i+=SIMDSIZE) {
      simd_load(vsrc,&local_src[i]);
      vdst = (SIMDTYPED)vsrc; // convert float to double
      simd_store(vdst,&local_dst[i]);
    }

    // DMA put result
    dma(dma_put_dst, (long)(dst_ptr+off), (long)(local_dst));
    dma_wait(&replyput, 1); replyput = 0;
  }

  if(off<local_count) {
    /*
    dma_set_op(&dma_get_src_ua, DMA_GET);
    dma_set_mode(&dma_get_src_ua, PE_MODE);
    dma_set_reply(&dma_get_src_ua, &input_replyget);

    dma_set_op(&dma_put_dst_ua, DMA_PUT);
    dma_set_mode(&dma_put_dst_ua, PE_MODE);
    dma_set_reply(&dma_put_dst_ua, &replyput);

    dma_set_size(&dma_get_src_ua,restNum*sizeof(float));
    dma_set_size(&dma_put_dst_ua,restNum*sizeof(double));
    dma(dma_get_src_ua,(long)(src_ptr+blockNum*BUFFSIZE),(long)local_src);
    dma_wait(&input_replyget, 1); input_replyget = 0;*/
    athread_get(PE_MODE,(src_ptr+off),local_src,(local_count-off)*sizeof(float),&input_replyget,0,0,0);
    while(input_replyget!=1) ; input_replyget = 0;

    for(i=0; i+SIMDSIZE-1<local_count-off; i+=SIMDSIZE) {
      simd_load(vsrc,&local_src[i]);
      vdst = (SIMDTYPED)vsrc; // convert float to double
      simd_store(vdst,&local_dst[i]);
    }
    for(;i<local_count-off;i++) {
      local_dst[i]=local_src[i];
    }

    athread_put(PE_MODE,local_dst,(dst_ptr+off),(local_count-off)*sizeof(double),&replyput,0,0);
    while(replyput!=1) ; replyput = 0;
    //dma(dma_put_dst_ua,(long)(dst_ptr+blockNum*BUFFSIZE),(long)local_dst);
    //dma_wait(&replyput, 1); replyput = 0;
  }

  ldm_free(local_src, BUFFSIZE*sizeof(float));
  ldm_free(local_dst, BUFFSIZE*sizeof(double));
}
