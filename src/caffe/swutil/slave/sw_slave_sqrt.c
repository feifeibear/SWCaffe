#include <slave.h>
#include <simd.h>
#include <dma.h>
#include <math.h>
// BUFFSIZE: number of float/double numbers in LDM buffer

__thread_local_fix dma_desc dma_get_sqrt_src, dma_put_sqrt_dst;

typedef struct SqrtPara_st {
  void *src;
  void *dst;
  int count;
}SqrtPara;

void sw_slave_sqrt_d(SqrtPara *para) {
  const int BUFFSIZE = 4*1024;
  const int SIMDSIZE = 4;
  const int SPNUM = 64;
  double* local_src = (double*)ldm_malloc(BUFFSIZE*sizeof(double));
  double*  local_dst = (double *)ldm_malloc(BUFFSIZE*sizeof(double));
  doublev4 vsrc;
  doublev4 vdst;
  int id = athread_get_id(-1);
  int count = para->count;
  int local_count = count/SPNUM + (id<(count%SPNUM));
  int start = id*(count/SPNUM) + (id<(count%SPNUM)?id:(count%SPNUM));
  double* src_ptr = &(((double*)para->src)[start]);
  double*  dst_ptr = &(((double *)para->dst)[start]);
  volatile int replyget=0, replyput=0;
  int i,off;
  // DMA settings
  //dma_desc dma_get_sqrt_src, dma_put_sqrt_dst;

  dma_set_op(&dma_get_sqrt_src, DMA_GET);
  dma_set_mode(&dma_get_sqrt_src, PE_MODE);
  dma_set_reply(&dma_get_sqrt_src, &replyget);

  dma_set_op(&dma_put_sqrt_dst, DMA_PUT);
  dma_set_mode(&dma_put_sqrt_dst, PE_MODE);
  dma_set_reply(&dma_put_sqrt_dst, &replyput);

  dma_set_size(&dma_get_sqrt_src,BUFFSIZE*sizeof(double));
  dma_set_size(&dma_put_sqrt_dst,BUFFSIZE*sizeof(double));

  for(off = 0; off+BUFFSIZE-1<local_count; off+=BUFFSIZE)
  {
    // DMA get a block
    dma(dma_get_sqrt_src, (long)(src_ptr+off), (long)(local_src));
    dma_wait(&replyget, 1); replyget = 0;
    
    for(i=0; i<BUFFSIZE; i+=SIMDSIZE) {
      simd_load(vsrc,&local_src[i]);
      vdst = simd_vsqrtd(vsrc); // sqrt
      simd_store(vdst,&local_dst[i]);
    }

    for(i=0; i<BUFFSIZE; i++) local_dst[i] = sqrt(local_src[i]);
    // DMA put result
    dma(dma_put_sqrt_dst, (long)(dst_ptr+off), (long)(local_dst));
    dma_wait(&replyput, 1); replyput = 0;
  }

  if(off<local_count) {
    dma_set_size(&dma_get_sqrt_src,(local_count-off)*sizeof(double));
    dma(dma_get_sqrt_src, (long)(src_ptr+off), (long)(local_src));
    dma_wait(&replyget, 1); replyget = 0;
    
    //for(i=0; i<local_count-off; i++) local_dst[i] = sqrt(local_src[i]);

    for(i=0; i+SIMDSIZE-1<local_count-off; i+=SIMDSIZE) {
      simd_load(vsrc,&local_src[i]);
      vdst = simd_vsqrtd(vsrc); // sqrt
      simd_store(vdst,&local_dst[i]);
    }
    for(; i<local_count-off; i++) {
      local_dst[i] = sqrt(local_src[i]);
    }
    dma_set_size(&dma_put_sqrt_dst,(local_count-off)*sizeof(double));
    dma(dma_put_sqrt_dst, (long)(dst_ptr+off), (long)(local_dst));
    dma_wait(&replyput, 1); replyput = 0;

  }

  ldm_free(local_src, BUFFSIZE*sizeof(double));
  ldm_free(local_dst, BUFFSIZE*sizeof(double));
}
void sw_slave_sqrt_f(SqrtPara *para) {
  const int BUFFSIZE = 4*1024;
  const int SIMDSIZE = 4;
  const int SPNUM = 64;
  float * local_src = (float *)ldm_malloc(BUFFSIZE*sizeof(float ));
  float * local_dst = (float *)ldm_malloc(BUFFSIZE*sizeof(float ));
  floatv4 vsrc;
  floatv4 vdst;
  int id = athread_get_id(-1);
  int count = para->count;
  int local_count = count/SPNUM + (id<(count%SPNUM));
  int start = id*(count/SPNUM) + (id<(count%SPNUM)?id:(count%SPNUM));
  float * src_ptr = &(((float *)para->src)[start]);
  float * dst_ptr = &(((float *)para->dst)[start]);
  volatile int replyget=0, replyput=0;
  int i,off;
  // DMA settings
  //dma_desc dma_get_sqrt_src, dma_put_sqrt_dst;

  dma_set_op(&dma_get_sqrt_src, DMA_GET);
  dma_set_mode(&dma_get_sqrt_src, PE_MODE);
  dma_set_reply(&dma_get_sqrt_src, &replyget);

  dma_set_op(&dma_put_sqrt_dst, DMA_PUT);
  dma_set_mode(&dma_put_sqrt_dst, PE_MODE);
  dma_set_reply(&dma_put_sqrt_dst, &replyput);

  dma_set_size(&dma_get_sqrt_src,BUFFSIZE*sizeof(float));
  dma_set_size(&dma_put_sqrt_dst,BUFFSIZE*sizeof(float));

  for(off = 0; off+BUFFSIZE-1<local_count; off+=BUFFSIZE)
  {
    // DMA get a block
    dma(dma_get_sqrt_src, (long)(src_ptr+off), (long)(local_src));
    dma_wait(&replyget, 1); replyget = 0;

    for(i=0; i<BUFFSIZE; i+=SIMDSIZE) {
      simd_load(vsrc,&local_src[i]);
      vdst = simd_vsqrts(vsrc); // sqrt
      simd_store(vdst,&local_dst[i]);
    }
 
    //for(i=0; i<BUFFSIZE; i++) local_dst[i] = sqrt(local_src[i]);
    // DMA put result
    dma(dma_put_sqrt_dst, (long)(dst_ptr+off), (long)(local_dst));
    dma_wait(&replyput, 1); replyput = 0;
  }

  if(off<local_count) {
    dma_set_size(&dma_get_sqrt_src,(local_count-off)*sizeof(float));
    dma(dma_get_sqrt_src, (long)(src_ptr+off), (long)(local_src));
    dma_wait(&replyget, 1); replyget = 0;

    //for(i=0; i<local_count-off; i++) local_dst[i] = sqrt(local_src[i]);
    
    for(i=0; i+SIMDSIZE-1<local_count-off; i+=SIMDSIZE) {
      simd_load(vsrc,&local_src[i]);
      vdst = simd_vsqrts(vsrc); // sqrt
      simd_store(vdst,&local_dst[i]);
    }
    for(; i<local_count-off; i++) {
      local_dst[i] = sqrt(local_src[i]);
    }
    dma_set_size(&dma_put_sqrt_dst,(local_count-off)*sizeof(float));
    dma(dma_put_sqrt_dst, (long)(dst_ptr+off), (long)(local_dst));
    dma_wait(&replyput, 1); replyput = 0;

  }
  ldm_free(local_src, BUFFSIZE*sizeof(float));
  ldm_free(local_dst, BUFFSIZE*sizeof(float));
}
