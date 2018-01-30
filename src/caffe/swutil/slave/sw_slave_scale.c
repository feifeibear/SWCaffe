#include <slave.h>
#include <simd.h>
#include <dma.h>
// BUFFSIZE: number of float/double numbers in LDM buffer
#define BUFFSIZE 2*1024
#define SIMDSIZE 4
#define SIMDTYPED doublev4
#define SIMDTYPEF floatv4
#define SPNUM 64

__thread_local dma_desc dma_get_src, dma_put_dst;

typedef struct ScalePra_st {
  void *src;
  void *scale;
  void *dst;
  int outer_dim,inner_dim,scale_dim;
}ScalePra;

void sw_slave_scale_d(ScalePra *para) {
  double* local_src = (double*)ldm_malloc(BUFFSIZE*sizeof(double));
  double*  local_dst = (double *)ldm_malloc(BUFFSIZE*sizeof(double));
  SIMDTYPED vsrc,vscale;
  SIMDTYPED vdst;
  int id = athread_get_id(-1);
  int outer_dim = para->outer_dim;
  int inner_dim = para->inner_dim;
  int scale_dim = para->scale_dim;
  int count = outer_dim*inner_dim*scale_dim;
  int local_count = count/SPNUM + (id<(count%SPNUM));
  int start = id*(count/SPNUM) + (id<(count%SPNUM)?id:(count%SPNUM));
  double* src_ptr = &(((double*)para->src)[start]);
  double* scale_ptr = ((double*)para->scale);
  double* dst_ptr = &(((double *)para->dst)[start]);
  volatile int replyget=0, replyput=0;
  int i,off;
  // DMA settings
  //dma_desc dma_get_src, dma_put_dst;

  dma_set_op(&dma_get_src, DMA_GET);
  dma_set_mode(&dma_get_src, PE_MODE);
  dma_set_reply(&dma_get_src, &replyget);

  dma_set_op(&dma_put_dst, DMA_PUT);
  dma_set_mode(&dma_put_dst, PE_MODE);
  dma_set_reply(&dma_put_dst, &replyput);

  dma_set_size(&dma_put_dst,BUFFSIZE*sizeof(double));

  double* local_scale = (double*)ldm_malloc(scale_dim*sizeof(double));
  dma_set_size(&dma_get_src,scale_dim*sizeof(double));
  dma(dma_get_src, (long)(scale_ptr), (long)(local_scale));
  dma_wait(&replyget, 1); replyget = 0;

  dma_set_size(&dma_get_src,BUFFSIZE*sizeof(double));
  int base_size = outer_dim * inner_dim;
  int index = 0; 
  for(off = 0; off+BUFFSIZE-1<local_count; off+=BUFFSIZE)
  {
    // DMA get a block
    dma(dma_get_src, (long)(src_ptr+off), (long)(local_src));
    dma_wait(&replyget, 1); replyget = 0;
    for(i=0; i<BUFFSIZE; i+=SIMDSIZE) {
      simd_load(vsrc,&local_src[i]);
      index = (off+i) / base_size;
      simd_loade(vscale,&local_scale[index]);
      vdst = simd_vmuld(vsrc,vscale); // 
      simd_store(vdst,&local_dst[i]);
    }

    // DMA put result
    dma(dma_put_dst, (long)(dst_ptr+off), (long)(local_dst));
    dma_wait(&replyput, 1); replyput = 0;
  }

  if(off<local_count) {
    dma_set_size(&dma_get_src,(local_count-off)*sizeof(double));
    dma(dma_get_src, (long)(src_ptr+off), (long)(local_src));
    dma_wait(&replyget, 1); replyget = 0;

    for(i=0; i+SIMDSIZE-1<local_count-off; i+=SIMDSIZE) {
      simd_load(vsrc,&local_src[i]);
      index = i / base_size;
      simd_loade(vscale,&local_scale[index]);
      vdst = simd_vmuld(vsrc,vscale); // 
      simd_store(vdst,&local_dst[i]);
    }
    for(;i<local_count-off;i++) {
      index = i / base_size;
      local_dst[i]=local_src[i]*local_scale[index];
    }
    dma_set_size(&dma_put_dst,(local_count-off)*sizeof(double));
    dma(dma_put_dst, (long)(dst_ptr+off), (long)(local_dst));
    dma_wait(&replyput, 1); replyput = 0;

  }

  ldm_free(local_src, BUFFSIZE*sizeof(double));
  ldm_free(local_scale, scale_dim*sizeof(double));
  ldm_free(local_dst, BUFFSIZE*sizeof(double));
}
#undef BUFFSIZE
#define BUFFSIZE 4*1024

void sw_slave_scale_f(ScalePra *para) {
  float * local_src = (float *)ldm_malloc(BUFFSIZE*sizeof(float ));
  float * local_dst = (float *)ldm_malloc(BUFFSIZE*sizeof(float ));
  SIMDTYPEF vsrc,vscale;
  SIMDTYPEF vdst;
  int id = athread_get_id(-1);
  int outer_dim = para->outer_dim;
  int inner_dim = para->inner_dim;
  int scale_dim = para->scale_dim;
  int count = outer_dim*inner_dim*scale_dim;
  int local_count = count/SPNUM + (id<(count%SPNUM));
  int start = id*(count/SPNUM) + (id<(count%SPNUM)?id:(count%SPNUM));
  float * src_ptr = &(((float *)para->src)[start]);
  float * scale_ptr = ((float*)para->scale);
  float * dst_ptr = &(((float *)para->dst)[start]);
  volatile int replyget=0, replyput=0;
  int i,off;
  // DMA settings
  //dma_desc dma_get_src, dma_put_dst;

  dma_set_op(&dma_get_src, DMA_GET);
  dma_set_mode(&dma_get_src, PE_MODE);
  dma_set_reply(&dma_get_src, &replyget);

  dma_set_op(&dma_put_dst, DMA_PUT);
  dma_set_mode(&dma_put_dst, PE_MODE);
  dma_set_reply(&dma_put_dst, &replyput);

  dma_set_size(&dma_put_dst,BUFFSIZE*sizeof(float));
  dma_set_size(&dma_put_dst,BUFFSIZE*sizeof(float));

  float * local_scale = (float *)ldm_malloc(scale_dim*sizeof(float ));
  dma_set_size(&dma_get_src,scale_dim*sizeof(float));
  dma(dma_get_src, (long)(scale_ptr), (long)(local_scale));
  dma_wait(&replyget, 1); replyget = 0;

  dma_set_size(&dma_get_src,BUFFSIZE*sizeof(float));
  int base_size = outer_dim * inner_dim;
  int index = 0; 

  for(off = 0; off+BUFFSIZE-1<local_count; off+=BUFFSIZE)
  {
    // DMA get a block
    dma(dma_get_src, (long)(src_ptr+off), (long)(local_src));
    dma_wait(&replyget, 1); replyget = 0;

    for(i=0; i<BUFFSIZE; i+=SIMDSIZE) {
      simd_load(vsrc,&local_src[i]);
      index = (off+i) / base_size;
      simd_loade(vscale,&local_scale[index]);
      vdst = simd_vmuls(vsrc,vscale); // 
      simd_store(vdst,&local_dst[i]);
    }

    // DMA put result
    dma(dma_put_dst, (long)(dst_ptr+off), (long)(local_dst));
    dma_wait(&replyput, 1); replyput = 0;
  }

  if(off<local_count) {
    dma_set_size(&dma_get_src,(local_count-off)*sizeof(float));
    dma(dma_get_src, (long)(src_ptr+off), (long)(local_src));
    dma_wait(&replyget, 1); replyget = 0;

    for(i=0; i+SIMDSIZE-1<local_count-off; i+=SIMDSIZE) {
      simd_load(vsrc,&local_src[i]);
      index = i / base_size;
      simd_loade(vscale,&local_scale[index]);
      vdst = simd_vmuls(vsrc,vscale); // 
      simd_store(vdst,&local_dst[i]);
    }
    for(;i<local_count-off;i++) {
      index = i / base_size;
      local_dst[i]=local_src[i]*local_scale[index];
    }
    dma_set_size(&dma_put_dst,(local_count-off)*sizeof(float));
    dma(dma_put_dst, (long)(dst_ptr+off), (long)(local_dst));
    dma_wait(&replyput, 1); replyput = 0;

  }

  ldm_free(local_src, BUFFSIZE*sizeof(float));
  ldm_free(local_scale, scale_dim*sizeof(float));
  ldm_free(local_dst, BUFFSIZE*sizeof(float));
}

