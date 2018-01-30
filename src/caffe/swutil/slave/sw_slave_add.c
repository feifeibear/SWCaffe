#include <slave.h>
#include <simd.h>
#include <dma.h>
// BUFFSIZE: number of float/double numbers in LDM buffer
/*#define BUFFSIZE 2*1024*/
#define SIMDSIZE 4
#define SIMDTYPED doublev4
#define SIMDTYPEF floatv4
/*#define SPNUM 64*/

__thread_local_fix dma_desc dma_get_src, dma_put_dst;

typedef struct addPara_st {
  void *src1;
  void *src2;
  void *dst;
  int count;
}addPara;

/*void sw_slave_add_d(addPara *para) {*/
  /*double* local_src1 = (double*)ldm_malloc(BUFFSIZE*sizeof(double));*/
  /*double* local_src2 = (double*)ldm_malloc(BUFFSIZE*sizeof(double));*/
  /*double* local_dst  = (double*)ldm_malloc(BUFFSIZE*sizeof(double));*/
  /*SIMDTYPED vsrc1,vsrc2;*/
  /*SIMDTYPED vdst;*/
  /*int id = athread_get_id(-1);*/
  /*int count = para->count;*/
  /*int local_count = count/SPNUM + (id<(count%SPNUM));*/
  /*int start = id*(count/SPNUM) + (id<(count%SPNUM)?id:(count%SPNUM));*/
  /*double* src_ptr1 = &(((double*)para->src1)[start]);*/
  /*double* src_ptr2 = &(((double*)para->src2)[start]);*/
  /*double* dst_ptr  = &(((double *)para->dst)[start]);*/
  /*volatile int replyget=0, replyput=0;*/
  /*int i,off;*/
  /*// DMA settings*/
  /*//dma_desc dma_get_src, dma_put_dst;*/

  /*dma_set_op(&dma_get_src, DMA_GET);*/
  /*dma_set_mode(&dma_get_src, PE_MODE);*/
  /*dma_set_reply(&dma_get_src, &replyget);*/

  /*dma_set_op(&dma_put_dst, DMA_PUT);*/
  /*dma_set_mode(&dma_put_dst, PE_MODE);*/
  /*dma_set_reply(&dma_put_dst, &replyput);*/

  /*dma_set_size(&dma_get_src,BUFFSIZE*sizeof(double));*/
  /*dma_set_size(&dma_put_dst,BUFFSIZE*sizeof(double));*/

  /*for(off = 0; off+BUFFSIZE-1<local_count; off+=BUFFSIZE)*/
  /*{*/
    /*// DMA get a block*/
    /*dma(dma_get_src, (long)(src_ptr1+off), (long)(local_src1));*/
    /*dma(dma_get_src, (long)(src_ptr2+off), (long)(local_src2));*/
    /*dma_wait(&replyget, 2); replyget = 0;*/

    /*for(i=0; i<BUFFSIZE; i+=SIMDSIZE) {*/
      /*simd_load(vsrc1,&local_src1[i]);*/
      /*simd_load(vsrc2,&local_src2[i]);*/
      /*vdst = vsrc1 + vsrc2; // */
      /*simd_store(vdst,&local_dst[i]);*/
    /*}*/

    /*// DMA put result*/
    /*dma(dma_put_dst, (long)(dst_ptr+off), (long)(local_dst));*/
    /*dma_wait(&replyput, 1); replyput = 0;*/
  /*}*/

  /*if(off<local_count) {*/
    /*dma_set_size(&dma_get_src,(local_count-off)*sizeof(double));*/
    /*dma(dma_get_src, (long)(src_ptr1+off), (long)(local_src1));*/
    /*dma(dma_get_src, (long)(src_ptr2+off), (long)(local_src2));*/
    /*dma_wait(&replyget, 2); replyget = 0;*/

    /*for(i=0; i+SIMDSIZE-1<local_count-off; i+=SIMDSIZE) {*/
      /*simd_load(vsrc1,&local_src1[i]);*/
      /*simd_load(vsrc2,&local_src2[i]);*/
      /*vdst = vsrc1 + vsrc2; // */
      /*simd_store(vdst,&local_dst[i]);*/
    /*}*/
    /*for(;i<local_count-off;i++) {*/
      /*local_dst[i]=local_src1[i]+local_src2[i];*/
    /*}*/
    /*dma_set_size(&dma_put_dst,(local_count-off)*sizeof(double));*/
    /*dma(dma_put_dst, (long)(dst_ptr+off), (long)(local_dst));*/
    /*dma_wait(&replyput, 1); replyput = 0;*/

  /*}*/

  /*ldm_free(local_src1, BUFFSIZE*sizeof(double));*/
  /*ldm_free(local_src2, BUFFSIZE*sizeof(double));*/
  /*ldm_free(local_dst, BUFFSIZE*sizeof(double));*/
/*}*/
/*#undef BUFFSIZE*/
/*#undef SPNUM*/
//modified by zwl
void sw_slave_add_f(addPara *para) {
  int id = athread_get_id(-1);
  id = (id%16)*4+(id/16);
  volatile unsigned long replyget1= 0;
  volatile unsigned long replyget2= 0;
  volatile unsigned long replyput = 0;
  const int BUFFSIZE = 256;
  const int SPNUM = 64;
  SIMDTYPEF vsrc1,vsrc2;
  SIMDTYPEF vdst;
  float * local_src1 = (float *)ldm_malloc(BUFFSIZE*sizeof(float));
  float * local_src2 = (float *)ldm_malloc(BUFFSIZE*sizeof(float));

  int count = para->count;

  int local_offset = id*BUFFSIZE;

  while(local_offset < count){
    float* src1_ptr = (float*)para->src1+local_offset;
    float* src2_ptr = (float*)para->src2+local_offset;
    float* dst_ptr = (float*)para->dst+local_offset;
    athread_get(PE_MODE, (void*)src1_ptr, (void*)local_src1, BUFFSIZE*sizeof(float), (void*)&replyget1, 0, 0, 0);
    athread_get(PE_MODE, (void*)src2_ptr, (void*)local_src2, BUFFSIZE*sizeof(float), (void*)&replyget2, 0, 0, 0);
    while(replyget1!=1);
    while(replyget2!=1);
    replyget1 = 0;
    replyget2 = 0;
    int i;
    for(i=0; i<BUFFSIZE; i+=SIMDSIZE) {
      simd_load(vsrc1,&local_src1[i]);
      simd_load(vsrc2,&local_src2[i]);
      vdst = vsrc1 + vsrc2; 
      simd_store(vdst,&local_src2[i]);
    }

    athread_put(PE_MODE, (void*)local_src2, (void*)dst_ptr, BUFFSIZE*sizeof(float), (void*)&replyput, 0, 0);
    while(replyput!=1);
    replyput = 0;
    local_offset += SPNUM*BUFFSIZE;
  }
  ldm_free(local_src1, BUFFSIZE*sizeof(float));
  ldm_free(local_src2, BUFFSIZE*sizeof(float));

}
/*void sw_slave_add_f(addPara *para) {*/
  /*float * local_src1 = (float *)ldm_malloc(BUFFSIZE*sizeof(float ));*/
  /*float * local_src2 = (float *)ldm_malloc(BUFFSIZE*sizeof(float ));*/
  /*float * local_dst  = (float *)ldm_malloc(BUFFSIZE*sizeof(float ));*/
  /*SIMDTYPEF vsrc1,vsrc2;*/
  /*SIMDTYPEF vdst;*/
  /*int id = athread_get_id(-1);*/
  /*int count = para->count;*/
  /*int local_count = count/SPNUM + (id<(count%SPNUM));*/
  /*int start = id*(count/SPNUM) + (id<(count%SPNUM)?id:(count%SPNUM));*/
  /*float * src_ptr1 = &(((float *)para->src1)[start]);*/
  /*float * src_ptr2 = &(((float *)para->src2)[start]);*/
  /*float * dst_ptr  = &(((float *)para->dst)[start]);*/
  /*volatile int replyget=0, replyput=0;*/
  /*int i,off;*/
  /*// DMA settings*/
  /*//dma_desc dma_get_src, dma_put_dst;*/

  /*dma_set_op(&dma_get_src, DMA_GET);*/
  /*dma_set_mode(&dma_get_src, PE_MODE);*/
  /*dma_set_reply(&dma_get_src, &replyget);*/

  /*dma_set_op(&dma_put_dst, DMA_PUT);*/
  /*dma_set_mode(&dma_put_dst, PE_MODE);*/
  /*dma_set_reply(&dma_put_dst, &replyput);*/

  /*dma_set_size(&dma_get_src,BUFFSIZE*sizeof(float));*/
  /*dma_set_size(&dma_put_dst,BUFFSIZE*sizeof(float));*/

  /*for(off = 0; off+BUFFSIZE-1<local_count; off+=BUFFSIZE)*/
  /*{*/
    /*// DMA get a block*/
    /*dma(dma_get_src, (long)(src_ptr1+off), (long)(local_src1));*/
    /*dma(dma_get_src, (long)(src_ptr2+off), (long)(local_src2));*/
    /*dma_wait(&replyget, 2); replyget = 0;*/

    /*for(i=0; i<BUFFSIZE; i+=SIMDSIZE) {*/
      /*simd_load(vsrc1,&local_src1[i]);*/
      /*simd_load(vsrc2,&local_src2[i]);*/
      /*vdst = vsrc1 + vsrc2; // */
      /*simd_store(vdst,&local_dst[i]);*/
    /*}*/

    /*// DMA put result*/
    /*dma(dma_put_dst, (long)(dst_ptr+off), (long)(local_dst));*/
    /*dma_wait(&replyput, 1); replyput = 0;*/
  /*}*/

  /*if(off<local_count) {*/
    /*dma_set_size(&dma_get_src,(local_count-off)*sizeof(float));*/
    /*dma(dma_get_src, (long)(src_ptr1+off), (long)(local_src1));*/
    /*dma(dma_get_src, (long)(src_ptr2+off), (long)(local_src2));*/
    /*dma_wait(&replyget, 2); replyget = 0;*/

    /*for(i=0; i+SIMDSIZE-1<local_count-off; i+=SIMDSIZE) {*/
      /*simd_load(vsrc1,&local_src1[i]);*/
      /*simd_load(vsrc2,&local_src2[i]);*/
      /*vdst = vsrc1 + vsrc2; // */
      /*simd_store(vdst,&local_dst[i]);*/
    /*}*/
    /*for(;i<local_count-off;i++) {*/
      /*local_dst[i]=local_src1[i]+local_src2[i];*/
    /*}*/
    /*dma_set_size(&dma_put_dst,(local_count-off)*sizeof(float));*/
    /*dma(dma_put_dst, (long)(dst_ptr+off), (long)(local_dst));*/
    /*dma_wait(&replyput, 1); replyput = 0;*/

  /*}*/

  /*ldm_free(local_src1, BUFFSIZE*sizeof(float));*/
  /*ldm_free(local_src2, BUFFSIZE*sizeof(float));*/
  /*ldm_free(local_dst, BUFFSIZE*sizeof(float));*/
/*}*/

