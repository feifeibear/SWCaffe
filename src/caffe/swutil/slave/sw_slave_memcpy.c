#include <slave.h>
#include <simd.h>
#include <dma.h>
// BUFFSIZE: number of float/double numbers in LDM buffer

__thread_local dma_desc dma_get_memcpy_src, dma_put_memcpy_dst;

typedef struct MemcpyPara_st {
  void *src;
  void *dst;
  int count,num;
}MemcpyPara;

inline void mb()
{
    asm volatile("":::"memory");
    asm volatile("memb");
}
void sw_slave_memcpy_d(MemcpyPara *para) {
  const int BUFFSIZE = 2*1024;
  const int SPNUM = 64;
  double* local_src = (double*)ldm_malloc(BUFFSIZE*sizeof(double));
  int id = athread_get_id(-1);
  int count = para->count;
  int local_count = count/SPNUM + (id<(count%SPNUM));
  int start = id*(count/SPNUM) + (id<(count%SPNUM)?id:(count%SPNUM));
  double * src_ptr = (double *)(para->src+start*sizeof(double));
  double * dst_ptr = (double *)(para->dst+start*sizeof(double));
  volatile int replyget=0, replyput=0;
  int i,off;
  // DMA settings
  //dma_desc dma_get_memcpy_src, dma_put_memcpy_dst;

  dma_set_op(&dma_get_memcpy_src, DMA_GET);
  dma_set_mode(&dma_get_memcpy_src, PE_MODE);
  dma_set_reply(&dma_get_memcpy_src, &replyget);

  dma_set_op(&dma_put_memcpy_dst, DMA_PUT);
  dma_set_mode(&dma_put_memcpy_dst, PE_MODE);
  dma_set_reply(&dma_put_memcpy_dst, &replyput);

  dma_set_size(&dma_get_memcpy_src,BUFFSIZE*sizeof(double));
  dma_set_size(&dma_put_memcpy_dst,BUFFSIZE*sizeof(double));

  for(off = 0; off+BUFFSIZE-1<local_count; off+=BUFFSIZE)
  {
    // DMA get a block
    dma(dma_get_memcpy_src, (long)(src_ptr+off), (long)(local_src));
    dma_wait(&replyget, 1); replyget = 0;
    //dma_barrier();
    mb();
    // DMA put result
    dma(dma_put_memcpy_dst, (long)(dst_ptr+off), (long)(local_src));
    dma_wait(&replyput, 1); replyput = 0;
  }

  if(off<local_count) {
    dma_set_size(&dma_get_memcpy_src,(local_count-off)*sizeof(double));
    dma_set_size(&dma_put_memcpy_dst,(local_count-off)*sizeof(double));
    dma(dma_get_memcpy_src, (long)(src_ptr+off), (long)(local_src));
    dma_wait(&replyget, 1); replyget = 0;
    //dma_barrier();
    mb();
    // DMA put result
    dma(dma_put_memcpy_dst, (long)(dst_ptr+off), (long)(local_src));
    dma_wait(&replyput, 1); replyput = 0;
  }

  ldm_free(local_src, BUFFSIZE*sizeof(double));
}

//modified by zwl
void sw_slave_memcpy_f(MemcpyPara *para) {
  int id = athread_get_id(-1);
  id = (id%16)*4+(id/16);
  volatile unsigned long replyget=0;
  volatile unsigned long replyput=0 ; 
  const int BUFFSIZE = 256;
  const int SPNUM = 64;
  float * local_src = (float *)ldm_malloc(BUFFSIZE*sizeof(float));
  int count = para->count;

  int local_offset = id*BUFFSIZE;

  while(local_offset < count){
    float* src_ptr = (float*)para->src+local_offset;
    float* dst_ptr = (float*)para->dst+local_offset;
    athread_get(PE_MODE, (void*)src_ptr, (void*)local_src, BUFFSIZE*sizeof(float), (void*)&replyget, 0, 0, 0);
    while(replyget!=1);
    replyget = 0;
    athread_put(PE_MODE, (void*)local_src, (void*)dst_ptr, BUFFSIZE*sizeof(float), (void*)&replyput, 0, 0);
    while(replyput!=1);
    replyput = 0;
    local_offset += SPNUM*BUFFSIZE;
  }
  ldm_free(local_src, BUFFSIZE*sizeof(float));
}
/*void sw_slave_memcpy_f(MemcpyPara *para) {*/
  /*const int BUFFSIZE = 4*1024;*/
  /*const int SPNUM = 64;*/
  /*float * local_src = (float *)ldm_malloc(BUFFSIZE*sizeof(float ));*/
  /*int id = athread_get_id(-1);*/
  /*int count = para->count;*/
  /*int local_count = count/SPNUM + (id<(count%SPNUM));*/
  /*int start = id*(count/SPNUM) + (id<(count%SPNUM)?id:(count%SPNUM));*/
  /*float * src_ptr = &(((float *)para->src)[start]);*/
  /*float * dst_ptr = &(((float *)para->dst)[start]);*/
  /*volatile int replyget=0, replyput=0;*/
  /*int i,off;*/
  /*// DMA settings*/
  /*//dma_desc dma_get_memcpy_src, dma_put_memcpy_dst;*/
  /*dma_set_op(&dma_get_memcpy_src, DMA_GET);*/
  /*dma_set_mode(&dma_get_memcpy_src, PE_MODE);*/
  /*dma_set_reply(&dma_get_memcpy_src, &replyget);*/

  /*dma_set_op(&dma_put_memcpy_dst, DMA_PUT);*/
  /*dma_set_mode(&dma_put_memcpy_dst, PE_MODE);*/
  /*dma_set_reply(&dma_put_memcpy_dst, &replyput);*/

  /*dma_set_size(&dma_get_memcpy_src,BUFFSIZE*sizeof(float));*/
  /*dma_set_size(&dma_put_memcpy_dst,BUFFSIZE*sizeof(float));*/

  /*for(off = 0; off+BUFFSIZE-1<local_count; off+=BUFFSIZE)*/
  /*{*/
    /*// DMA get a block*/
    /*dma(dma_get_memcpy_src, (long)(src_ptr+off), (long)(local_src));*/
    /*dma_wait(&replyget, 1); replyget = 0;*/
    /*//dma_barrier();*/
    /*mb();*/
    /*// DMA put result*/
    /*dma(dma_put_memcpy_dst, (long)(dst_ptr+off), (long)(local_src));*/
    /*dma_wait(&replyput, 1); replyput = 0;*/
  /*}*/

  /*if(off<local_count) {*/
    /*dma_set_size(&dma_get_memcpy_src,(local_count-off)*sizeof(float));*/
    /*dma_set_size(&dma_put_memcpy_dst,(local_count-off)*sizeof(float));*/
    /*dma(dma_get_memcpy_src, (long)(src_ptr+off), (long)(local_src));*/
    /*dma_wait(&replyget, 1); replyget = 0;*/
    /*//dma_barrier();*/
    /*mb();*/
    /*// DMA put result*/
    /*dma(dma_put_memcpy_dst, (long)(dst_ptr+off), (long)(local_src));*/
    /*dma_wait(&replyput, 1); replyput = 0;*/
  /*}*/

  /*ldm_free(local_src, BUFFSIZE*sizeof(float));*/
/*}*/
 
void sw_slave_memcpy_i(MemcpyPara *para) {
  const int BUFFSIZE = 4*1024;
  const int SPNUM = 64;
  int * local_src = (int *)ldm_malloc(BUFFSIZE*sizeof(int));
  int id = athread_get_id(-1);
  int count = para->count;
  int local_count = count/SPNUM + (id<(count%SPNUM));
  int start = id*(count/SPNUM) + (id<(count%SPNUM)?id:(count%SPNUM));
  int * src_ptr = (int *)(para->src+start*sizeof(int));
  int * dst_ptr = (int *)(para->dst+start*sizeof(int));
  volatile int replyget=0, replyput=0;
  int i,off;
  // DMA settings
  //dma_desc dma_get_memcpy_src, dma_put_memcpy_dst;
  dma_set_op(&dma_get_memcpy_src, DMA_GET);
  dma_set_mode(&dma_get_memcpy_src, PE_MODE);
  dma_set_reply(&dma_get_memcpy_src, &replyget);

  dma_set_op(&dma_put_memcpy_dst, DMA_PUT);
  dma_set_mode(&dma_put_memcpy_dst, PE_MODE);
  dma_set_reply(&dma_put_memcpy_dst, &replyput);

  dma_set_size(&dma_get_memcpy_src,BUFFSIZE*sizeof(int));
  dma_set_size(&dma_put_memcpy_dst,BUFFSIZE*sizeof(int));

  for(off = 0; off+BUFFSIZE-1<local_count; off+=BUFFSIZE)
  {
    // DMA get a block
    dma(dma_get_memcpy_src, (long)(src_ptr+off), (long)(local_src));
    dma_wait(&replyget, 1); replyget = 0;
    //dma_barrier();
    mb();
    // DMA put result
    dma(dma_put_memcpy_dst, (long)(dst_ptr+off), (long)(local_src));
    dma_wait(&replyput, 1); replyput = 0;
  }

  if(off<local_count) {
    dma_set_size(&dma_get_memcpy_src,(local_count-off)*sizeof(int));
    dma_set_size(&dma_put_memcpy_dst,(local_count-off)*sizeof(int));
    dma(dma_get_memcpy_src, (long)(src_ptr+off), (long)(local_src));
    dma_wait(&replyget, 1); replyget = 0;
    //dma_barrier();
    mb();
    // DMA put result
    dma(dma_put_memcpy_dst, (long)(dst_ptr+off), (long)(local_src));
    dma_wait(&replyput, 1); replyput = 0;
  }

  ldm_free(local_src, BUFFSIZE*sizeof(int));
}
void sw_slave_memcpy_ui(MemcpyPara *para) {
  const int BUFFSIZE = 4*1024;
  const int SPNUM = 64;
  unsigned int * local_src = (unsigned int *)ldm_malloc(BUFFSIZE*sizeof(unsigned int));
  int id = athread_get_id(-1);
  int count = para->count;
  int local_count = count/SPNUM + (id<(count%SPNUM));
  int start = id*(count/SPNUM) + (id<(count%SPNUM)?id:(count%SPNUM));
  unsigned int * src_ptr = (unsigned int *)(para->src+start*sizeof(unsigned int));
  unsigned int * dst_ptr = (unsigned int *)(para->dst+start*sizeof(unsigned int));
  volatile int replyget=0, replyput=0;
  int i,off;
  // DMA settings
  //dma_desc dma_get_memcpy_src, dma_put_memcpy_dst;
  dma_set_op(&dma_get_memcpy_src, DMA_GET);
  dma_set_mode(&dma_get_memcpy_src, PE_MODE);
  dma_set_reply(&dma_get_memcpy_src, &replyget);

  dma_set_op(&dma_put_memcpy_dst, DMA_PUT);
  dma_set_mode(&dma_put_memcpy_dst, PE_MODE);
  dma_set_reply(&dma_put_memcpy_dst, &replyput);

  dma_set_size(&dma_get_memcpy_src,BUFFSIZE*sizeof(unsigned int));
  dma_set_size(&dma_put_memcpy_dst,BUFFSIZE*sizeof(unsigned int));

  for(off = 0; off+BUFFSIZE-1<local_count; off+=BUFFSIZE)
  {
    // DMA get a block
    dma(dma_get_memcpy_src, (long)(src_ptr+off), (long)(local_src));
    dma_wait(&replyget, 1); replyget = 0;
    //dma_barrier();
    mb();
    // DMA put result
    dma(dma_put_memcpy_dst, (long)(dst_ptr+off), (long)(local_src));
    dma_wait(&replyput, 1); replyput = 0;
  }

  if(off<local_count) {
    dma_set_size(&dma_get_memcpy_src,(local_count-off)*sizeof(unsigned int));
    dma_set_size(&dma_put_memcpy_dst,(local_count-off)*sizeof(unsigned int));
    dma(dma_get_memcpy_src, (long)(src_ptr+off), (long)(local_src));
    dma_wait(&replyget, 1); replyget = 0;
    //dma_barrier();
    mb();
    // DMA put result
    dma(dma_put_memcpy_dst, (long)(dst_ptr+off), (long)(local_src));
    dma_wait(&replyput, 1); replyput = 0;
  }

  ldm_free(local_src, BUFFSIZE*sizeof(unsigned int));
}
void sw_slave_weights_memcpy_f(MemcpyPara *para) {
  const int BUFFSIZE = 4*1024;
  const int SPNUM = 64;
  float * local_src = (float *)ldm_malloc(BUFFSIZE*sizeof(float ));
  int id = athread_get_id(-1);
  int count = para->count;
  int num = para->num;
  int local_count = count/SPNUM + (id<(count%SPNUM));
  int start = id*(count/SPNUM) + (id<(count%SPNUM)?id:(count%SPNUM));
  float * src_ptr = &(((float *)para->src)[start]);
  float * dst_ptr = &(((float *)para->dst)[start]);
  volatile int replyget=0, replyput=0;
  int i,off;
  // DMA settings
  //dma_desc dma_get_memcpy_src, dma_put_memcpy_dst;
  dma_set_op(&dma_get_memcpy_src, DMA_GET);
  dma_set_mode(&dma_get_memcpy_src, PE_MODE);
  dma_set_reply(&dma_get_memcpy_src, &replyget);

  dma_set_op(&dma_put_memcpy_dst, DMA_PUT);
  dma_set_mode(&dma_put_memcpy_dst, PE_MODE);
  dma_set_reply(&dma_put_memcpy_dst, &replyput);

  dma_set_size(&dma_get_memcpy_src,BUFFSIZE*sizeof(float));
  dma_set_size(&dma_put_memcpy_dst,BUFFSIZE*sizeof(float));

  for(off = 0; off+BUFFSIZE-1<local_count; off+=BUFFSIZE)
  {
    // DMA get a block
    dma(dma_get_memcpy_src, (long)(src_ptr+off), (long)(local_src));
    dma_wait(&replyget, 1); replyget = 0;
    //dma_barrier();
    mb();
    // DMA put result
    for(i =0;i< num;i++)
    {
      dma(dma_put_memcpy_dst, (long)(dst_ptr+off+i*count), (long)(local_src));
    }
    dma_wait(&replyput, num); replyput = 0;
    mb();
  }

  if(off<local_count) {
    dma_set_size(&dma_get_memcpy_src,(local_count-off)*sizeof(float));
    dma_set_size(&dma_put_memcpy_dst,(local_count-off)*sizeof(float));
    dma(dma_get_memcpy_src, (long)(src_ptr+off), (long)(local_src));
    dma_wait(&replyget, 1); replyget = 0;
    //dma_barrier();
    mb();
    // DMA put result
    for(i =0; i < num;i++)
    {
      dma(dma_put_memcpy_dst, (long)(dst_ptr+off +i *count), (long)(local_src));
    }
    dma_wait(&replyput, num); replyput = 0;
    mb();
  }

  ldm_free(local_src, BUFFSIZE*sizeof(float));
}
void sw_slave_weights_memcpy_d(MemcpyPara *para) {
  const int BUFFSIZE = 4*1024;
  const int SPNUM = 64;
  double * local_src = (double *)ldm_malloc(BUFFSIZE*sizeof(double ));
  int id = athread_get_id(-1);
  int count = para->count;
  int num = para->num;
  int local_count = count/SPNUM + (id<(count%SPNUM));
  int start = id*(count/SPNUM) + (id<(count%SPNUM)?id:(count%SPNUM));
  double * src_ptr = &(((double *)para->src)[start]);
  double * dst_ptr = &(((double *)para->dst)[start]);
  volatile int replyget=0, replyput=0;
  int i,off;
  // DMA settings
  //dma_desc dma_get_memcpy_src, dma_put_memcpy_dst;
  dma_set_op(&dma_get_memcpy_src, DMA_GET);
  dma_set_mode(&dma_get_memcpy_src, PE_MODE);
  dma_set_reply(&dma_get_memcpy_src, &replyget);

  dma_set_op(&dma_put_memcpy_dst, DMA_PUT);
  dma_set_mode(&dma_put_memcpy_dst, PE_MODE);
  dma_set_reply(&dma_put_memcpy_dst, &replyput);

  dma_set_size(&dma_get_memcpy_src,BUFFSIZE*sizeof(double));
  dma_set_size(&dma_put_memcpy_dst,BUFFSIZE*sizeof(double));

  for(off = 0; off+BUFFSIZE-1<local_count; off+=BUFFSIZE)
  {
    // DMA get a block
    dma(dma_get_memcpy_src, (long)(src_ptr+off), (long)(local_src));
    dma_wait(&replyget, 1); replyget = 0;
    //dma_barrier();
    mb();
    // DMA put result
    for(i =0;i< num;i++)
    {
      dma(dma_put_memcpy_dst, (long)(dst_ptr+off+i*count), (long)(local_src));
    }
    dma_wait(&replyput, num); replyput = 0;
    mb();
  }

  if(off<local_count) {
    dma_set_size(&dma_get_memcpy_src,(local_count-off)*sizeof(double));
    dma_set_size(&dma_put_memcpy_dst,(local_count-off)*sizeof(double));
    dma(dma_get_memcpy_src, (long)(src_ptr+off), (long)(local_src));
    dma_wait(&replyget, 1); replyget = 0;
    //dma_barrier();
    mb();
    // DMA put result
    for(i =0; i < num;i++)
    {
      dma(dma_put_memcpy_dst, (long)(dst_ptr+off +i *count), (long)(local_src));
    }
    dma_wait(&replyput, num); replyput = 0;
    mb();
  }

  ldm_free(local_src, BUFFSIZE*sizeof(double));
}
