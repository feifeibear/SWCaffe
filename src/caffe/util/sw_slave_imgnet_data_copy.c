#include <slave.h>
#include <simd.h>
#include <dma.h>

typedef struct DataParam_st {
  unsigned char* data;
  void* top0;
  void* mean;
  int n_cols;
  int n_rows;
  int batch_size;
}DataParam;

__thread_local dma_desc dma_get_data, dma_get_mean, dma_put_top0;

void sw_slave_imgnet_f(DataParam *para) {
#define Type float
  int id = athread_get_id(-1);
  int n_cols = para->n_cols;
  int n_rows = para->n_rows;
  int batch_size = para->batch_size;

  int begin_pos = id*(3*n_cols/64) + (id<((3*n_cols)%64)?id:((3*n_cols)%64));
  unsigned char * data_ptr = para->data + begin_pos*n_rows;
  Type* top0_ptr = (Type*)para->top0 + begin_pos*n_rows;
  Type* mean_ptr = (Type*)para->mean + begin_pos*n_rows;

  int img_size = 3*n_rows*n_cols;
  int mySize = 3*n_cols/64 + (id<((3*n_cols)%64));
  int local_data_size = n_rows*mySize;
  unsigned char* local_data = (unsigned char*)ldm_malloc(local_data_size*sizeof(char));
  Type* local_mean = (Type*)ldm_malloc(local_data_size*sizeof(Type));
  Type* local_top0 = (Type*)ldm_malloc(local_data_size*sizeof(Type));

  int img,i;
  volatile int data_replyget=0, mean_replyget=0, replyput=0;

  dma_set_op(&dma_get_data, DMA_GET);
  dma_set_mode(&dma_get_data, PE_MODE);
  dma_set_reply(&dma_get_data, &data_replyget);

  dma_set_op(&dma_get_mean, DMA_GET);
  dma_set_mode(&dma_get_mean, PE_MODE);
  dma_set_reply(&dma_get_mean, &mean_replyget);

  dma_set_op(&dma_put_top0, DMA_PUT);
  dma_set_mode(&dma_put_top0, PE_MODE);
  dma_set_reply(&dma_put_top0, &replyput);

  dma_set_size(&dma_get_data,local_data_size*sizeof(char));
  dma_set_size(&dma_get_mean,local_data_size*sizeof(Type));
  dma_set_size(&dma_put_top0,local_data_size*sizeof(Type));

  dma(dma_get_mean,(long)(mean_ptr),(long)(local_mean));
  dma_wait(&mean_replyget, 1); mean_replyget=0;

  for(img=0;img<batch_size;++img) {
    dma(dma_get_data,(long)(data_ptr),(long)local_data);
    dma_wait(&data_replyget, 1); data_replyget=0;
    data_ptr+=img_size;

    for(i=0;i<local_data_size;++i) {
      local_top0[i] = (Type)local_data[i] - local_mean[i];
    }

    dma(dma_put_top0,(long)top0_ptr,(long)local_top0);
    dma_wait(&replyput, 1); replyput = 0;
    top0_ptr+=img_size;
  }

  ldm_free(local_data,local_data_size*sizeof(char));
  ldm_free(local_mean,local_data_size*sizeof(Type));
  ldm_free(local_top0,local_data_size*sizeof(Type));
#undef Type
}

void sw_slave_imgnet_d(DataParam *para) {
#define Type double
  int id = athread_get_id(-1);
  int n_cols = para->n_cols;
  int n_rows = para->n_rows;
  int batch_size = para->batch_size;

  int begin_pos = id*(3*n_cols/64) + (id<((3*n_cols)%64)?id:((3*n_cols)%64));
  unsigned char * data_ptr = para->data + begin_pos*n_rows;
  Type* top0_ptr = (Type*)para->top0 + begin_pos*n_rows;
  Type* mean_ptr = (Type*)para->mean + begin_pos*n_rows;

  int img_size = 3*n_rows*n_cols;
  int mySize = 3*n_cols/64 + (id<((3*n_cols)%64));
  int local_data_size = n_rows*mySize;
  unsigned char* local_data = (unsigned char*)ldm_malloc(local_data_size*sizeof(char));
  Type* local_mean = (Type*)ldm_malloc(local_data_size*sizeof(Type));
  Type* local_top0 = (Type*)ldm_malloc(local_data_size*sizeof(Type));

  int img,i;
  volatile int data_replyget=0, mean_replyget=0, replyput=0;

  dma_set_op(&dma_get_data, DMA_GET);
  dma_set_mode(&dma_get_data, PE_MODE);
  dma_set_reply(&dma_get_data, &data_replyget);

  dma_set_op(&dma_get_mean, DMA_GET);
  dma_set_mode(&dma_get_mean, PE_MODE);
  dma_set_reply(&dma_get_mean, &mean_replyget);

  dma_set_op(&dma_put_top0, DMA_PUT);
  dma_set_mode(&dma_put_top0, PE_MODE);
  dma_set_reply(&dma_put_top0, &replyput);

  dma_set_size(&dma_get_data,local_data_size*sizeof(char));
  dma_set_size(&dma_get_mean,local_data_size*sizeof(Type));
  dma_set_size(&dma_put_top0,local_data_size*sizeof(Type));

  dma(dma_get_mean,(long)(mean_ptr),(long)(local_mean));
  dma_wait(&mean_replyget, 1); mean_replyget=0;

  for(img=0;img<batch_size;++img) {
    dma(dma_get_data,(long)(data_ptr),(long)local_data);
    dma_wait(&data_replyget, 1); data_replyget=0;
    data_ptr+=img_size;

    for(i=0;i<local_data_size;++i) {
      local_top0[i] = (Type)local_data[i] - local_mean[i];
    }

    dma(dma_put_top0,(long)top0_ptr,(long)local_top0);
    dma_wait(&replyput, 1); replyput = 0;
    top0_ptr+=img_size;
  }

  ldm_free(local_data,local_data_size*sizeof(char));
  ldm_free(local_mean,local_data_size*sizeof(Type));
  ldm_free(local_top0,local_data_size*sizeof(Type));
#undef Type
}
