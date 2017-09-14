#ifndef MATRIX_TRANS_H_
#define MATRIX_TRANS_H_

//#include <simd.h>

#define HWSIZE       48
#define BSIZE        64
#define NUM_THREADS 64
#define BUFFS       3072 //(24*1024) 

typedef struct _tagSlaveParam
{
	int B,N,H,W,splitNB,splitHW;
	int nCount,nBNThreadsNum,nBNLeftThreadsNum,nNBHWThreadsNum,nNBHWLeftThreadsNum;
	double *pIn,*pOut;
}SlaveParam;

typedef struct _tagSlaveParam_f
{
  int B,N,H,W,splitNB,splitHW;
  int nCount,nBNThreadsNum,nBNLeftThreadsNum,nNBHWThreadsNum,nNBHWLeftThreadsNum;
  float *pIn,*pOut;
}SlaveParam_f;

void weight_caffe_to_swdnn_back_d(double*in,double*out,int B,int N,int H,int W);
void image_caffe_to_swdnn_d(double* in,double* out,int B,int N,int H,int W);
void image_swdnn_to_caffe_d(double*in,double*out,int B,int N,int H,int W);
void weight_swdnn_to_caffe_d(double* in, double* out, int B, int N, int H, int W);
void image_caffe_to_swdnn_back_d(double* in, double* out,int B, int N, int H, int W);
void weight_caffe_to_swdnn_d(double* in, double* out, int B, int N, int H, int W);
void weight_caffe_to_swdnn_back_f(float*in,float *out,int B,int N,int H,int W);
void image_caffe_to_swdnn_f(float*in,float*out,int B,int N,int H,int W);
void image_swdnn_to_caffe_f(float*in,float*out,int B,int N,int H,int W);
void weight_swdnn_to_caffe_f(float* in, float* out, int B, int N, int H, int W);
void image_caffe_to_swdnn_back_f(float* in, float* out,int B, int N, int H, int W);
void weight_caffe_to_swdnn_f(float* in, float* out, int B, int N, int H, int W);

#endif
