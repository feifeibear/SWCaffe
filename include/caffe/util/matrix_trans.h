#ifndef MATRIX_TRANS_H_
#define MATRIX_TRANS_H_

//#include <simd.h>

#define HWSIZE       48
#define BSIZE        64
#define NUM_THREADS 64
#define BUFFS       3072 //(24*1024) 

#define Type double

typedef struct _tagSlaveParam
{
	int B,N,H,W,splitNB,splitHW;
	int nCount,nBNThreadsNum,nBNLeftThreadsNum,nNBHWThreadsNum,nNBHWLeftThreadsNum;
	double *pIn,*pOut;
}SlaveParam;


void weight_caffe_to_swdnn_back(double*in,double*out,int B,int N,int H,int W);
void image_caffe_to_swdnn(double* in,double* out,int B,int N,int H,int W);
void image_swdnn_to_caffe(double*in,double*out,int B,int N,int H,int W);
void weight_swdnn_to_caffe(double* in, double* out, int B, int N, int H, int W);
void image_caffe_to_swdnn_back(double* in, double* out,int B, int N, int H, int W);

//void weight_caffe_to_swdnn_back(float*in,float *out,int B,int N,int H,int W) {}
//void image_caffe_to_swdnn(float*in,float*out,int B,int N,int H,int W){}
//void image_swdnn_to_caffe(float*in,float*out,int B,int N,int H,int W){}
//void weight_swdnn_to_caffe(float* in, float* out, int B, int N, int H, int W){}
//void image_caffe_to_swdnn_back(float* in, float* out,int B, int N, int H, int W) {}






#endif
