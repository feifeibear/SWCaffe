#include <stdio.h>
#include <stdlib.h>
#include <athread.h>
#include <simd.h>
#include "caffe/swlayers/sw_pool_layer_impl.h"

extern SLAVE_FUN(poolingForwardMax)();
extern SLAVE_FUN(poolingForwardAvg)();
extern SLAVE_FUN(poolingBackwardMax)();
extern SLAVE_FUN(poolingBackwardAvg)();

int pooling_judge_condition(int N,int C,int pooled_height_,int pooled_width_)
{
	const int nMinBuffSize = 16;
	if((N*C) < nMinBuffSize || (pooled_height_*pooled_width_) < nMinBuffSize) return -1;
	return 1;
}

void pooling_forward_max(int N,int C,double *pTopData,const double *pBottomData,int*pMask,double*pTopMask,int nBottomOffset,
				int nTopOffset,int use_top_mask,int pooled_height_,int pooled_width_,int stride_h_,
				int stride_w_,int pad_h_,int pad_w_,int kernel_h_,int kernel_w_,int height_,int width_)
{
	SlavePoolingParam param;
	param.pooled_height_ = pooled_height_;
	param.pooled_width_ = pooled_width_;
	param.stride_h_ = stride_h_;
	param.stride_w_ = stride_w_;
	param.pad_h_ = pad_h_;
	param.pad_w_ = pad_w_;	
	param.kernel_h_ = kernel_h_;	
	param.kernel_w_ = kernel_w_;	
	param.height_ = height_;	
	param.width_ = width_;	
	param.nBottomOffset = nBottomOffset;	
	param.nTopOffset = nTopOffset;	
	param.use_top_mask = use_top_mask;	
	param.pMask = pMask;	
	param.pTopMask = pTopMask;	
	param.pTopData = pTopData;	
	param.pBottomData = pBottomData;	
	
	int nNB = N*C;
	param.nCount = nNB/NUM_THREADS;
	param.nThreadsNum = param.nCount >0 ? NUM_THREADS:nNB;
	param.nLeftThreadsNum = param.nCount >0 ? nNB%NUM_THREADS:nNB;
	
	athread_spawn(poolingForwardMax,(void *)&param);
	athread_join();
}

void pooling_forward_avg(int N,int C,double *pTopData,const double *pBottomData,int nBottomOffset,
				int nTopOffset,int pooled_height_,int pooled_width_,int stride_h_,
				int stride_w_,int pad_h_,int pad_w_,int kernel_h_,int kernel_w_,int height_,int width_)
{
	SlavePoolingParam param;
	param.pooled_height_ = pooled_height_;
	param.pooled_width_ = pooled_width_;
	param.stride_h_ = stride_h_;
	param.stride_w_ = stride_w_;
	param.pad_h_ = pad_h_;
	param.pad_w_ = pad_w_;	
	param.kernel_h_ = kernel_h_;	
	param.kernel_w_ = kernel_w_;	
	param.height_ = height_;	
	param.width_ = width_;	
	param.nBottomOffset = nBottomOffset;	
	param.nTopOffset = nTopOffset;	
	param.pTopData = pTopData;	
	param.pBottomData = pBottomData;	
	
	int nNB = N*C;
	param.nCount = nNB/NUM_THREADS;
	param.nThreadsNum = param.nCount >0 ? NUM_THREADS:nNB;
	param.nLeftThreadsNum = param.nCount >0 ? nNB%NUM_THREADS:nNB;
	
	athread_spawn(poolingForwardAvg,(void *)&param);
	athread_join();
}

void pooling_backward_max(int N,int C,const double *pTopData,double *pBottomData,const int*pMask,const double*pTopMask,int nBottomOffset,
				int nTopOffset,int use_top_mask,int pooled_height_,int pooled_width_,int stride_h_,
				int stride_w_,int pad_h_,int pad_w_,int kernel_h_,int kernel_w_,int height_,int width_)
{
	SlavePoolingParam param;
	param.pooled_height_ = pooled_height_;
	param.pooled_width_ = pooled_width_;
	param.stride_h_ = stride_h_;
	param.stride_w_ = stride_w_;
	param.pad_h_ = pad_h_;
	param.pad_w_ = pad_w_;	
	param.kernel_h_ = kernel_h_;	
	param.kernel_w_ = kernel_w_;	
	param.height_ = height_;	
	param.width_ = width_;	
	param.nBottomOffset = nBottomOffset;	
	param.nTopOffset = nTopOffset;	
	param.use_top_mask = use_top_mask;	
	param.pMask = pMask;	
	param.pTopMask = pTopMask;	
	param.pTopData = pTopData;	
	param.pBottomData = pBottomData;	
	
	int nNB = N*C;
	param.nCount = nNB/NUM_THREADS;
	param.nThreadsNum = param.nCount >0 ? NUM_THREADS:nNB;
	param.nLeftThreadsNum = param.nCount >0 ? nNB%NUM_THREADS:nNB;
	
	athread_spawn(poolingBackwardMax,(void *)&param);
	athread_join();
}

void pooling_backward_avg(int N,int C,const double *pTopData,double *pBottomData,int nBottomOffset,
				int nTopOffset,int pooled_height_,int pooled_width_,int stride_h_,
				int stride_w_,int pad_h_,int pad_w_,int kernel_h_,int kernel_w_,int height_,int width_)
{
	SlavePoolingParam param;
	param.pooled_height_ = pooled_height_;
	param.pooled_width_ = pooled_width_;
	param.stride_h_ = stride_h_;
	param.stride_w_ = stride_w_;
	param.pad_h_ = pad_h_;
	param.pad_w_ = pad_w_;	
	param.kernel_h_ = kernel_h_;	
	param.kernel_w_ = kernel_w_;	
	param.height_ = height_;	
	param.width_ = width_;	
	param.nBottomOffset = nBottomOffset;	
	param.nTopOffset = nTopOffset;	
	param.pTopData = pTopData;	
	param.pBottomData = pBottomData;	
	
	int nNB = N*C;
	param.nCount = nNB/NUM_THREADS;
	param.nThreadsNum = param.nCount >0 ? NUM_THREADS:nNB;
	param.nLeftThreadsNum = param.nCount >0 ? nNB%NUM_THREADS:nNB;
	
	athread_spawn(poolingBackwardAvg,(void *)&param);
	athread_join();
}
