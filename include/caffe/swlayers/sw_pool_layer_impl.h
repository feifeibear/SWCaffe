#ifndef SW_POOL_LAYER_IMPL_H_
#define SW_POOL_LAYER_IMPL_H_

#define NUM_THREADS 64

typedef double Type;
typedef struct _tagSlavePoolingParam
{
	int pooled_height_,pooled_width_,stride_h_,stride_w_,pad_h_,pad_w_,kernel_h_,kernel_w_,height_,width_;
	int nCount,nThreadsNum,nLeftThreadsNum;
	int nBottomOffset,nTopOffset,use_top_mask;
	int  *pMask;
	Type *pTopData,*pBottomData,*pTopMask;
}SlavePoolingParam;

 int pooling_judge_condition(int N,int C,int pooled_height_,int pooled_width_);

 void pooling_forward_max(int N,int C,double *pTopData,const double *pBottomData,int*pMask,double *pTopMask,int nBottomOffset,
				int nTopOffset,int use_top_mask,int pooled_height_,int pooled_width_,int stride_h_,
				int stride_w_,int pad_h_,int pad_w_,int kernel_h_,int kernel_w_,int height_,int width_);
 void pooling_forward_avg(int N,int C,double *pTopData,const double*pBottomData,int nBottomOffset,
				int nTopOffset,int pooled_height_,int pooled_width_,int stride_h_,
				int stride_w_,int pad_h_,int pad_w_,int kernel_h_,int kernel_w_,int height_,int width_);				
 void pooling_backward_max(int N,int C,const double *pTopData,double *pBottomData,const int*pMask,const double *pTopMask,int nBottomOffset,
				int nTopOffset,int use_top_mask,int pooled_height_,int pooled_width_,int stride_h_,
				int stride_w_,int pad_h_,int pad_w_,int kernel_h_,int kernel_w_,int height_,int width_);
 void pooling_backward_avg(int N,int C,const double *pTopData,double*pBottomData,int nBottomOffset,
				int nTopOffset,int pooled_height_,int pooled_width_,int stride_h_,
				int stride_w_,int pad_h_,int pad_w_,int kernel_h_,int kernel_w_,int height_,int width_);				

#endif
