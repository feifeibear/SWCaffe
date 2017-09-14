#ifndef SW_IM2COL_IMPL_H_
#define SW_IM2COL_IMPL_H_

typedef struct _tagSlaveIm2ColParam_f
{
	int channels,height,width,kernel_h,kernel_w,pad_h, pad_w,stride_h,stride_w,dilation_h,dilation_w;
	int nCount,nThreadsNum,nLeftThreadsNum,nMsgSize,nTimes;
	int nChannelsCount,nColsWidth;
	float *pIn,*pOut;
}SlaveIm2ColParam_f;

typedef struct _tagSlaveIm2ColParam
{
	int channels,height,width,kernel_h,kernel_w,pad_h, pad_w,stride_h,stride_w,dilation_h,dilation_w;
	int nCount,nThreadsNum,nLeftThreadsNum,nMsgSize,nTimes;
	int nChannelsCount,nColsWidth;
	double *pIn,*pOut;
}SlaveIm2ColParam;


void sw_im2col_impl_f(const float* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    float* data_col) ;
/*	
void sw_im2col_impl_d(const double* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    double* data_col) ;

void sw_col2im_impl_f(const float* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    float* data_im) ;
	
void sw_col2im_impl_d(const float* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    float* data_im) ;
*/	
#endif
