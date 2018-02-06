/******************************************
 * Created by Liandeng Li
 * Date: 2017/10/5
 * ***************************************/

#ifndef SW_DNN_H_
#define SW_DNN_H_
// Precondition: already athread_init()
void sw_mul_d(const double* src1,const double* src2, double* dst, const int count);
void sw_mul_f(const float * src1, const float *src2,float * dst,const  int count);
void sw_div_d(const double* src1,const double *src2, double* dst, const int count) ;
void sw_div_f(const float* src1,const float *src2, float* dst, const int count) ;
void sw_sqr_d(const double* src,double* dst,const  int count) ;
void sw_sqr_f(const float* src,float* dst,const  int count) ;
void sw_sqrt_d(const double* src,double* dst,const  int count) ;
void sw_sqrt_f(const float* src,float* dst,const  int count) ;
void sw_scale_layer_d(const double* src,const double *scale, double* dst, const int outer_dim,const int inner_dim,const int scale_dim) ;
void sw_scale_layer_f(const float* src,const float *scale, float* dst, const int outer_dim,const int inner_dim,const int scale_dim) ;
void sw_memcpy_d(const double* src, double* dst,const int count) ;
void sw_memcpy_f(const float* src, float* dst,const int count) ;
void sw_memcpy_i(const int* src, int* dst,const int count) ;
void sw_memcpy_ui(const unsigned int* src,unsigned int* dst,const int count) ;
void sw_repeat_memcpy_d(const double* src, double* dst,const int count,const int num) ;
void sw_repeat_memcpy_f(const float* src, float* dst,const int count,const int num) ;
void sw_weights_add_f(float *dst,const float *src,int num,int count);
void sw_weights_add_d(double *dst,const double *src,int num,int count);
void sw_sub_d(const double* src1,const double *src2, double* dst, const int count) ;
void sw_sub_f(const float* src1,const float *src2, float* dst, const int count) ;
void sw_add_f(const float* src1,const float *src2, float* dst, const int count) ;
void sw_add_d(const double* src1,const double *src2, double* dst, const int count) ;
//void sw_net_param_sub_f(const int count,const int numprocs,float*data,float*diff);
//void sw_net_param_sub_d(const int count,const int numprocs,double*data,double*diff);
//void sw_bias_f(const int count,const int M,const int N,float *A,float*B,float*C);
//void sw_bias_d(const int count,const int M,const int N,double *A,double*B,double*C);
//void sw_trans_f(const int num,const int row,const int col,const float * src,float * dst);
//void sw_trans_d(const int num,const int row,const int col,const double * src,double * dst);
//void sw_gemm_d(const int num,const int M, const int N, const int K, const double alpha, const double* A, const double* B, const double beta, double* C) ;
//void sw_gemm_f(const int num,const int M, const int N, const int K, const float alpha, const float* A, const float* B, const float beta, float* C) ;
/*void sw_memset_d(double* src,const double val,const int count); 
void sw_memset_f(float* src,const float val,const int count) ;
void sw_memset_i(int* src,const int val,const int count) ;
void sw_dropout_layer_d(const double* src,const unsigned int* mask,double *dst,const double scale ,const int count) ;
void sw_dropout_layer_f(const float* src,const unsigned int* mask,float *dst,const float scale ,const int count) ;
void sw_weights_memcpy_d(const double* src, double* dst,const int count,const int num) ;
void sw_weights_memcpy_f(const float* src, float* dst,const int count,const int num) ;
void sw_add_d(const double* src1,const double *src2, double* dst, const int count) ;
void sw_add_f(const float* src1,const float *src2, float* dst, const int count) ;
void sw_sub_d(const double* src1,const double *src2, double* dst, const int count) ;
void sw_sub_f(const float* src1,const float *src2, float* dst, const int count) ;
void sw_abs_d(const double* src, double* dst,const  int count) ;
void sw_abs_f(const float* src, float* dst, const int count) ;
void sw_axpby_d(const double alpha,const double *src,const double beta, double* dst,const  int count) ;
void sw_axpby_f(const float alpha,const float *src,const float beta,float* dst,const  int count) ;
void sw_axpy_d(const double alpha,const double *src, double* dst,const  int count) ;
void sw_axpy_f(const float alpha,const float *src, float* dst, const int count) ;
void sw_pow_d(const double* src,const double val, double* dst,const int count) ;
void sw_pow_f(const float* src,const float val,float* dst,const int count) ;
void sw_log_d(const double* src, double* dst, const int count) ;
void sw_log_f(const float* src, float* dst,const int count) ;
void sw_exp_d(const double* src, double* dst,const int count) ;
void sw_exp_f(const float* src, float* dst,const int count) ;
void sw_add_scalar_d(const double alpha,double *src,const int count) ;
void sw_add_scalar_f(const float alpha,float *src,const int count) ;
void sw_sscal_d(const double*src,const double alpha,double *dst,const int count) ;
void sw_sscal_f(const float*src,const float alpha,float *dst,const int count) ;
*/
#endif
