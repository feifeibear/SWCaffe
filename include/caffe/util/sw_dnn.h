/******************************************
 * Created by Liandeng Li
 * Date: 2017/10/5
 * ***************************************/

#ifndef SW_DNN_H_
#define SW_DNN_H_
// Precondition: already athread_init()
void sw_memcpy_f(const float* src, float* dst,const int count) ;
//void sw_memcpy_d(const double* src, double* dst,const int count) ;
//void sw_add_d(const double* src1,const double *src2, double* dst, const int count) ;
void sw_add_f(const float* src1,const float *src2, float* dst, const int count) ;
#endif
