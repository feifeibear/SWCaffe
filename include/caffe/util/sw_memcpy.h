/******************************************
 * Created by Xin You
 * Date: 2017/8/7
 * Memory copy functions: (called by host)
 * 1. double: sw_memcpy_d(double* src, double* dst, int count)
 * 2. float : sw_memcpy_f(float * src, float * dst, int count)
 * ***************************************/

#ifndef SW_MEMCPY_H_
#define SW_MEMCPY_H_
// Precondition: already athread_init()
void sw_memcpy_d(double* src, double* dst, int count);
void sw_memcpy_f(float * src, float * dst, int count);

#endif
