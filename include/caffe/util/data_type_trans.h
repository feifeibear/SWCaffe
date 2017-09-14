/******************************************
 * Created by Xin You
 * Date: 2017/8/4
 * Data type transfer functions: (called by host)
 * 1. double to float: double2float(double* src, float* dst, int count)
 * 2. float to double: float2double(float* src, double* dst, int count)
 * ***************************************/

#ifndef DATA_TYPE_TRANS_H_
#define DATA_TYPE_TRANS_H_
// Precondition: already athread_init()
void double2float(double* src, float* dst, int count);
void float2double(float* src, double* dst, int count);

#endif
