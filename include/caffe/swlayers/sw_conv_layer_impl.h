/*************************************************************************
	> File Name: sw_conv_forward_impl.h
	> Author: THU Code Farmer
	> mail: thu@thu.thu
	> Created Time: Fri 30 Dec 2016 04:17:22 PM CST
 ************************************************************************/
#ifndef SW_CONV_FORWARD_IMPL_H_
#define SW_CONV_FORWARD_IMPL_H_

void sw_conv_forward_impl_d(
        const double * in,
        const double * weight,
        double * out,
        //Type* bias,
        int Ci,
        int Ri,
        int K,
        int Ni,
        int No,
        int B);

void sw_conv_backward_impl_d(
        const double* in,
        const double* out_grad,
        const double* weight,
        double* in_grad,
        double* weight_diff,
        //Type* bias_grad,
        int Ci,
        int Ri,
        int K,
        int Ni,
        int No,
        int B);

#endif
