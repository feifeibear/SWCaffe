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

void sw_conv_forward_pad_impl_f(
        const float* in,
        const float* weight,
        float* out,
        int Ci,
        int Ri,
        int K,
        int Ni,
        int No,
        int B,
        int pad);
void sw_conv_forward_pad_impl_f_ori(
        const float* in,
        const float* weight,
        float* out,
        int Ci,
        int Ri,
        int K,
        int Ni,
        int No,
        int B,
        int pad);



void sw_conv_forward_pad_impl_d(
        const double * in,
        const double * weight,
        double * out,
        //Type* bias,
        int Ci,
        int Ri,
        int K,
        int Ni,
        int No,
        int B,
        int pad);

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

void sw_conv_backward_pad_impl_d(
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
        int B,
        int pad);

void sw_conv_backward_pad_impl_f(
        const float* in,
        const float* out_grad,
        const float* weight,
        float* in_grad,
        float* weight_diff,
        //Type* bias_grad,
        int Ci,
        int Ri,
        int K,
        int Ni,
        int No,
        int B,
        int pad);

void sw_conv_backward_pad_weight_diff_impl_f(
        const float* in,
        const float* out_grad,
        const float* weight,
        float* in_grad,
        float* weight_diff,
        int Ci,
        int Ri,
        int K,
        int Ni,
        int No,
        int B,
        int pad);

void sw_conv_backward_pad_in_diff_impl_f(
        const float* in,
        const float* out_grad,
        const float* weight,
        float* in_grad,
        float* weight_diff,
        int Ci,
        int Ri,
        int K,
        int Ni,
        int No,
        int B,
        int pad);



void sw_conv_backward_pad_weight_diff_impl_d(
        const double* in,
        const double* out_grad,
        const double* weight,
        double* in_grad,
        double* weight_diff,
        int Ci,
        int Ri,
        int K,
        int Ni,
        int No,
        int B,
        int pad);

void sw_conv_backward_pad_in_diff_impl_d(
        const double* in,
        const double* out_grad,
        const double* weight,
        double* in_grad,
        double* weight_diff,
        int Ci,
        int Ri,
        int K,
        int Ni,
        int No,
        int B,
        int pad);


#endif
