/*************************************************************************
	> File Name: ConvLayer_impl.c
	> Author: THU Code Farmer
	> mail: thu@thu.thu
	> Created Time: Fri 30 Dec 2016 10:25:15 AM CST
 ************************************************************************/

#include <stdio.h>
#include <string.h>
#include "test_conv_layer_impl.hpp"
#include "glog/logging.h"

//TODO B, Ni, Ci, Ri
inline int inGetIdx(int cB, int cNi, int cRi, int cCi, int B, int Ni, int Ri,int Ci){
//  return cB + cNi*B + cCi*B*Ni + cRi*Ci*Ni*B;
  return (((cB * Ni + cNi)*Ri + cRi)*Ci + cCi);
}

//TODO B, No, Co, Ro
inline int outGetIdx(int cB, int cNo, int cRo, int cCo, int B, int No, int Ro, int Co){
//  return cB + cNo*B + cCo*B*No + cRo*Co*No*B;
  return (((cB * No + cNo)*Ro + cRo)*Co + cCo);
}

//Ni, No, K, K
//TODO FJR diff from SW version
//Kc, Kr, Ni, No
inline int weightGetIdx(int cNo, int cNi, int cKr, int cKc,  int No, int Ni, int K){
  //return cNo + cNo*Ni + (cKr*K + cKc)*No*Ni;
  return (((cNo*Ni + cNi)*K + cKr)*K + cKc);
}

inline int offset(const int n, const int c, const int h,
    const int w,
    const int batchs,
    const int channels,
    const int height,
    const int width) {
  return ((n * channels + c) * height + h) * width + w;
//  return ((c * batchs + n) * height + h) * width + w;
}

template<typename Type>
void conv_forward_impl(Type* input,
    Type* weight,
    Type* output,
    Type* bias,
        int Ci,
        int Ri,
        int K,
        int Ni,
        int No,
        int B)
{
  int cB,cNo,cNi,cRo,cCo,cKr,cKc;
  int Co = Ci-K+1;
  int Ro = Ri-K+1;

  DLOG(INFO) << "Ci " << Ci << "Ri " << Ri << "K " << K 
    << "Ni " << Ni << "No " << No << "B " << B;

  //bias is (No)
  for(cRo = 0; cRo < Ro; cRo++)
    for(cCo=0; cCo<Co; cCo++)
      for(cNo=0; cNo<No; cNo++)
        for(cB = 0; cB<B; cB++)
          *(output + outGetIdx(cB, cNo, cRo, cCo, B, No, Ro, Co)) = 
            *(bias + cNo);

  for(cRo=0; cRo<Ro; cRo++)
    for(cCo=0; cCo<Co; cCo++){
      for(cNo=0; cNo<No; cNo++){
          for(cKr = 0 ;cKr<K; cKr++)
            for(cKc = 0; cKc<K; cKc++){
              for(cNi = 0; cNi<Ni; cNi++){
                for(cB = 0; cB<B; cB++){
                  *(output + outGetIdx(cB, cNo, cRo, cCo, B, No, Ro, Co)) += 
                    *(input + inGetIdx(cB, cNi, cRo+cKr, cCo+cKc, B, Ni, Ri, Ci)) * 
                    *(weight + weightGetIdx(cNo, cNi, cKr, cKc, No, Ni, K));
                }
            }
          }
      }//cNo
    }//cCo

  printf("conv output forward is OK\n");
}

template<typename Type>
void conv_backward_impl(Type* in,
        Type* out_grad,
        Type* weight,
        Type* in_grad,
        Type* weight_diff,
        Type* bias_grad,
        int Ci,
        int Ri,
        int K,
        int Ni,
        int No,
        int B)
{
    int cNi, cNo, cB, cCo, cRo, cCi, cRi, cKr, cKc;
    DLOG(INFO) << "back: Ci " << Ci << " Ri " << Ri << " K " << K 
    << " Ni " << Ni << " No " << No << " B " << B;
    int Co = Ci-K+1;
    int Ro = Ri-K+1;
    int Pad = K-1;
    int gr, gc, lr, lc;

// in_grad = conv(out_grad, rot180(weight), 'full')
    for( cB = 0; cB < B; cB++ ){
        for( cNo = 0; cNo < No; cNo++ )    
            for( cNi = 0; cNi < Ni; cNi++ )    
                for( cCi = 0; cCi < Ci; cCi++)
                    for( cRi = 0; cRi < Ri; cRi++){
                        Type sum = 0.0;
                        for(cKr=0; cKr < K; ++cKr)
                            for(cKc=0; cKc < K; ++cKc){
                                gr = cKr+cRi;
                                gc = cKc+cCi;
                                lr = gr-Pad;
                                lc = gc-Pad;
                                if(lr >= 0 && lr < Ro && lc >= 0 && lc < Co){
                                    sum += *(out_grad+outGetIdx(cB, cNo, lr, lc, B, No, Ro, Co)) * 
                                      //*(weight + weightGetIdx(cNo, cNi, cKr, cKc,  No, Ni, K)); 
                                      //TODO rot180 for weight
                                      *(weight+weightGetIdx( cNo, cNi,  K-1-cKr, K-1-cKc, No, Ni, K)); 
                                }
                            }//cKc
                        *(in_grad+inGetIdx(cB, cNi, cRi, cCi, B, Ni, Ri, Ci)) += sum;
                    }
    }//cB
  
    memset(bias_grad, 0, No*sizeof(Type));
//TODO bias_grad = out_grad(B, No, :, :)
    for(cRo = 0; cRo < Ro; cRo++)
      for(cCo=0; cCo<Co; cCo++)
        for( cB = 0; cB < B; cB++ )
          for( cNo = 0; cNo < No; cNo++ )    
              *(bias_grad + cNo) += *(out_grad + 
                outGetIdx(cB, cNo, cRo, cCo, B, No, Ro, Co));

// weight_diff = conv(rot180(in), out_grad, 'valid')
    for(cB = 0; cB<B; cB++)
      for(cNo=0; cNo<No; cNo++)
        for(cNi = 0; cNi<Ni; cNi++)
          for(cKr = 0 ;cKr<K; cKr++)
		        for(cKc = 0; cKc<K; cKc++)
              for(cRo=0; cRo<Ro; cRo++)
                for(cCo=0; cCo<Co; cCo++)
                {
                  *(weight_diff+ weightGetIdx(cNo, cNi, cKr, cKc, No, Ni, K)) += 
                    *(in + inGetIdx(cB, cNi, cRo+cKr, cCo+cKc, B, Ni, Ri, Ci)) * 
                    //rot180 for in
                    //*(in + inGetIdx(cB, cNi, Ri-1-(cRo+cKr), Ci-1-(cCo+cKc), B, Ni, Ri, Ci)) * 
                    *(out_grad + outGetIdx(cB, cNo, cRo, cCo, B, No, Ro, Co));
                }
}

