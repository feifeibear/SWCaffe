/*************************************************************************
	> File Name: conv_layer_impl.h
	> Author: Jiarui Fang 
	> mail: fang_jiarui@163.com
  > Created Time: Fri 30 Dec 2016 10:24:37 AM CST
  > This file provide a MPE version for correctness check for SPE version on Sunway
 ************************************************************************/
#ifndef CONVLAYER_IMPL
#define CONVLAYER_IMPL

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
}

template<typename Type>
void conv_forward_pad_impl(Type* input,
    Type* weight,
    Type* output,
        int Ci,
        int Ri,
        int K,
        int Ni,
        int No,
        int B,
        int pad)
{
  int cB,cNo,cNi,cRo,cCo,cKr,cKc;
  int Co = Ci+2*pad-K+1;
  int Ro = Ri+2*pad-K+1;

  DLOG(INFO) << "MPE VERSION: " << "Ci " << Ci << " Ri " << Ri << " K " << K 
    << " Ni " << Ni << " No " << No << " B " << B << " Co " << Co << " Ro " << Ro;

  for(cB = 0; cB<B; cB++)
    for(cNo=0; cNo<No; cNo++)
      for(cNi = 0; cNi<Ni; cNi++)
        for(cRo=0; cRo<Ro; cRo++)
          for(cCo=0; cCo<Co; cCo++)
            for(cKr = 0 ;cKr<K; cKr++)
              for(cKc = 0; cKc<K; cKc++)
              {
                  int cRi = cRo+cKr-pad;
                  int cCi = cCo+cKc-pad;
                  if(cRi >= 0 && cRi < Ri && cCi >= 0 && cCi < Ci) {
                    *(output + outGetIdx(cB, cNo, cRo, cCo, B, No, Ro, Co)) +=
                      *(input + inGetIdx(cB, cNi, cRi, cCi, B, Ni, Ri, Ci)) *
                      *(weight + weightGetIdx(cNo, cNi, cKr, cKc, No, Ni, K));
                  }
              }
  printf("conv output forward is OK\n");
}

template<typename Type>
void conv_forward_impl(
    const Type* input,
    const Type* weight,
    Type* output,
    //const Type* bias,
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

  //DLOG(INFO) << "Ci " << Ci << "Ri " << Ri << "K " << K 
  //  << "Ni " << Ni << "No " << No << "B " << B;

  memset(output, (Type)0, sizeof(Type)*No*B*Co*Ro);
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

  //printf("conv output forward is OK\n");
}

template<typename Type>
void conv_backward_impl(
        const Type* in,
        const Type* out_grad,
        const Type* weight,
        Type* in_grad,
        Type* weight_diff,
        //Type* bias_grad,
        int Ci,
        int Ri,
        int K,
        int Ni,
        int No,
        int B)
{
    int cNi, cNo, cB, cCo, cRo, cCi, cRi, cKr, cKc;
    //DLOG(INFO) << "back: Ci " << Ci << " Ri " << Ri << " K " << K 
    //<< " Ni " << Ni << " No " << No << " B " << B;
    int Co = Ci-K+1;
    int Ro = Ri-K+1;
    int Pad = K-1;
    int gr, gc, lr, lc;

// in_grad = conv(out_grad, rot180(weight), 'full')
    memset(in_grad, 0, sizeof(Type)*Ni*B*Ci*Ri);
    for( cB = 0; cB < B; cB++ ) {
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
                    sum +=
                    *(out_grad+outGetIdx(cB, cNo, lr, lc, B, No, Ro, Co)) *
                    *(weight+weightGetIdx( cNo, cNi,  K-1-cKr, K-1-cKc, No, Ni, K));
                  }
                }//cKc
                *(in_grad+inGetIdx(cB, cNi, cRi, cCi, B, Ni, Ri, Ci)) += sum;
            }//cRi
    }//cB

// weight_diff = conv(rot180(in), out_grad, 'valid')
    memset(weight_diff, 0, sizeof(Type)*Ni*No*K*K);
    for(cB = 0; cB<B; cB++)
      for(cNo=0; cNo<No; cNo++)
        for(cNi = 0; cNi<Ni; cNi++)
          for(cKr = 0 ;cKr<K; cKr++)
		        for(cKc = 0; cKc<K; cKc++)
              for(cRo=0; cRo<Ro; cRo++)
                for(cCo=0; cCo<Co; cCo++)
                 {
                  *(weight_diff+weightGetIdx(cNo, cNi, cKr, cKc, No, Ni, K)) += 
                    *(in + inGetIdx(cB, cNi, cRo+cKr, cCo+cKc, B, Ni, Ri, Ci)) * 
                    *(out_grad + outGetIdx(cB, cNo, cRo, cCo, B, No, Ro, Co));
                }
}


template<typename Type>
void conv_backward_pad_impl(
        const Type* in,
        const Type* out_grad,
        const Type* weight,
        Type* in_grad,
        Type* weight_diff,
        int Ci,
        int Ri,
        int K,
        int Ni,
        int No,
        int B,
        int pad)
{
    int cNi, cNo, cB, cCo, cRo, cCi, cRi, cKr, cKc;
    DLOG(INFO) << "back: Ci " << Ci << " Ri " << Ri << " K " << K 
    << " Ni " << Ni << " No " << No << " B " << B;
    //int Co = Ci-K+1;
    //int Ro = Ri-K+1;
    //pad
    int Co = Ci+2*pad-K+1;
    int Ro = Ri+2*pad-K+1;
    int gr, gc, lr, lc;
    //printf("Ci=%d Ri=%d Co=%d Ro=%d\n",Ci,Ri,Co,Ro);
    // in_grad = conv(out_grad, rot180(weight), 'full')
    // can be implemented with sw_slave_conv_pad_full
    //TODO
    memset(in_grad, 0, sizeof(Type)*Ni*B*Ci*Ri);
    for( cB = 0; cB < B; cB++ ) {
       for( cNo = 0; cNo < No; cNo++ )
        for( cNi = 0; cNi < Ni; cNi++ )
              for(cKr=0; cKr < K; ++cKr)
                for(cKc=0; cKc < K; ++cKc){
                  for(cRo=0;cRo<Ro;cRo++){
                   int cRi = cRo +cKr -pad;
                   for(cCo=0;cCo<Co;cCo++){
                    int cCi = cCo+cKc - pad;
                    if( !(cCi >=0 && cCi < Ci && cRi >= 0 && cRi < Ri) )
                    continue;
                    *(in_grad+inGetIdx(cB, cNi, cRi, cCi, B, Ni, Ri, Ci)) +=
                    *(out_grad+outGetIdx(cB, cNo, cRo, cCo, B, No, Ro, Co)) *
                    *(weight+weightGetIdx( cNo, cNi,  K-1-cKr, K-1-cKc, No, Ni, K));
                }//cCo
              }//cRo 
            }//cKc
      }//cB

// weight_diff = conv(pad(in), out_grad, 'valid')
// can be implemented with sw_slave_conv_pad
    memset(weight_diff, 0, sizeof(Type)*Ni*No*K*K);
    for(cB = 0; cB<B; cB++)
      for(cNo=0; cNo<No; cNo++)
        for(cNi = 0; cNi<Ni; cNi++)
          for(cKr = 0 ;cKr<K; cKr++)
		        for(cKc = 0; cKc<K; cKc++)
              for(cRo=0; cRo<Ro; cRo++)
                for(cCo=0; cCo<Co; cCo++)
                {
                  int cCi_map = cCo + cKc - pad;
                  int cRi_map = cRo + cKr - pad;
                  //int cCi_map = cCo - pad;
                  //int cRi_map = cRo - pad;
                  if( !(cCi_map >= 0 && cCi_map < Ci && cRi_map >= 0 && cRi_map < Ri) )
                    continue;
                  *(weight_diff+ weightGetIdx(cNo, cNi, cKr, cKc, No, Ni, K)) += 
                    *(in + inGetIdx(cB, cNi, cRi_map, cCi_map, B, Ni, Ri, Ci)) * 
                    *(out_grad + outGetIdx(cB, cNo, cRo, cCo, B, No, Ro, Co));
                }

}


#endif
