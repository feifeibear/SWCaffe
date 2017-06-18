#include <stdio.h>
#include <math.h>
#include <stdlib.h>

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

  printf(" B : %d, Ro : %d, Co : %d, Ri : %d, Ci : %d, No : %d, Ni : %d, K : %d\n",
                B,      Ro,       Co,     Ri, Ci, No, Ni, K);

  for(cB = 0; cB<B; cB++)
  for(cRo=0; cRo<Ro; cRo++)
    for(cCo=0; cCo<Co; cCo++)
      for(cNo=0; cNo<No; cNo++)
          for(cKr = 0 ;cKr<K; cKr++)
            for(cKc = 0; cKc<K; cKc++)
              for(cNi = 0; cNi<Ni; cNi++){
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


void test_forward_pad() {
  int Ni, No, B, Co, Ro, Ci, Ri, K, pad;
  //Ni = 128;
  //No = 256;
  //B  = 128;
  Ni = 16;
  No = 32;
  B  = 4;
  K  = 3;
  pad = 1;
  Ci = 8;
  Ri = 8;
  Co = Ci+2*pad-K+1;
  Ro = Ri+2*pad-K+1;

  int in_size     = Ni*B*Ci*Ri;
  int weight_size = Ni*No*K*K;
  int out_size    = No*B*Co*Ro;

  double* in = (double*)malloc(sizeof(double)*in_size);
  double* weight = (double*)malloc(sizeof(double)*weight_size);
  double* out = (double*)malloc(sizeof(double)*out_size);
  double* out_ref = (double*)malloc(sizeof(double)*out_size);

  for( int i = 0; i < in_size; ++i )
    in[i] = rand()/(double)RAND_MAX;

  for( int i = 0; i < weight_size; ++i )
    weight[i] = rand()/(double)RAND_MAX;

  for( int i = 0; i < out_size; ++i ) {
    out_ref[i] = 0;
    out[i] = 0;
  }


  for( int st = 0; st < 10; ++st ){
    /*
    sw_conv_forward_pad_impl_d(
        in,
        weight,
        out,
        Ci,
        Ri,
        K,
        Ni,
        No,
        B,
        pad);
    printf("sw version pad conv OK\n");
*/
    printf("inner loop before!\n");
    conv_forward_pad_impl<double>(
        in,
        weight,
        out_ref,
        Ci,
        Ri,
        K,
        Ni,
        No,
        B,
        pad);
    printf("inner loop OK!\n");
  }
  //if(!athread_halt())
  //  printf("athread halt not OK!\n");

  double sum = 0, sum_ref = 0;
  for( int i = 0; i < out_size; ++i ) {
   if( fabs(out_ref[i] - out[i]) > 1e-4 )
     printf("%lf vs %lf\n", out_ref[i], out[i]);
   sum += out[i];
   sum_ref += out_ref[i];
  }
  printf("sum %lf vs sum_ref %lf athread forward OK!\n", sum, sum_ref);

  free(out_ref);
  free(out);
  free(in);
  free(weight);

}


int main() {
  test_forward_pad();
  return 0;
}
