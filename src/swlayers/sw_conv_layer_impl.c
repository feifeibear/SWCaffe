#include <stdio.h>
#include <assert.h>
#include "athread.h"
#include <math.h>
//typedef double Type;
//#include "util.h"
#include "caffe/swlayers/sw_conv_layer_impl.h"
#include "caffe/util/matrix_trans.h"

//#ifndef ACC_TRANS
//#define ACC_TRANS
//#endif

extern SLAVE_FUN(conv_valid)();
extern SLAVE_FUN(conv_full)();
extern SLAVE_FUN(conv_pad)();
extern SLAVE_FUN(conv_full_pad)();

// high -> low
// B, N, R, C
inline int image_caffe_offset(int n, int c, int h, int w, int N, int C, int H, int W) {
  return (((n*C + c)*H + h)*W + w);
}
// R, C, N, B
inline int image_swdnn_offset(int n, int c, int h, int w, int N, int C, int H, int W) {
  return (((h*W + w)*C + c)*N + n);
}
// R, C, B, N
inline int image_swdnn_offset_back(int n, int c, int h, int w, int N, int C, int H, int W) {
  return (((h*W + w)*N + n)*C + c);
}
// No, Ni, Kr, Kc
inline int weight_caffe_offset(int no, int ni, int kr, int kc, int No, int Ni, int K) {
  return (( no*Ni + ni )*K + kr)*K + kc;
}
// Kr, Kc, No, Ni
inline int weight_swdnn_offset(int no, int ni, int kr, int kc, int No, int Ni, int K) {
  return ((( kr*K + kc )*No + no) * Ni + ni );
}
// Kr, Kc, Ni, No
inline int weight_swdnn_offset_back(int no, int ni, int kr, int kc, int No, int Ni, int K) {
  return ((( kr*K + kc )*Ni + ni) * No + no );
}

//#define weight_swdnn_to_caffe(in,out,B,N,H,W) swapBN_HW(in,out,H,W,B,N)
//#define weight_caffe_to_swdnn(in,out,B,N,H,W) swapBN_HW(in,out,B,N,H,W)
//#define image_caffe_to_swdnn_back(in,out,B,N,H,W)  swapBN_HW(in,out,B,N,H,W)



//typedef double Type;

typedef struct ConvData_st{
  Type* input; //0
  Type* weight; //8
  Type* output; //16
  //   24,  28,  32,  36, 40,  44,  48, 52, 56 
  int _Ni, _Ri, _Ci, _No, _K, _Ro, _Co, _B, _Costride, _bCo, _pad;
}ConvData;


static int init_flag = 0; 

void sw_conv_forward_pad_impl_d(
        const Type* in, 
        const Type* weight, 
        Type* out,
        //Type* bias,
        int Ci,
        int Ri,
        int K,
        int Ni,
        int No,
        int B,
        int pad)
{
    int i;
    int cKr, cKc, cNo;
    int cRo, cCo, cB;
    int cRi, cCi, cNi;
    int Ro = Ri+2*pad-K+1 , Co = Ci+2*pad-K+1;

    if(init_flag == 0){
      int rtcode = athread_init();
      if( rtcode != 1)
	      printf("thread init error, return code %d\n", rtcode);
      init_flag = 1 ;
    }

    ConvData* param = (ConvData*)malloc(sizeof(ConvData));
    param->input =  in;
    param->weight = weight;
    param->output = out;
	  param->_Ni = Ni;
	  param->_Ri = Ri;
	  param->_Ci = Ci;
	  param->_No = No;
	  param->_K  = K;
	  param->_Ro = Ri+2*pad-K+1;
	  param->_Co = Ci+2*pad-K+1;
	  param->_B  = B;
    param->_pad = pad;

    assert(param->_B >= 128 && param->_B%128 == 0);
    assert(param->_Ni >= 64 && param->_Ni%32 == 0);
    assert(param->_No >= 64 && param->_No%32 == 0);

    //fjr1buff 7.13
	  int Costride = (64*60*1024/8 - Ni*B-Ni*No)/(No*B);
	  param->_Costride = Costride;
    assert(Costride > 0);
	  int ldm_consume = 8*(Ni*No + No*B*Costride + Ni*B);
	  //printf("ldm comsumption is %d\n", ldm_consume/64);
	  assert(ldm_consume < 64*1024*64);
    //memset(param->output, (Type)0, sizeof(Type)*Ni*B*Ci*Ri);
	  //printf("befor init forward OK\n");

	  athread_spawn(conv_pad, param);
	  athread_join();

    free(param);
}



void sw_conv_backward_pad_impl_d(
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
        int B,
        int pad)
{
	  printf("begin Backward Pad Impl\n");

    int cKr, cKc, cNo;
    int cRo, cCo, cB;
    int cRi, cCi, cNi;
    int Ro = Ri+2*pad-K+1 , Co = Ci+2*pad-K+1;

    ConvData* param = (ConvData*)malloc(sizeof(ConvData));
    //Transformation and rot180: in (B, N, R, C) -> (R, C, N, B)
    if( init_flag == 0 ){
      int rtcode = athread_init();
      if( rtcode != 1 )
        printf("init error");
      init_flag = 1;
    }

    param->input  = in;
    param->weight = out_grad;
    param->output = weight_diff;
	  param->_Ni  = B;
	  param->_Ri  = Ro;//+2*pad-K+1;
	  param->_Ci  = Co;//+2*pad-K+1;
	  param->_No  = No;
	  param->_K   = Ci+2*pad-K+1;
	  param->_Ro  = K;
	  param->_Co  = K;
	  param->_B   = Ni;
    param->_pad = pad;

    assert(param->_B >= 128 && param->_B%128 == 0);
    assert(param->_Ni >= 64 && param->_Ni%32 == 0);
    assert(param->_No >= 64 && param->_No%32 == 0);

    //fjr1buff 7.13
	  int Costride = (64*55*1024/8-param->_Ni*param->_B-
            param->_Ni*param->_No)/
        (param->_No*param->_B);
	  param->_Costride = Costride;
    assert(Costride > 0);

    // weight_diff = conv(pad(in), out_grad, 'valid')
	  athread_spawn(conv_pad, param);
	  athread_join();

    param->input  =   out_grad;
    param->weight =   weight;
    param->output =   in_grad;
	  param->_Ni = No;
	  param->_Ri = Ro;
	  param->_Ci = Co;
	  param->_No = Ni;
	  param->_K  = K;
	  param->_Ro = Ri;
	  param->_Co = Ci;
	  param->_B  = B;
	  param->_pad  = pad;

    //fjr1buff
    Costride = (64*55*1024/8-param->_Ni*param->_B-param->_Ni*param->_No)/
            (param->_No*param->_B);
	  param->_Costride = Costride;
	  //printf("Costride is %d\n", Costride);
    assert(Costride > 0);

    //memset(my_in_grad, 0, sizeof(Type)*Ni*B*Ci*Ri);
    // pad_inv(in_grad) = conv(out_grad, rot180(weight), 'full')
	  //  athread_spawn(conv_full_pad, param);
	  athread_spawn(conv_full_pad,param);
    athread_join();

    free(param);
}
