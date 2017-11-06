/***************
 * GEMM PLAN 
 * Jerry Fang 
 * 2017 June 15 
 *
 * input  is of dim(B, Ni)
 * weight is of dim(Ni, No)
 * ouput  is of dim(B, No)
 *
 * No overlap input DMA and weight DMA
 * ************/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <slave.h>
#include <math.h>
#include "simd.h"
#include "dma.h"
#include "caffe/swlayers/gemm.h"

#define SIMDSIZE  4
#define SIMDType  floatv4
#define Type      float

void conv_pad_float(ConvData* param)
{
  int cB, cNi, cRi, cCi, cKr, cKc, ccCore, crCore, cNo;
  int ii, jj, cRo, cCo;
  int CoStart;
  int id = athread_get_id(-1);
  int cid = id%8, rid = id/8;
  int input_calc_index=1, input_load_index=0;
  int weight_calc_index=1, weight_load_index=0;
  int i, j;
  int Ni, Ri, Ci, No, K, Ro, Co, B, pad;
  Ni = param->_Ni;
  Ri = param->_Ri;
  Ci = param->_Ci;
  No = param->_No;
  K  = param->_K;
  Ro = param->_Ro;
  Co = param->_Co;
  B  = param->_B;
  pad = param->_pad;
  int CStride=param->_Costride;

//B, Ni, Ci, Ri
//fjr1buf
  SIMDType* local_input  = (SIMDType*) (long)ldm_malloc(sizeof(Type)*Ni*B/8/8);
  int local_input_size = Ni*B/8/8/SIMDSIZE;
//No, Ni, K, K
//fjr1buf
  Type* local_weight = (Type*)(long) ldm_malloc(sizeof(Type)*Ni*No/8/8);
  int local_weight_size = Ni*No/64;
//B, No, Co, Ro
  SIMDType* local_output = (SIMDType*)(long) ldm_malloc(sizeof(Type)*No*B/8/8*CStride);
  int local_output_size = No*B/8/8*CStride;

//  Type local_weight[K*K*Ni/64*No];
//initilize DMA variables
  volatile int  input_replyget = 0, weight_replyget = 0,  replyput = 0;
  dma_desc dma_get_input, dma_get_weight, dma_get_output, dma_put_output;

  dma_set_op(&dma_get_input, DMA_GET);
  dma_set_mode(&dma_get_input, PE_MODE);
  dma_set_reply(&dma_get_input, &input_replyget);

  dma_set_op(&dma_get_weight, DMA_GET);
  dma_set_mode(&dma_get_weight, PE_MODE);
  dma_set_reply(&dma_get_weight, &weight_replyget);

  dma_set_op(&dma_put_output, DMA_PUT);
  dma_set_mode(&dma_put_output, PE_MODE);
  dma_set_reply(&dma_put_output, &replyput);

  //DMA for local_iutput(B/8, Ni/8)
  dma_set_size(&dma_get_input, B*Ni/8/8/SIMDSIZE*sizeof(SIMDType));
  dma_set_bsize(&dma_get_input, B/SIMDSIZE/8*sizeof(SIMDType));
  dma_set_stepsize(&dma_get_input, B/SIMDSIZE/8*7*sizeof(SIMDType));

  //DMA for local_weight(No/8, Ni/8)
  dma_set_size(&dma_get_weight, No*Ni/8/8*sizeof(Type));
  dma_set_bsize(&dma_get_weight, Ni/8*sizeof(Type));
  dma_set_stepsize(&dma_get_weight, Ni/8*7*sizeof(Type));

  //DMA for local_output(B/8, No/8)
  dma_set_size(&dma_get_output, B*No/8/8/SIMDSIZE*sizeof(SIMDType));
  dma_set_bsize(&dma_get_output, B/SIMDSIZE/8*sizeof(SIMDType));
  dma_set_stepsize(&dma_get_output, B/SIMDSIZE/8*7*sizeof(SIMDType));

  //DMA for local_output(B/8, No/8)
  dma_set_size(&dma_put_output, B*No/8/8/SIMDSIZE*sizeof(SIMDType));
  dma_set_bsize(&dma_put_output, B/SIMDSIZE/8*sizeof(SIMDType));
  dma_set_stepsize(&dma_put_output, B/SIMDSIZE/8*7*sizeof(SIMDType));

//1st weight_load
  Type* weight_start = (Type*)param->weight+(cid*No/8*Ni+rid*Ni/8);
  Type* weight_ptr = weight_start;

  dma(dma_get_weight, (long)(weight_ptr), (long)(local_weight));
  dma_wait(&weight_replyget, 1); weight_replyget = 0;

  //DMA for 1st input
  Type* input_start = (Type*)param->input+rid*B/8+cid*Ni/8*B;
  dma(dma_get_input, (long)(input_start), (long)(local_input));
  dma_wait(&input_replyget, 1); input_replyget = 0;

  for(CoStart=0; CoStart<Co; CoStart+=CStride){
    int CoEnd = CoStart+CStride;
    int CiEnd = CoStart+CStride+K;
    if(CoEnd > Co)
      CoEnd = Co;
    //fjrpad
    if(CiEnd > Ci + 2*pad)
      CiEnd = Ci + 2*pad;
    //input init
    for(cRo=0; cRo<Ro; ++cRo){

      Type* output_ptr = (Type*)param->output + rid*B/8 + cid*No/8*B + B*No*(cRo*Co+CoStart);
	    //init local_output
	    for(i = 0; i<local_output_size/SIMDSIZE; ++i)
		    local_output[i] = 0.0;

      for(cKr=0; cKr<K; ++cKr){

        cRi = cRo+cKr;
        //fjrpad
        int lr = cRi - pad;
        if(!(lr >= 0 && lr < Ri))
            continue;

		    for(cCi=CoStart; cCi<CiEnd; ++cCi){
            //fjrpad
            int lc = cCi - pad;
            if(!(lc >= 0 && lc < Ci))
                continue;

    			  dma(dma_get_input, (long)(input_start + (lc+lr*Ci)*Ni*B), (long)(local_input));
    			  dma_wait(&input_replyget, 1); input_replyget = 0;

            for(cKc=0; cKc<K; ++cKc){

              cCo = cCi - cKc;
              if(cCo >= CoStart && cCo < CoEnd){

			          dma(dma_get_weight, (long)(weight_ptr + (cKc+cKr*K)*Ni*No), (long)(local_weight));
			          dma_wait(&weight_replyget, 1); weight_replyget = 0;

    			  	  gemmfloat((Type*)(local_input),
    			  	    (Type*)(local_weight),
    			  	    (Type*)(local_output + (cCo-CoStart)*No*B/64/SIMDSIZE),
    			  	    B/8/4,
    			  	    B/8/4,
    			  	    No/8,
    			  	    Ni/8,
    			  	    rid,
    			  	    cid);
			        }//if
            }//cKc
          }//cCi

      }//cKc

      //input back outer
      jj=0;
      for(ii=CoStart; ii<CoEnd; ++ii){
          dma(dma_put_output, (long)(output_ptr), (long)(local_output+jj*B*No/64/SIMDSIZE));
          dma_wait(&replyput, 1); replyput = 0;
		      output_ptr += B*No;
          jj++;
      }

    }//cRo

  }//CoStart

  //fjr1buf
  ldm_free(local_input, sizeof(SIMDType)*local_input_size);
  //fjr1buf
  ldm_free(local_weight, sizeof(Type)*local_weight_size);
  ldm_free(local_output, sizeof(Type)*local_output_size);

}//main func
#undef SIMDType
#undef Type
