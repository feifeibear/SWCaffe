#include <stdio.h>
#include <stdlib.h>
#include <athread.h>
#include <simd.h>
#include "caffe/util/matrix_trans.h"

extern SLAVE_FUN(swapBN)();
extern SLAVE_FUN(swapNBHW)();

void printf_buf(char *msg, Type * buf,unsigned int buf_len)
{
    int i;
    printf("%s\n",msg);
    for(i=0;i<buf_len;i++)
    {
        printf("%5.3f ",buf[i]);
        if( (i+1)%32 == 0 )
            printf("\n");
    }
    printf("\n");
}
void format_result(double dPacketSize,unsigned char cMsg[20])
{
	if(dPacketSize<1024)               {sprintf(cMsg, "%3.2f  B", dPacketSize);}
    else if(dPacketSize<1024*1024)     {sprintf(cMsg, "%3.2f KB", dPacketSize/1024);}
    else if(dPacketSize<1024*1024*1024){sprintf(cMsg, "%3.2f MB", dPacketSize/(1024*1024));}
    else                            {sprintf(cMsg, "%3.2f GB", dPacketSize/(1024*1024*1024));}
}
inline unsigned long rpcc()
{
    //unsigned long rpcc;
    //asm volatile("rtc %0":"=r"(rpcc));
	//return rpcc;
    struct timeval   val;
	gettimeofday(&val,NULL);
	return (val.tv_sec*1000000 + val.tv_usec);        
}
// high -> low
// B, N, W, H
inline int image_caffe_offset(int b, int n, int h, int w, int B, int N, int H, int W) {
  return (((b*N + n)*H + h)*W + w);
}
// W, H, N, B
inline int image_swdnn_offset(int b, int n, int h, int w, int B, int N, int H, int W) {
  return (((h*W + w)*N + n)*B + b);
}

void MatrixInvert(Type*in,Type*out,int B,int N,int H,int W)
{
	int cRi, cCi, cNi, cB;
	
	Type* sout   = (Type*)malloc(sizeof(Type)*B*N*H*W);
    int nNB = N*B;
		
	SlaveParam param;
	param.B = B;
	param.N = N;
	param.H = H;
	param.W = W;
	param.pIn = in;
	param.pOut = sout;	
	param.nCount = nNB/NUM_THREADS;
	int nTmp = nNB%NUM_THREADS;
	param.nBNThreadsNum = param.nCount >0 ? NUM_THREADS:nTmp;
	param.nBNLeftThreadsNum = param.nCount >0 ? nTmp:0;
	
	athread_set_num_threads(NUM_THREADS);
	athread_spawn(swapBN,(void *)&param);
	
	int nHW = H*W;
	int nWSplit = nHW>>2;
	
	if(nWSplit < HWSIZE)
		param.splitHW=nWSplit;
	else
		param.splitHW=HWSIZE;
	
	param.splitNB=BSIZE > B ? B : BSIZE;
	nTmp = NUM_THREADS*param.splitNB;	
	param.nCount = nNB/nTmp;
	param.nNBHWThreadsNum = param.nCount >0 ? NUM_THREADS:(nNB/param.splitNB)%NUM_THREADS;
	param.nNBHWLeftThreadsNum = param.nCount >0 ? nNB%nTmp:0;
	
	athread_join();		
	
	param.pIn = sout;
	param.pOut = out;	
	athread_spawn(swapNBHW,(void *)&param);
	
	//Left process
	int nLeft = nHW-(nWSplit<<2);
	
	if(nLeft >0)
	{
		cRi = H -1;
		int nHWN = cRi*W*N*B;
		int nBHW = B*H*W;
		int nHW1 = cRi*W;
		for(cCi = nLeft%W; cCi < W; ++cCi)
			for(cNi = 0; cNi < N; ++cNi)
			  for(cB = 0; cB < B; ++cB)
				out[nHWN+cCi*nNB+cNi*B+cB] =  sout[cNi*nBHW+cB*nHW+nHW1+cCi];    
	}	
	athread_join();	
	
	free(sout);
}
