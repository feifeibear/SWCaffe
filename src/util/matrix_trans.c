#include <stdio.h>
#include <stdlib.h>
#include <athread.h>
#include <simd.h>
#include "caffe/util/matrix_trans.h"

extern SLAVE_FUN(swapBN)();
extern SLAVE_FUN(swapNBHW)();
extern SLAVE_FUN(swapNBHW_ROLL)();
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
inline int get_split_size(int nSize,int nMaxSize)
{
	int nVal = nSize/nMaxSize,nSplitSize = 0;
	if(nVal<1) 
	{
      nSplitSize = nSize - nSize%4;
	}
	else if(nVal>=nMaxSize) nSplitSize = nMaxSize;
	else{
		int nModHW = nSize - nSize%4,nTmp=0;

		nSplitSize = 0;
		for(;nVal<nMaxSize;nVal++)
		{
			nTmp = nModHW/nVal;
			if(nTmp <nMaxSize && (nTmp % 4 == 0))
			{
				nSplitSize = nTmp;
				break;
			}
		}
		if(nSplitSize<16){
			nSplitSize = (nModHW>>2);
			nSplitSize = nSplitSize - nSplitSize%4;
		}
	}
	return nSplitSize;
}

void swapBN(Type*in,Type*out,int B,int N,int H, int W)
{
	int nNB = N*B;		
	SlaveParam param;
	
	param.B = B;
	param.N = N;
	param.H = H;
	param.W = W;
	param.pIn = in;
	param.pOut = out;	
	param.nCount = nNB/NUM_THREADS;
	int nTmp = nNB%NUM_THREADS;
	param.nBNThreadsNum = param.nCount >0 ? NUM_THREADS:nTmp;
	param.nBNLeftThreadsNum = param.nCount >0 ? nTmp:0;
	
	athread_spawn(swapBN,(void *)&param);
	athread_join();
}

void swapBN_HW(Type*in,Type*out,int B,int N,int H, int W)
{
	int cRi, cCi, cNi, cB;
	int nHW = H*W;
	int nNB = N*B;
	//process the problem of the (H,W) very small
	if(nHW < 4)
	{
	    for(cCi = 0; cCi < nHW; ++cCi)
			for(cRi = 0; cRi < nNB; ++cRi)
				out[cCi*nNB+cRi] = in[cRi*nHW+cCi]; 
        return;
	}
	
		
	SlaveParam param;
	param.B = B;
	param.N = N;
	param.H = H;
	param.W = W;
	param.pIn = in;
	param.pOut = out;	
	
	param.splitHW = get_split_size(nHW,HWSIZE);		
	param.splitNB = get_split_size(nNB,BSIZE);		
	int nTmp = NUM_THREADS*param.splitNB;	
	//printf("N=%d B=%d H=%d W=%d splitNB=%d splitHW=%d\n",N,B,H,W,param.splitNB,param.splitHW);
	param.nCount = nNB/nTmp;
	nTmp = (nNB/param.splitNB)%NUM_THREADS;
	param.nNBHWThreadsNum = param.nCount >0 ? NUM_THREADS:nTmp;
	param.nNBHWLeftThreadsNum = param.nCount >0 ? nTmp:0;
	
	athread_spawn(swapNBHW,(void *)&param);
	//process the slave core left data
	int nHWLeft = nHW%(param.splitHW);	
	int nBNLeft = nNB%(param.splitNB);	
	if(nHWLeft >0 || nBNLeft >0)
	{
		int nC = nHW - nHWLeft;
		int nR = nNB - nBNLeft;
		
		for(cCi = nC; cCi < nHW; ++cCi)
			for(cRi = 0; cRi < nNB; ++cRi)
				out[cCi*nNB+cRi] = in[cRi*nHW+cCi]; 
    
		for(cRi = nR; cRi < nNB; ++cRi)
			for(cCi = 0; cCi < nC; ++cCi)
				out[cCi*nNB+cRi] = in[cRi*nHW+cCi]; 
    }	
    athread_join();			
}
void weight_caffe_to_swdnn_back(Type*in,Type*out,int B,int N,int H,int W)
{
	int cRi, cCi, cNi, cB;
	//process the problem of the (H,W) very small
	int nHW = H*W;
	int nNB = N*B;
	int nTmp = 0;
	if(nHW < 4)
	{
		nTmp = nHW-1;
	    for(cCi = 0; cCi <nHW; ++cCi)
			for(cRi = 0; cRi < nNB; ++cRi)
				out[(nTmp-cCi)*nNB+cRi] = in[cRi*nHW+cCi]; 
        return;
	}
	
	Type* sout   = (Type*)malloc(sizeof(Type)*B*N*H*W);
	if(sout == NULL)
	{
		printf("weight_caffe_to_swdnn_back malloc failure!\n");
		return;
	}
	swapBN(in,sout,B,N,H,W);
	
	SlaveParam param;
	param.B = B;
	param.N = N;
	param.H = H;
	param.W = W;
	param.pIn = sout;
	param.pOut = out;	
	
	param.splitHW = get_split_size(nHW,HWSIZE);		
	param.splitNB = get_split_size(nNB,BSIZE);		
	nTmp = NUM_THREADS*param.splitNB;	
	//printf("splitHW=%d splitNB=%d\n",param.splitHW,param.splitNB);
	
	param.nCount = nNB/nTmp;
	nTmp = (nNB/param.splitNB)%NUM_THREADS;
	param.nNBHWThreadsNum = param.nCount >0 ? NUM_THREADS:nTmp;
	param.nNBHWLeftThreadsNum = param.nCount >0 ? nTmp:0;
	
	athread_spawn(swapNBHW_ROLL,(void *)&param);
	//process the slave core left data
	int nHWLeft = nHW%(param.splitHW);	
	int nBNLeft = nNB%(param.splitNB);	
	if(nHWLeft >0 || nBNLeft >0)
	{
		int nC = nHW - nHWLeft;
		int nR = nNB - nBNLeft;
		nTmp = nHW-1;
	    for(cCi = nC; cCi < nHW; ++cCi)
			for(cRi = 0; cRi < nNB; ++cRi)
				out[(nTmp-cCi)*nNB+cRi] = sout[cRi*nHW+cCi]; 
    
		for(cRi = nR; cRi < nNB; ++cRi)
			for(cCi = 0; cCi < nC; ++cCi)
				out[(nTmp-cCi)*nNB+cRi] = sout[cRi*nHW+cCi]; 
    }	
    athread_join();
	
	free(sout);
}
void image_caffe_to_swdnn(Type*in,Type*out,int B,int N,int H,int W)
{
	int cRi, cCi, cNi, cB;
	//process the problem of the (H,W) very small
	int nHW = H*W;
	if(nHW < 4)
	{
	   for(cB = 0; cB < B; ++cB)
		  for(cNi = 0; cNi < N; ++cNi)
			  for(cRi = 0; cRi < H; ++cRi)
				  for(cCi = 0; cCi < W; ++cCi)
					   out[image_swdnn_offset(cB, cNi, cRi, cCi, B, N, H, W)] = in[image_caffe_offset(cB, cNi, cRi, cCi, B, N, H, W)];
	   return;
	}
	Type* sout   = (Type*)malloc(sizeof(Type)*B*N*H*W);
	if(sout == NULL)
	{
		printf("image_caffe_to_swdnn malloc failure!\n");
		return;
	}
	swapBN(in,sout,B,N,H,W);
	swapBN_HW(sout,out,B,N,H,W);
	
	free(sout);
}
void image_swdnn_to_caffe(Type*in,Type*out,int B,int N,int H,int W)
{
	int cRi, cCi, cNi, cB;
	//process the problem of the (H,W) very small
	int nBN = B*N;
	if(nBN < 4)
	{
	   for(cB = 0; cB < B; ++cB)
		  for(cNi = 0; cNi < N; ++cNi)
			  for(cRi = 0; cRi < H; ++cRi)
				  for(cCi = 0; cCi < W; ++cCi)
					   out[image_caffe_offset(cB, cNi, cRi, cCi, B, N, H, W)] = in[image_swdnn_offset(cB, cNi, cRi, cCi, B, N, H, W)];
	   return;
	}
	Type* sout   = (Type*)malloc(sizeof(Type)*B*N*H*W);
	if(sout == NULL)
	{
		printf("image_swdnn_to_caffe malloc failure!\n");
		return;
	}
	swapBN_HW(in,sout,H,W,N,B);
	swapBN(sout,out,N,B,H,W);
	
	free(sout);
}

