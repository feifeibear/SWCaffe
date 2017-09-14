#include <stdio.h> 
#include <slave.h>
#include <dma.h>
#include <simd.h>
#include <math.h>
#include "caffe/util/matrix_trans.h"


__thread_local_fix dma_desc dmaget2,dmaput2;
typedef float      Type;
typedef floatv4    SIMDType;

#define SWAPABCD2(in0,in1,in2,in3){\
	SIMDType o0 = simd_vshff(in1,in0,68 );  \
	SIMDType o1 = simd_vshff(in1,in0,238);  \
	SIMDType o2 = simd_vshff(in3,in2,68 ); \
	SIMDType o3 = simd_vshff(in3,in2,238); \
	in0 = simd_vshff(o2,o0,136);  \
	in1 = simd_vshff(o2,o0,221);  \
	in2 = simd_vshff(o3,o1,136);  \
	in3 = simd_vshff(o3,o1,221);\
}
inline void mb()
{
    asm volatile("memb");
}
void swapBN_f(SlaveParam_f *pParam)
{
    const int nMaxBuffSize = 55296;//54KB 
	const int nMaxSize = nMaxBuffSize/sizeof(Type);
	
	int B,N,H,W,nSize;
	int i,j,nMaxThreadsNum,nLeftMaxThreadsNum,nOffset,nBNCount; 
	int nHWCount,nHWLeft,nPutOffset,nGetOffset;
	volatile int getreply=0,putreply=0;	
	int myid = athread_get_id(-1),n,b,nH;
	Type * pBuff;
	
	B = pParam->B;
	N = pParam->N;
	H = pParam->H;
	W = pParam->W;
	nBNCount = pParam->nCount;
	nMaxThreadsNum = pParam->nBNThreadsNum;
	nLeftMaxThreadsNum = pParam->nBNLeftThreadsNum;
	nSize = nH = H*W;
	if(myid >= nMaxThreadsNum) return;	
	// dma_desc dmaget2,dmaput2;
	dma_set_op(&dmaget2, DMA_GET);
	dma_set_mode(&dmaget2, PE_MODE);
	dma_set_reply(&dmaget2, &getreply);
	
	dma_set_op(&dmaput2, DMA_PUT);
	dma_set_mode(&dmaput2, PE_MODE);
	dma_set_reply(&dmaput2, &putreply);	
	
	nBNCount = nBNCount < 1 ? 1:nBNCount;
	pBuff = (Type*)(long)ldm_malloc(nMaxBuffSize);

	if(pBuff == NULL)
	{
		printf("swapBN dm_malloc failure!\n");
		return;
	}
	
	if(nH > nMaxSize)
	{
		nHWCount = 1;
		while(nSize>nMaxSize) 
		{
			nSize = nSize>>1;
            nHWCount = nHWCount<<1;			
		}
		nHWLeft = nH%nSize;	
		//if(myid<1)printf("Size=%d nHWCount=%d nHWLeft=%d\n",nSize,nHWCount,nHWLeft);
        for(i=0;i<nBNCount;i++)
		{   
			nOffset = i*nMaxThreadsNum + myid;
			dma_set_size(&dmaget2, sizeof(Type)*nSize);
			dma_set_size(&dmaput2, sizeof(Type)*nSize);   
			b = nOffset/N;
			n = nOffset%N; 
			
			nGetOffset = nOffset*nH;
			nPutOffset = n*B+b;
			nPutOffset = nPutOffset*nH;
			
			for(j=0;j<nHWCount;j++)
			{
				mb();
				dma(dmaget2,(long)(pParam->pIn+nGetOffset+j*nSize),(long)(pBuff));
				dma_wait(&getreply,1);getreply=0;				
		        mb();
				dma(dmaput2,(long)(pParam->pOut+nPutOffset+j*nSize),(long)(pBuff));			
				dma_wait(&putreply,1);putreply=0;
				
			}
			if(nHWLeft>0)
			{
				dma_set_size(&dmaget2, sizeof(Type)*nHWLeft);
				dma_set_size(&dmaput2, sizeof(Type)*nHWLeft);   
				nGetOffset = nGetOffset+nHWCount*nSize;				
				nPutOffset = nPutOffset+nHWCount*nSize;				
				
				mb();
				dma(dmaget2,(long)(pParam->pIn+nGetOffset),(long)(pBuff));
				dma_wait(&getreply,1);getreply=0;
				
				dma(dmaput2,(long)(pParam->pOut+nPutOffset),(long)(pBuff));			
				dma_wait(&putreply,1);putreply=0;
				mb();
			}
		}	  
		
		//Left data process		
		if(nLeftMaxThreadsNum >0 && myid < nLeftMaxThreadsNum)
		{
			nOffset = nBNCount*nMaxThreadsNum + myid;
			dma_set_size(&dmaget2, sizeof(Type)*nSize);
			dma_set_size(&dmaput2, sizeof(Type)*nSize);   
			b = nOffset/N;
			n = nOffset%N; 
			
			nGetOffset = nOffset*nH;
			nPutOffset = n*B+b;
			nPutOffset = nPutOffset*nH;
			
			for(j=0;j<nHWCount;j++)
			{
				mb();
				dma(dmaget2,(long)(pParam->pIn+nGetOffset+j*nSize),(long)(pBuff));
				dma_wait(&getreply,1);getreply=0;				
		        mb();
				dma(dmaput2,(long)(pParam->pOut+nPutOffset+j*nSize),(long)(pBuff));			
				dma_wait(&putreply,1);putreply=0;
				
			}
			if(nHWLeft>0)
			{
				dma_set_size(&dmaget2, sizeof(Type)*nHWLeft);
				dma_set_size(&dmaput2, sizeof(Type)*nHWLeft);   
				nGetOffset = nGetOffset+nHWCount*nSize;				
				nPutOffset = nPutOffset+nHWCount*nSize;				
				
				mb();
				dma(dmaget2,(long)(pParam->pIn+nGetOffset),(long)(pBuff));
				dma_wait(&getreply,1);getreply=0;
				
				dma(dmaput2,(long)(pParam->pOut+nPutOffset),(long)(pBuff));			
				dma_wait(&putreply,1);putreply=0;
				mb();
			}
		}		
	}
	else
	{
		dma_set_size(&dmaget2, sizeof(Type)*nSize);
		dma_set_size(&dmaput2, sizeof(Type)*nSize);   
        for(i=0;i<nBNCount;i++)
		{   
			nOffset = i*nMaxThreadsNum + myid;
			mb();
			dma(dmaget2,(long)(pParam->pIn+nOffset*nSize),(long)(pBuff));
				
			b = nOffset/N;
			n = nOffset%N;
					
			dma_wait(&getreply,1);getreply=0;
				
		   	nOffset = n*B+b;
			dma(dmaput2,(long)(pParam->pOut+nOffset*nSize),(long)(pBuff));			
			dma_wait(&putreply,1);putreply=0;
			mb();			
		}
	  
		//Left data process		
		if(nLeftMaxThreadsNum >0 && myid < nLeftMaxThreadsNum)
		{
			nOffset = nBNCount*nMaxThreadsNum + myid;
			mb();
			dma(dmaget2,(long)(pParam->pIn+nOffset*nSize),(long)(pBuff));
			
			b = nOffset/N;
			n = nOffset%N;
			dma_wait(&getreply,1);getreply=0;	
			
			nOffset = n*B + b;			
			dma(dmaput2,(long)(pParam->pOut+nOffset*nSize),(long)(pBuff));			
			dma_wait(&putreply,1);putreply=0;	
            mb();			
		}	
	}
	
	ldm_free(pBuff,nMaxBuffSize);
}

void swapNBHW_f(SlaveParam_f *pParam)
{
    Type *pTmp,*pLocalIn,*pLocalOut;

	int B,N,H,W,splitNB,splitHW,nSize,w,h,l;
	int i,nMaxThreadsNum,nLeftMaxThreadsNum,nOffset0,nOffset1,nCount,nIndex; 
	SIMDType va0,va1,va2,va3;
	volatile int getreply=0,putreply=0;	
	int myid = athread_get_id(-1);
	
	B = pParam->B;
	N = pParam->N;
	H = pParam->H;
	W = pParam->W;
	splitNB = pParam->splitNB;
	splitHW = pParam->splitHW;
	nCount = pParam->nCount;
	nMaxThreadsNum = pParam->nNBHWThreadsNum;
	nLeftMaxThreadsNum = pParam->nNBHWLeftThreadsNum;
	nSize = splitNB*splitHW*sizeof(Type);
	//if(myid<1)printf("splitHW=%d splitNB=%d nSize=%d\n",splitHW,splitNB,nSize);
	if(myid >= nMaxThreadsNum) return;	
    //pTmp = (Type*)(long)ldm_malloc(2*sizeof(double));
	pLocalIn = (Type*)(long)ldm_malloc(nSize);
	if(pLocalIn == NULL)
	{
		printf("swapNBHW In ldm_malloc failure!\n");
		return;
	}
	pLocalOut = (Type*)(long)ldm_malloc(nSize);
	if(pLocalOut == NULL)
	{
		printf("swapNBHW Out ldm_malloc failure!\n");
		return;
	}
	int nW = H*W;	
	int nHW = nW - (nW%splitHW);
	int nH = N*B;
	dma_desc dmaget,dmaput;
	dma_set_op(&dmaget, DMA_GET);
	dma_set_mode(&dmaget, PE_MODE);
	dma_set_reply(&dmaget, &getreply);
	dma_set_size(&dmaget, nSize);
	dma_set_bsize(&dmaget, splitHW*sizeof(Type));
	dma_set_stepsize(&dmaget, (nW-splitHW)*sizeof(Type));
	
	dma_set_op(&dmaput, DMA_PUT);
	dma_set_mode(&dmaput, PE_MODE);
	dma_set_reply(&dmaput, &putreply);	
	dma_set_size(&dmaput, nSize);
	dma_set_bsize(&dmaput, splitNB*sizeof(Type));
	dma_set_stepsize(&dmaput, (nH-splitNB)*sizeof(Type));
	
	nCount = nCount < 1 ? 1:nCount;	
	
	for(i=0;i<nCount;i++)
	{   
        nOffset0 = (i*nMaxThreadsNum + myid)*splitNB;
        nOffset1 = nOffset0*nW;
		for(h=0;h<nHW;h+=splitHW)
		{
			mb();
			dma(dmaget,(long)(pParam->pIn+nOffset1+h),(long)(pLocalIn));
			dma_wait(&getreply,1);getreply=0;
			mb();	
			for(w=0;w<splitHW;w+=4)
			{
				for(l=0;l<splitNB;l+=4)
				{
					nIndex = l*splitHW+w;
					simd_load(va0,pLocalIn+nIndex);
					nIndex = nIndex+splitHW;
					simd_load(va1,pLocalIn+nIndex);
					nIndex = nIndex+splitHW;
					simd_load(va2,pLocalIn+nIndex);
					nIndex = nIndex+splitHW;
					simd_load(va3,pLocalIn+nIndex);					
					nIndex = w*splitNB+l;
					
					SWAPABCD2(va0,va1,va2,va3);
					
					simd_store(va0,pLocalOut+nIndex);
					nIndex = nIndex+splitNB;
					simd_store(va1,pLocalOut+nIndex);
					nIndex = nIndex+splitNB;
					simd_store(va2,pLocalOut+nIndex);
					nIndex = nIndex+splitNB;
					simd_store(va3,pLocalOut+nIndex);				
				}
			}
			mb(); 
			dma(dmaput,(long)(pParam->pOut+nOffset0+h*nH),(long)(pLocalOut));			
			dma_wait(&putreply,1);putreply=0;
			mb();
		}	
	}
	
	//Left data process
	if(nLeftMaxThreadsNum >0 && myid < nLeftMaxThreadsNum)
	{
		nOffset0 = (nCount*nMaxThreadsNum + myid)*splitNB;
		nOffset1 = nOffset0 *nW;
		for(h=0;h<nHW;h+=splitHW)
		{
			mb();
			dma(dmaget,(long)(pParam->pIn+nOffset1+h),(long)(pLocalIn));
			dma_wait(&getreply,1);getreply=0;
			mb();
			for(w=0;w<splitHW;w+=4)
			{
				for(l=0;l<splitNB;l+=4)
				{
					nIndex = l*splitHW+w;
					simd_load(va0,pLocalIn+nIndex);
					nIndex = nIndex+splitHW;
					simd_load(va1,pLocalIn+nIndex);
					nIndex = nIndex+splitHW;
					simd_load(va2,pLocalIn+nIndex);
					nIndex = nIndex+splitHW;
					simd_load(va3,pLocalIn+nIndex);					
					nIndex = w*splitNB+l;
					
					SWAPABCD2(va0,va1,va2,va3);
					
					simd_store(va0,pLocalOut+nIndex);
					nIndex = nIndex+splitNB;
					simd_store(va1,pLocalOut+nIndex);
					nIndex = nIndex+splitNB;
					simd_store(va2,pLocalOut+nIndex);
					nIndex = nIndex+splitNB;
					simd_store(va3,pLocalOut+nIndex);			
				}
			}		
			mb();
			dma(dmaput,(long)(pParam->pOut+nOffset0+h*nH),(long)(pLocalOut));			
			dma_wait(&putreply,1);putreply=0;
			mb();
		}
	}	
	ldm_free(pLocalIn,nSize);
	ldm_free(pLocalOut,nSize);
	//ldm_free(pTmp,2*sizeof(double));	
}
void swapNBHW_ROLL_f(SlaveParam_f *pParam)
{
    Type *pTmp,*pLocalIn,*pLocalOut;

	int B,N,H,W,splitNB,splitHW,nSize,w,h,l;
	int i,nMaxThreadsNum,nLeftMaxThreadsNum,nOffset0,nOffset1,nCount,nIndex; 
	SIMDType va0,va1,va2,va3;
	volatile int getreply=0,putreply=0;	
	int myid = athread_get_id(-1);
	
	B = pParam->B;
	N = pParam->N;
	H = pParam->H;
	W = pParam->W;
	splitNB = pParam->splitNB;
	splitHW = pParam->splitHW;
	nCount = pParam->nCount;
	nMaxThreadsNum = pParam->nNBHWThreadsNum;
	nLeftMaxThreadsNum = pParam->nNBHWLeftThreadsNum;
	nSize = splitNB*splitHW*sizeof(Type);
	
	if(myid >= nMaxThreadsNum) return;	
    //pTmp = (Type*)(long)ldm_malloc(2*sizeof(double));
	pLocalIn = (Type*)(long)ldm_malloc(nSize);
	if(pLocalIn == NULL)
	{
		printf("swapNBHW In ldm_malloc failure!\n");
		return;
	}
	pLocalOut = (Type*)(long)ldm_malloc(nSize);
	if(pLocalOut == NULL)
	{
		printf("swwapNBHW Out ldm_malloc failure!\n");
		return;
	}
	int nW = H*W;	
	int nH = N*B;
	int nPad = nW%splitHW;
	int nHW = nW - nPad;
	int nNB = nH - (nH%splitNB);
	int nStart = nHW-splitHW+nPad;
	
	dma_desc dmaget,dmaput;
	dma_set_op(&dmaget, DMA_GET);
	dma_set_mode(&dmaget, PE_MODE);
	dma_set_reply(&dmaget, &getreply);
	dma_set_size(&dmaget, nSize);
	dma_set_bsize(&dmaget, splitHW*sizeof(Type));
	dma_set_stepsize(&dmaget, (nW-splitHW)*sizeof(Type));
	
	dma_set_op(&dmaput, DMA_PUT);
	dma_set_mode(&dmaput, PE_MODE);
	dma_set_reply(&dmaput, &putreply);	
	dma_set_size(&dmaput, nSize);
	dma_set_bsize(&dmaput, splitNB*sizeof(Type));
	dma_set_stepsize(&dmaput, (nH-splitNB)*sizeof(Type));
	
	nCount = nCount < 1 ? 1:nCount;	
	
	for(i=0;i<nCount;i++)
	{   
        nOffset0 = (i*nMaxThreadsNum + myid)*splitNB;
        nOffset1 = nOffset0*nW;
		for(h=0;h<nHW;h+=splitHW)
		{
			mb();
			dma(dmaget,(long)(pParam->pIn+nOffset1+h),(long)(pLocalIn));
			dma_wait(&getreply,1);getreply=0;
			mb();	
			for(w=0;w<splitHW;w+=4)
			{
				for(l=0;l<splitNB;l+=4)
				{
					nIndex = l*splitHW+w;
					simd_load(va0,pLocalIn+nIndex);
					nIndex = nIndex+splitHW;
					simd_load(va1,pLocalIn+nIndex);
					nIndex = nIndex+splitHW;
					simd_load(va2,pLocalIn+nIndex);
					nIndex = nIndex+splitHW;
					simd_load(va3,pLocalIn+nIndex);					
					nIndex = (splitHW-4-w)*splitNB+l;
					
					SWAPABCD2(va0,va1,va2,va3);
				
					simd_store(va3,pLocalOut+nIndex);
					nIndex = nIndex+splitNB;
					simd_store(va2,pLocalOut+nIndex);
					nIndex = nIndex+splitNB;
					simd_store(va1,pLocalOut+nIndex);
					nIndex = nIndex+splitNB;
					simd_store(va0,pLocalOut+nIndex);	
	
				}
			}
			mb(); 
			dma(dmaput,(long)(pParam->pOut+nOffset0+(nStart-h)*nH),(long)(pLocalOut));			
			dma_wait(&putreply,1);putreply=0;
			mb();
		}	
	}
	
	//Left data process
	if(nLeftMaxThreadsNum >0 && myid < nLeftMaxThreadsNum)
	{
		nOffset0 = (nCount*nMaxThreadsNum + myid)*splitNB;
		nOffset1 = nOffset0 *nW;
		for(h=0;h<nHW;h+=splitHW)
		{
			mb();
			dma(dmaget,(long)(pParam->pIn+nOffset1+h),(long)(pLocalIn));
			dma_wait(&getreply,1);getreply=0;
			mb();
			for(w=0;w<splitHW;w+=4)
			{
				for(l=0;l<splitNB;l+=4)
				{
					nIndex = l*splitHW+w;
					simd_load(va0,pLocalIn+nIndex);
					nIndex = nIndex+splitHW;
					simd_load(va1,pLocalIn+nIndex);
					nIndex = nIndex+splitHW;
					simd_load(va2,pLocalIn+nIndex);
					nIndex = nIndex+splitHW;
					simd_load(va3,pLocalIn+nIndex);					
					nIndex = (splitHW-4-w)*splitNB+l;
					
					SWAPABCD2(va0,va1,va2,va3);
					
					simd_store(va0,pLocalOut+nIndex);
					nIndex = nIndex+splitNB;
					simd_store(va1,pLocalOut+nIndex);
					nIndex = nIndex+splitNB;
					simd_store(va2,pLocalOut+nIndex);
					nIndex = nIndex+splitNB;
					simd_store(va3,pLocalOut+nIndex);			
				}
			}		
			mb();
			dma(dmaput,(long)(pParam->pOut+nOffset0+(nStart-h)*nH),(long)(pLocalOut));			
			dma_wait(&putreply,1);putreply=0;
			mb();
		}
	}	
	ldm_free(pLocalIn,nSize);
	ldm_free(pLocalOut,nSize);
	//ldm_free(pTmp,2*sizeof(double));	
}
