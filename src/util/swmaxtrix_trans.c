#include <stdio.h> 
#include <slave.h>
#include <dma.h>
#include <simd.h>
#include "caffe/util/matrix_trans.h"


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

void swapBN(SlaveParam *pParam)
{
	int B,N,H,W,nSize;
	int i,nMaxThreadsNum,nLeftMaxThreadsNum,nOffset,nCount; 
	dma_desc dmaget,dmaput;
    volatile int getreply=0,putreply=0;	
	int myid = athread_get_id(-1),n,b,nH;
	Type * pBuff;
	
	B = pParam->B;
	N = pParam->N;
	H = pParam->H;
	W = pParam->W;
	nCount = pParam->nCount;
	nMaxThreadsNum = pParam->nBNThreadsNum;
	nLeftMaxThreadsNum = pParam->nBNLeftThreadsNum;
	nH = H*W;
	
	nSize = sizeof(Type)*nH;	
	
	if(myid >= nMaxThreadsNum) return;	
	
	pBuff = (Type*)ldm_malloc(nSize);
	if(pBuff == NULL)
	{
		printf("ldm_malloc failure!\n");
		return;
	}
	if(myid<1)printf("pBuff=%d %d\n",pBuff,	nH);
	fflush(stdout);
	dma_set_op(&dmaget, DMA_GET);
    dma_set_mode(&dmaget, PE_MODE);
    dma_set_reply(&dmaget, &getreply);
    dma_set_size(&dmaget, nSize);
	
    dma_set_op(&dmaput, DMA_PUT);
    dma_set_mode(&dmaput, PE_MODE);
    dma_set_reply(&dmaput, &putreply);	
	dma_set_size(&dmaput, nSize);
   
	nCount = nCount < 1 ? 1:nCount;
	
	for(i=0;i<nCount;i++)
    {   
        nOffset = i*nMaxThreadsNum + myid;	
        dma(dmaget,(long)(pParam->pIn+nOffset*nH),(long)(pBuff));
		
		b = nOffset/N;
		n = nOffset%N;
		dma_wait(&getreply,1);getreply=0;	
		nOffset = n*B + b;
		dma(dmaput,(long)(pParam->pOut+nOffset*nH),(long)(pBuff));			
		dma_wait(&putreply,1);putreply=0;	
		
	}while(i<nCount);
	
	//Left data process
	if(nLeftMaxThreadsNum >0 && myid < nLeftMaxThreadsNum)
	{
		nOffset = nCount*NUM_THREADS + myid;
		dma(dmaget,(long)(pParam->pIn+nOffset*nH),(long)(pBuff));
		
		b = nOffset/N;
		n = nOffset%N;
		dma_wait(&getreply,1);getreply=0;	
		nOffset = n*B + b;
		
		dma(dmaput,(long)(pParam->pOut+nOffset*nH),(long)(pBuff));			
		dma_wait(&putreply,1);putreply=0;	
	}	
	ldm_free(pBuff,nSize);
}

void swapNBHW(SlaveParam *pParam)
{
    Type *pTmp,*pLocalIn,*pLocalOut;

	int B,N,H,W,splitNB,splitHW,nSize,w,h,l;
	int i,nMaxThreadsNum,nLeftMaxThreadsNum,nOffset0,nOffset1,nCount,nIndex; 
    SIMDType va0,va1,va2,va3,vb0,vb1,vb2,vb3,vxhigh0,vxlow0,vxhigh1,vxlow1;
	dma_desc dmaget,dmaput;
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
	nSize = splitNB*splitHW;
	
	if(myid >= nMaxThreadsNum) return;	
  //pTmp = (Type*)ldm_malloc(2*sizeof(double));
	pLocalIn = (Type*)ldm_malloc(nSize*sizeof(double));
	if(pLocalIn == NULL)
	{
		printf("ldm_malloc failure!\n");
		return;
	}
	pLocalOut = (Type*)ldm_malloc(nSize*sizeof(double));
	if(pLocalOut == NULL)
	{
		printf("ldm_malloc failure!\n");
		return;
	}
	int nW = H*W;	
	int nH = N*B;
	dma_set_op(&dmaget, DMA_GET);
    dma_set_mode(&dmaget, PE_MODE);
    dma_set_reply(&dmaget, &getreply);
    dma_set_size(&dmaget, nSize*sizeof(Type));
	dma_set_bsize(&dmaget, splitHW*sizeof(Type));
	dma_set_stepsize(&dmaget, (nW-splitHW)*sizeof(Type));
	
    dma_set_op(&dmaput, DMA_PUT);
    dma_set_mode(&dmaput, PE_MODE);
    dma_set_reply(&dmaput, &putreply);	
	dma_set_size(&dmaput, nSize*sizeof(Type));
    dma_set_bsize(&dmaput, splitNB*sizeof(Type));
	dma_set_stepsize(&dmaput, (nH-splitNB)*sizeof(Type));
	
	nCount = nCount < 1 ? 1:nCount;
	int nLeft = splitHW%4;
	
	for(i=0;i<nCount;i++)
    {   
        nOffset0 = (i*nMaxThreadsNum + myid)*splitNB;
        nOffset1 = nOffset0*nW;
		for(h=0;h<nW;h+=splitHW)
		{
			dma(dmaget,(long)(pParam->pIn+nOffset1+h),(long)(pLocalIn));
			dma_wait(&getreply,1);getreply=0;
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
           
			dma(dmaput,(long)(pParam->pOut+nOffset0+h*nH),(long)(pLocalOut));			
			dma_wait(&putreply,1);putreply=0;
		}		
	}while(i<nCount);
	
	//Left data process
	if(nLeftMaxThreadsNum >0 && myid < nLeftMaxThreadsNum)
	{
		nOffset0 = (nCount*NUM_THREADS + myid)*splitNB;
		nOffset1 = nOffset0 *nW;
		for(h=0;h<nW;h+=splitHW)
		{
			dma(dmaget,(long)(pParam->pIn+nOffset1+h),(long)(pLocalIn));
			dma_wait(&getreply,1);getreply=0;
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
			dma(dmaput,(long)(pParam->pOut+nOffset0+h*nH),(long)(pLocalOut));			
			dma_wait(&putreply,1);putreply=0;
		}
	}	
	ldm_free(pLocalIn,nSize);
	ldm_free(pLocalOut,nSize);
  //ldm_free(pTmp,2*sizeof(double));	
}


