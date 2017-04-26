#ifndef MATRIX_TRANS_H_
#define MATRIX_TRANS_H_

#include <simd.h>

#define HWSIZE       48
#define BSIZE        64
#define NUM_THREADS 64
#define BUFFS       3072 //(24*1024) 

typedef double Type;
typedef doublev4 SIMDType;

typedef struct _tagSlaveParam
{
	int B,N,H,W,splitNB,splitHW;
	int nCount,nBNThreadsNum,nBNLeftThreadsNum,nNBHWThreadsNum,nNBHWLeftThreadsNum;
	Type *pIn,*pOut;
}SlaveParam;



void MatrixInvert(Type*in,Type*out,int B,int N,int H,int W);

#endif
