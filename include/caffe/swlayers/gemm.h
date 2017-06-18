/*************************************************************************
	> File Name: gemm.h
	> Author: ma6174
	> Mail: ma6174@163.com 
	> Created Time: Mon 29 Aug 2016 01:05:51 PM CST
 ************************************************************************/

#ifndef _GEMM_H_
#define _GEMM_H_

typedef double Type;

typedef struct ConvData_st{
  Type* input; //0
  Type* weight; //8
  Type* output; //16
  //   24,  28,  32,  36, 40,  44,  48, 52, 56 
  int _Ni, _Ri, _Ci, _No, _K, _Ro, _Co, _B, _Costride, _bCo, _pad;
}ConvData;


void gemm(Type* input, Type* weight, Type* output, int M, int Mld, int N, int K, int rid, int cid);

#endif

