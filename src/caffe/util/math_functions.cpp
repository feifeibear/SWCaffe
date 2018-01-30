#include <boost/math/special_functions/next.hpp>
#include <boost/random.hpp>

#include <limits>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#ifdef USE_SWBASE
extern "C" {
#include "caffe/util/sw_dnn.h"
#include "sys/time.h"
}
#endif
#define MIN_SIZE 2048
namespace caffe {

template<>
void caffe_cpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
      ldb, beta, C, N);
}

template<>
void caffe_cpu_gemm<double>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const double alpha, const double* A, const double* B, const double beta,
    double* C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_dgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
      ldb, beta, C, N);
}

template <>
void caffe_cpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const float* A, const float* x,
    const float beta, float* y) {
  cblas_sgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}

template <>
void caffe_cpu_gemv<double>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const double alpha, const double* A, const double* x,
    const double beta, double* y) {
  cblas_dgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}

template <>
void caffe_axpy<float>(const int N, const float alpha, const float* X,
    float* Y) { 
#ifdef USE_SWBASE1
#ifdef DEBUG_SWBASE
      Dtype * p_data = (Dtype*)malloc(N*sizeof(Dtype));
      cblas_saxpy(N, alpha, X, 1, Y, 1); 
#endif
  if(N >MIN_SIZE)
    sw_axpy_f(alpha,X,Y,N);
  else
    cblas_saxpy(N, alpha, X, 1, Y, 1); 
#ifdef DEBUG_SWBASE
      double dSum1=0,dSum2=0;
      int times = 0;
      for(int i=0;i<N;i++){
        if(fabs(Y[i] - p_data[i])>1e-4){
          if(times++ <10) 
          printf(" %lf vs %lf \n",Y[i],p_data[i]);
        }
        dSum1 += Y[i];
        dSum2 += p_data[i];
      }
      printf("axpy dSum1 = %lf dSum2 =%lf\n",dSum1,dSum2);
      free(p_data);
#endif
#else
  cblas_saxpy(N, alpha, X, 1, Y, 1); 
#endif
}

template <>
void caffe_axpy<double>(const int N, const double alpha, const double* X,
    double* Y) { 
#ifdef USE_SWBASE1
  if(N >MIN_SIZE)
    sw_axpy_d(alpha,X,Y,N);
  else
    cblas_daxpy(N, alpha, X, 1, Y, 1); 
#else
  cblas_daxpy(N, alpha, X, 1, Y, 1); 
#endif
}

template <typename Dtype>
void caffe_set(const int N, const Dtype alpha, Dtype* Y) {
#ifdef USE_SWBASE1 //NOT USE ATHREAD !!!
    if(N > MIN_SIZE)
    {
       if(typeid(Dtype) == typeid(double))
         sw_memset_d((double*)Y,(const double)alpha,N);
       else if(typeid(Dtype) == typeid(float)){
         sw_memset_f((float*)Y,(const float)alpha,N);
       }
       else if(typeid(Dtype) == typeid(int)){  
         sw_memset_i((int*)Y,(const int)alpha,N);
       }
       else
       {
         if (alpha == 0) {
           memset(Y, 0, sizeof(Dtype) * N);  // NOLINT(caffe/alt_fn)
           return;
         }
         for (int i = 0; i < N; ++i) {
           Y[i] = alpha;
         }
       }
    }
    else{
      if (alpha == 0) {
        memset(Y, 0, sizeof(Dtype) * N);  // NOLINT(caffe/alt_fn)
        return;
      }
      for (int i = 0; i < N; ++i) {
        Y[i] = alpha;
      }
    }
#ifdef DEBUG_SWBASE
      Dtype * p_data = (Dtype*)malloc(N*sizeof(Dtype));
      if(alpha == 0) 
        memset(p_data, 0, sizeof(Dtype) * N);  // NOLINT(caffe/alt_fn)
      else {
        for (int i = 0; i < N; ++i) {
          p_data[i] = alpha;
        }
      }
      double dSum1=0,dSum2=0;
      int times = 0;
      for(int i=0;i<N;i++){
        if(fabs(Y[i] - p_data[i])>1e-4){
          if(times++ <10) 
          printf(" %lf vs %lf \n",Y[i],p_data[i]);
        }
        dSum1 += Y[i];
        dSum2 += p_data[i];
      }
      printf("memset dSum1 = %lf dSum2 =%lf\n",dSum1,dSum2);
      free(p_data);
#endif
      
#else
  if (alpha == 0) {
    memset(Y, 0, sizeof(Dtype) * N);  // NOLINT(caffe/alt_fn)
    return;
  }
  for (int i = 0; i < N; ++i) {
    Y[i] = alpha;
  }
#endif
}

template void caffe_set<int>(const int N, const int alpha, int* Y);
template void caffe_set<float>(const int N, const float alpha, float* Y);
template void caffe_set<double>(const int N, const double alpha, double* Y);

template <>
void caffe_add_scalar(const int N, const float alpha, float* Y) {
#ifdef USE_SWBASE1
#ifdef DEBUG_SWBASE
      float * p_data = (float*)malloc(N*sizeof(float));
      memcpy(p_data,Y,sizeof(float)*N);
      for (int i = 0; i < N; ++i) {
       p_data[i] += alpha;
      }
#endif
  if(N >MIN_SIZE)
    sw_add_scalar_f(alpha,Y,N);
  else{
    for (int i = 0; i < N; ++i) {
      Y[i] += alpha;
    }
  }
#ifdef DEBUG_SWBASE
      double dSum1=0,dSum2=0;
      int times = 0;
      for(int i=0;i<N;i++){
        if(fabs(Y[i] - p_data[i])>1e-4){
          if(times++ <10) 
          printf(" %lf vs %lf \n",Y[i],p_data[i]);
        }
        dSum1 += Y[i];
        dSum2 += p_data[i];
      }
      printf("add scalar dSum1 = %lf dSum2 =%lf\n",dSum1,dSum2);
      free(p_data);
#endif
#else
  for (int i = 0; i < N; ++i) {
    Y[i] += alpha;
  }
#endif
}

template <>
void caffe_add_scalar(const int N, const double alpha, double* Y) {
#ifdef USE_SWBASE1
  if(N >MIN_SIZE)
    sw_add_scalar_d(alpha,Y,N);
  else{
    for (int i = 0; i < N; ++i) {
      Y[i] += alpha;
    }
  }
#else
  for (int i = 0; i < N; ++i) {
    Y[i] += alpha;
  }
#endif
}

template <typename Dtype>
void caffe_copy(const int N, const Dtype* X, Dtype* Y) {
  if (X != Y) {
    if (Caffe::mode() == Caffe::GPU) {
#ifndef CPU_ONLY
      // NOLINT_NEXT_LINE(caffe/alt_fn)
      CUDA_CHECK(cudaMemcpy(Y, X, sizeof(Dtype) * N, cudaMemcpyDefault));
#else
      NO_GPU;
#endif
    } else {
#ifdef USE_SWBASE
#ifdef DEBUG_SWBASE
      Dtype * p_data = (Dtype*)malloc(N*sizeof(Dtype));
      memcpy(p_data,X, sizeof(Dtype) * N); 
#endif
    if(N > MIN_SIZE)
    {
       if(typeid(Dtype) == typeid(double))
         sw_memcpy_d((double*)X,(double*)Y,N);
       else if(typeid(Dtype) == typeid(float))
         sw_memcpy_f((float*)X,(float*)Y,N);
       else if(typeid(Dtype) == typeid(int))  
         sw_memcpy_i((int*)X,(int*)Y,N);
       else if(typeid(unsigned int) == typeid(unsigned int))  
         sw_memcpy_ui((unsigned int*)X,(unsigned int*)Y,N);
       else
          memcpy(Y, X, sizeof(Dtype) * N);  // NOLINT(caffe/alt_fn)
    }
    else
      memcpy(Y, X, sizeof(Dtype) * N);  // NOLINT(caffe/alt_fn)
#ifdef DEBUG_SWBASE
      double dSum1=0,dSum2=0;
      int times = 0;
      for(int i=0;i<N;i++){
        if(fabs(Y[i] - p_data[i])>1e-4){
          if(times++ <10) 
          printf(" %lf vs %lf \n",Y[i],p_data[i]);
        }
        dSum1 += Y[i];
        dSum2 += p_data[i];
      }
      printf("memcpy dSum1 = %lf dSum2 =%lf\n",dSum1,dSum2);
      free(p_data);
#endif
      
#else
      memcpy(Y, X, sizeof(Dtype) * N);  // NOLINT(caffe/alt_fn)
#endif
    }
  }
}

template void caffe_copy<int>(const int N, const int* X, int* Y);
template void caffe_copy<unsigned int>(const int N, const unsigned int* X,
    unsigned int* Y);
template void caffe_copy<float>(const int N, const float* X, float* Y);
template void caffe_copy<double>(const int N, const double* X, double* Y);

template <>
void caffe_scal<float>(const int N, const float alpha, float *X) {
#ifdef USE_SWBASE1
#ifdef DEBUG_SWBASE
      Dtype * p_data = (Dtype*)malloc(N*sizeof(Dtype));
      memcpy(p_data,X,sizeof(float)*N);
      cblas_sscal(N, alpha, p_data, 1);
#endif
    if(N > MIN_SIZE)
      sw_sscal_f(X,alpha, X,N);
    else
      cblas_sscal(N, alpha, X, 1);
#ifdef DEBUG_SWBASE
      double dSum1=0,dSum2=0;
      int times = 0;
      for(int i=0;i<N;i++){
        if(fabs(X[i] - p_data[i])>1e-4){
          if(times++ <10) 
          printf(" %lf vs %lf \n",X[i],p_data[i]);
        }
        dSum1 += X[i];
        dSum2 += p_data[i];
      }
      printf("axpby dSum1 = %lf dSum2 =%lf\n",dSum1,dSum2);
      free(p_data);
#endif
      
#else
    cblas_sscal(N, alpha, X, 1);
#endif
}

template <>
void caffe_scal<double>(const int N, const double alpha, double *X) {
#ifdef USE_SWBASE1
    if(N > MIN_SIZE)
      sw_sscal_d(X,alpha, X,N);
    else
      cblas_dscal(N, alpha, X, 1);
      
#else
    cblas_dscal(N, alpha, X, 1);
#endif
}
 
template <>
void caffe_cpu_axpby<float>(const int N, const float alpha, const float* X,
                            const float beta, float* Y) {
#ifdef USE_SWBASE1
#ifdef DEBUG_SWBASE
      Dtype * p_data = (Dtype*)malloc(N*sizeof(Dtype));
      cblas_saxpby(N, alpha, X, 1, beta, p_data, 1);
#endif
    if(N > MIN_SIZE)
      sw_axpby_f(alpha, X,beta, Y,N);
    else
      cblas_saxpby(N, alpha, X, 1, beta, Y, 1);
#ifdef DEBUG_SWBASE
      double dSum1=0,dSum2=0;
      int times = 0;
      for(int i=0;i<N;i++){
        if(fabs(Y[i] - p_data[i])>1e-4){
          if(times++ <10) 
          printf(" %lf vs %lf \n",Y[i],p_data[i]);
        }
        dSum1 += Y[i];
        dSum2 += p_data[i];
      }
      printf("axpby dSum1 = %lf dSum2 =%lf\n",dSum1,dSum2);
      free(p_data);
#endif
      
#else
    cblas_saxpby(N, alpha, X, 1, beta, Y, 1);
#endif
}

template <>
void caffe_cpu_axpby<double>(const int N, const double alpha, const double* X,
                             const double beta, double* Y) {
#ifdef USE_SWBASE1
    if(N > MIN_SIZE)
      sw_axpby_d(alpha, X, beta, Y,N);
    else
      cblas_daxpby(N, alpha, X, 1, beta, Y, 1);
      
#else
    cblas_daxpby(N, alpha, X, 1, beta, Y, 1);
#endif
}

template <>
void caffe_add<float>(const int n, const float* a, const float* b,
    float* y) {
#ifdef USE_SWBASE1
#ifdef DEBUG_SWBASE
      Dtype * p_data = (Dtype*)malloc(n*sizeof(Dtype));
      vsAdd(n, a, b, p_data);
#endif
    if(n > MIN_SIZE)
      sw_add_f(a,b,y,n);
    else
      vsAdd(n, a, b, y);
#ifdef DEBUG_SWBASE
      double dSum1=0,dSum2=0;
      int times = 0;
      for(int i=0;i<n;i++){
        if(fabs(y[i] - p_data[i])>1e-4){
          if(times++ <10) 
          printf(" %lf vs %lf \n",y[i],p_data[i]);
        }
        dSum1 += y[i];
        dSum2 += p_data[i];
      }
      printf("add dSum1 = %lf dSum2 =%lf\n",dSum1,dSum2);
      free(p_data);
#endif
      
#else
    vsAdd(n, a, b, y);
#endif
}

template <>
void caffe_add<double>(const int n, const double* a, const double* b,
    double* y) {
#ifdef USE_SWBASE1
    if(n > MIN_SIZE)
      sw_add_d(a,b,y,n);
    else
      vdAdd(n, a, b, y);
      
#else
    vdAdd(n, a, b, y);
#endif
}

template <>
void caffe_sub<float>(const int n, const float* a, const float* b,
    float* y) {
#ifdef USE_SWBASE1
#ifdef DEBUG_SWBASE
      Dtype * p_data = (Dtype*)malloc(n*sizeof(Dtype));
      vsSub(n, a, b, p_data);
#endif
    if(n > MIN_SIZE)
      sw_sub_f(a,b,y,n);
    else
      vsSub(n, a, b, y);
#ifdef DEBUG_SWBASE
      double dSum1=0,dSum2=0;
      int times = 0;
      for(int i=0;i<n;i++){
        if(fabs(y[i] - p_data[i])>1e-4){
          if(times++ <10) 
          printf(" %lf vs %lf \n",y[i],p_data[i]);
        }
        dSum1 += y[i];
        dSum2 += p_data[i];
      }
      printf("sub dSum1 = %lf dSum2 =%lf\n",dSum1,dSum2);
      free(p_data);
#endif
      
#else
    vsSub(n, a, b, y);
#endif
}

template <>
void caffe_sub<double>(const int n, const double* a, const double* b,
    double* y) {
#ifdef USE_SWBASE1
    if(n > MIN_SIZE)
      sw_sub_d(a,b,y,n);
    else
      vdSub(n, a, b, y);
      
#else
    vdSub(n, a, b, y);
#endif
}

template <>
void caffe_mul<float>(const int n, const float* a, const float* b,
    float* y) {
#ifdef USE_SWBASE
#ifdef DEBUG_SWBASE
      float * p_data = (float*)malloc(n*sizeof(float));
      vsMul(n, a, b, p_data);
#endif
    if(n > MIN_SIZE)
      sw_mul_f(a,b,y,n);
    else
      vsMul(n, a, b, y);
      
#ifdef DEBUG_SWBASE
      double dSum1=0,dSum2=0;
      int times = 0;
      for(int i=0;i<n;i++){
        if(fabs(y[i] - p_data[i])>1e-4)
        {
          if(times++ <10) 
          printf(" %lf vs %lf \n",y[i],p_data[i]);
        }
        dSum1 += y[i];
        dSum2 += p_data[i];
      }
      printf("mul dSum1 = %lf dSum2 =%lf\n",dSum1,dSum2);
      free(p_data);
#endif
#else
    vsMul(n, a, b, y);
#endif
}

template <>
void caffe_mul<double>(const int n, const double* a, const double* b,
    double* y) {
#ifdef USE_SWBASE
    if(n > MIN_SIZE)
      sw_mul_d(a,b,y,n);
    else
      vdMul(n, a, b, y);
      
#else
    vdMul(n, a, b, y);
#endif
}

template <>
void caffe_div<float>(const int n, const float* a, const float* b,
    float* y) {
#ifdef USE_SWBASE
#ifdef DEBUG_SWBASE
      float * p_data = (float*)malloc(n*sizeof(float));
      vsDiv(n, a, b, p_data);
#endif
    if(n > MIN_SIZE)
      sw_div_f(a,b,y,n);
    else
      vsDiv(n, a, b, y);
      
#ifdef DEBUG_SWBASE
      double dSum1=0,dSum2=0;
      int times = 0;
      for(int i=0;i<n;i++){
        if(fabs(y[i] - p_data[i])>1e-4){
          if(times++ <10) 
          printf(" %lf vs %lf \n",y[i],p_data[i]);
        }
        dSum1 += y[i];
        dSum2 += p_data[i];
      }
      printf("div dSum1 = %lf dSum2 =%lf\n",dSum1,dSum2);
      free(p_data);
#endif
#else
    vsDiv(n, a, b, y);
#endif
}

template <>
void caffe_div<double>(const int n, const double* a, const double* b,
    double* y) {
#ifdef USE_SWBASE
    if(n > MIN_SIZE)
      sw_div_d(a,b,y,n);
    else
      vdDiv(n, a, b, y);
      
#else
    vdDiv(n, a, b, y);
#endif
}

template <>
void caffe_powx<float>(const int n, const float* a, const float b,
    float* y) {
#ifdef USE_SWBASE1
#ifdef DEBUG_SWBASE
      float * p_data = (float*)malloc(n*sizeof(float));
      vsPowx(n, a, b, p_data);
#endif
    if(n > MIN_SIZE)
      sw_pow_f(a,b,y,n);
    else
      vsPowx(n, a, b, y);
      
#ifdef DEBUG_SWBASE
      double dSum1=0,dSum2=0;
      int times = 0;
      for(int i=0;i<n;i++){
        if(fabs(y[i] - p_data[i])>1e-4){
          if(times++ <10) 
          printf(" %lf vs %lf \n",y[i],p_data[i]);
        }
        dSum1 += y[i];
        dSum2 += p_data[i];
      }
      printf("powx dSum1 = %lf dSum2 =%lf\n",dSum1,dSum2);
      free(p_data);
#endif
#else
    vsPowx(n, a, b, y);
#endif
}

template <>
void caffe_powx<double>(const int n, const double* a, const double b,
    double* y) {
#ifdef USE_SWBASE1
    if(n > MIN_SIZE)
      sw_pow_d(a,b,y,n);
    else
      vdPowx(n, a, b, y);
      
#else
    vdPowx(n, a, b, y);
#endif
}

template <>
void caffe_sqr<float>(const int n, const float* a, float* y) {
#ifdef USE_SWBASE
#ifdef DEBUG_SWBASE
      float * p_data = (float*)malloc(n*sizeof(float));
      vsSqr(n, a, p_data);
#endif
    if(n > MIN_SIZE)
      sw_sqr_f(a,y,n);
    else
      vsSqr(n, a, y);
      
#ifdef DEBUG_SWBASE
      double dSum1=0,dSum2=0;
      int times = 0;
      for(int i=0;i<n;i++){
        if(fabs(y[i] - p_data[i])>1e-4){
          if(times++ <10)
          printf(" %lf vs %lf \n",y[i],p_data[i]);
        }
        dSum1 += y[i];
        dSum2 += p_data[i];
      }
      printf("sqr dSum1 = %lf dSum2 =%lf\n",dSum1,dSum2);
      free(p_data);
#endif
#else
    vsSqr(n, a, y);
#endif
}

template <>
void caffe_sqr<double>(const int n, const double* a, double* y) {
#ifdef USE_SWBASE
    if(n > MIN_SIZE)
      sw_sqr_d(a,y,n);
    else
      vdSqr(n, a, y);
      
#else
    vdSqr(n, a, y);
#endif
}

template <>
void caffe_sqrt<float>(const int n, const float* a, float* y) {
#ifdef USE_SWBASE
#ifdef DEBUG_SWBASE
      float * p_data = (float*)malloc(n*sizeof(float));
      vsSqrt(n, a, p_data);
#endif
    if(n > MIN_SIZE){
      sw_sqrt_f(a,y,n);
    }
    else{
      vsSqrt(n, a, y);
    }
#ifdef DEBUG_SWBASE
      double dSum1=0,dSum2=0;
      int times = 0;
      for(int i=0;i<n;i++){
        if(fabs(y[i] - p_data[i])>1e-4){
          if(times++ <10) 
          printf(" %lf vs %lf \n",y[i],p_data[i]);
        }
        dSum1 += y[i];
        dSum2 += p_data[i];
      }
      printf("sqrt dSum1 = %lf dSum2 =%lf\n",dSum1,dSum2);
      free(p_data);
#endif
      
#else
    vsSqrt(n, a, y);
#endif
}

template <>
void caffe_sqrt<double>(const int n, const double* a, double* y) {
#ifdef USE_SWBASE
    if(n > MIN_SIZE)
      sw_sqrt_d(a,y,n);
    else
      vdSqrt(n, a, y);
      
#else
    vdSqrt(n, a, y);
#endif
}

template <>
void caffe_exp<float>(const int n, const float* a, float* y) {
#ifdef USE_SWBASE1
#ifdef DEBUG_SWBASE
      Dtype * p_data = (Dtype*)malloc(n*sizeof(Dtype));
      vsExp(n, a, p_data);
#endif
    if(n > MIN_SIZE)
      sw_exp_f(a,y,n);
    else
      vsExp(n, a, y);
#ifdef DEBUG_SWBASE
      double dSum1=0,dSum2=0;
      int times = 0;
      for(int i=0;i<n;i++){
        if(fabs(y[i] - p_data[i])>1e-4){
          if(times++ <10) 
          printf(" %lf vs %lf \n",y[i],p_data[i]);
        }
        dSum1 += y[i];
        dSum2 += p_data[i];
      }
      printf("exp dSum1 = %lf dSum2 =%lf\n",dSum1,dSum2);
      free(p_data);
#endif
      
#else
    vsExp(n, a, y);
#endif
}

template <>
void caffe_exp<double>(const int n, const double* a, double* y) {
#ifdef USE_SWBASE1
    if(n > MIN_SIZE)
      sw_exp_d(a,y,n);
    else
      vdExp(n, a, y);
      
#else
    vdExp(n, a, y);
#endif
}

template <>
void caffe_log<float>(const int n, const float* a, float* y) {
#ifdef USE_SWBASE1
#ifdef DEBUG_SWBASE
      Dtype * p_data = (Dtype*)malloc(n*sizeof(Dtype));
      vsLn(n, a, p_data);
#endif
    if(n > MIN_SIZE)
      sw_log_f(a,y,n);
    else
      vsLn(n, a, y);
#ifdef DEBUG_SWBASE
      double dSum1=0,dSum2=0;
      int times = 0;
      for(int i=0;i<n;i++){
        if(fabs(y[i] - p_data[i])>1e-4){
          if(times++ <10) 
          printf(" %lf vs %lf \n",y[i],p_data[i]);
        }
        dSum1 += y[i];
        dSum2 += p_data[i];
      }
      printf("log dSum1 = %lf dSum2 =%lf\n",dSum1,dSum2);
      free(p_data);
#endif
      
#else
    vsLn(n, a, y);
#endif
}

template <>
void caffe_log<double>(const int n, const double* a, double* y) {
#ifdef USE_SWBASE1
    if(n > MIN_SIZE)
      sw_log_d(a,y,n);
    else
      vdLn(n, a, y);
      
#else
    vdLn(n, a, y);
#endif
}

template <>
void caffe_abs<float>(const int n, const float* a, float* y) {
#ifdef USE_SWBASE1
#ifdef DEBUG_SWBASE
      Dtype * p_data = (Dtype*)malloc(n*sizeof(Dtype));
      vsLn(n, a, p_data);
      vsAbs(n, a, p_data);
#endif
    if(n > MIN_SIZE)
      sw_abs_f(a,y,n);
    else
      vsAbs(n, a, y);
#ifdef DEBUG_SWBASE
      int times = 0;
      for(int i=0;i<n;i++){
        if(fabs(y[i] - p_data[i])>1e-4){
          if(times++ <10)
          printf(" %lf vs %lf \n",y[i],p_data[i]);
        }
        dSum1 += y[i];
        dSum2 += p_data[i];
      }
      printf("abs dSum1 = %lf dSum2 =%lf\n",dSum1,dSum2);
      free(p_data);
#endif
      
#else
    vsAbs(n, a, y);
#endif
}

template <>
void caffe_abs<double>(const int n, const double* a, double* y) {
#ifdef USE_SWBASE1
    if(n > MIN_SIZE)
      sw_abs_d(a,y,n);
    else
      vdAbs(n, a, y);
      
#else
    vdAbs(n, a, y);
#endif
}

unsigned int caffe_rng_rand() {
  return (*caffe_rng())();
}

template <typename Dtype>
Dtype caffe_nextafter(const Dtype b) {
  return boost::math::nextafter<Dtype>(
      b, std::numeric_limits<Dtype>::max());
}

template
float caffe_nextafter(const float b);

template
double caffe_nextafter(const double b);

template <typename Dtype>
void caffe_rng_uniform(const int n, const Dtype a, const Dtype b, Dtype* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_LE(a, b);
  boost::uniform_real<Dtype> random_distribution(a, caffe_nextafter<Dtype>(b));
  boost::variate_generator<caffe::rng_t*, boost::uniform_real<Dtype> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
}

template
void caffe_rng_uniform<float>(const int n, const float a, const float b,
                              float* r);

template
void caffe_rng_uniform<double>(const int n, const double a, const double b,
                               double* r);

template <typename Dtype>
void caffe_rng_gaussian(const int n, const Dtype a,
                        const Dtype sigma, Dtype* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GT(sigma, 0);
  boost::normal_distribution<Dtype> random_distribution(a, sigma);
  boost::variate_generator<caffe::rng_t*, boost::normal_distribution<Dtype> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
}

template
void caffe_rng_gaussian<float>(const int n, const float mu,
                               const float sigma, float* r);

template
void caffe_rng_gaussian<double>(const int n, const double mu,
                                const double sigma, double* r);

template <typename Dtype>
void caffe_rng_bernoulli(const int n, const Dtype p, int* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GE(p, 0);
  CHECK_LE(p, 1);
  boost::bernoulli_distribution<Dtype> random_distribution(p);
  boost::variate_generator<caffe::rng_t*, boost::bernoulli_distribution<Dtype> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
}

template
void caffe_rng_bernoulli<double>(const int n, const double p, int* r);

template
void caffe_rng_bernoulli<float>(const int n, const float p, int* r);

template <typename Dtype>
void caffe_rng_bernoulli(const int n, const Dtype p, unsigned int* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GE(p, 0);
  CHECK_LE(p, 1);
  boost::bernoulli_distribution<Dtype> random_distribution(p);
  boost::variate_generator<caffe::rng_t*, boost::bernoulli_distribution<Dtype> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = static_cast<unsigned int>(variate_generator());
  }
}

template
void caffe_rng_bernoulli<double>(const int n, const double p, unsigned int* r);

template
void caffe_rng_bernoulli<float>(const int n, const float p, unsigned int* r);

template <>
float caffe_cpu_strided_dot<float>(const int n, const float* x, const int incx,
    const float* y, const int incy) {
  return cblas_sdot(n, x, incx, y, incy);
}

template <>
double caffe_cpu_strided_dot<double>(const int n, const double* x,
    const int incx, const double* y, const int incy) {
  return cblas_ddot(n, x, incx, y, incy);
}

template <typename Dtype>
Dtype caffe_cpu_dot(const int n, const Dtype* x, const Dtype* y) {
  return caffe_cpu_strided_dot(n, x, 1, y, 1);
}

template
float caffe_cpu_dot<float>(const int n, const float* x, const float* y);

template
double caffe_cpu_dot<double>(const int n, const double* x, const double* y);

template <>
float caffe_cpu_asum<float>(const int n, const float* x) {
  return cblas_sasum(n, x, 1);
}

template <>
double caffe_cpu_asum<double>(const int n, const double* x) {
  return cblas_dasum(n, x, 1);
}

template <>
void caffe_cpu_scale<float>(const int n, const float alpha, const float *x,
                            float* y) {
#ifdef USE_SWBASE11
#ifdef DEBUG_SWBASE
      Dtype * p_data = (Dtype*)malloc(n*sizeof(Dtype));
      cblas_scopy(n, x, 1, p_data, 1);
      cblas_sscal(n, alpha, p_data, 1);
#endif
    if(n > MIN_SIZE)
      sw_sscal_f(x,alpha,y,n);
    else{
      cblas_scopy(n, x, 1, y, 1);
      cblas_sscal(n, alpha, y, 1);
    }
#ifdef DEBUG_SWBASE
      int times = 0;
      for(int i=0;i<n;i++){
        if(fabs(y[i] - p_data[i])>1e-4){
          if(times++ <10) 
          printf(" %lf vs %lf \n",y[i],p_data[i]);
        }
        dSum1 += y[i];
        dSum2 += p_data[i];
      }
      printf("scale dSum1 = %lf dSum2 =%lf\n",dSum1,dSum2);
      free(p_data);
#endif
#else
  cblas_scopy(n, x, 1, y, 1);
  cblas_sscal(n, alpha, y, 1);
#endif
}

template <>
void caffe_cpu_scale<double>(const int n, const double alpha, const double *x,
                             double* y) {
#ifdef USE_SWBASE1
    if(n > MIN_SIZE)
      sw_sscal_d(x,alpha,y,n);
    else{
      cblas_dcopy(n, x, 1, y, 1);
      cblas_dscal(n, alpha, y, 1);
    }
#else
  cblas_dcopy(n, x, 1, y, 1);
  cblas_dscal(n, alpha, y, 1);
#endif
}

}  // namespace caffe
