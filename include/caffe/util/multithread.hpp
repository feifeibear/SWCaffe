//zwl 4cg
#ifndef CAFFE_UTIL_MULTITHREAD_H_
#define CAFFE_UTIL_MULTITHREAD_H_
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#define NThread 4 

#ifdef SW4CG
extern "C"{
  int sys_m_cgid();
}

namespace caffe {

int caffe_get_cgid(){ return sys_m_cgid();}

}
#endif


#endif
