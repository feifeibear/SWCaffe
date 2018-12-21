#include "caffe/collectives.h"

// Copy data from one memory buffer to another on CPU or GPU.
// Both buffers must resize on the same device.
void copy(float* dst, float* src, size_t size) {
  // CPU memory allocation through standard allocator.
  for(int i = 0; i < size; i++) {
    dst[i]=src[i];
  }
}

// Copy data from one memory buffer to another on CPU or GPU.
// Both buffers must resize on the same device.
void reduce(float* dst, float* src, size_t size) {
  sw_add_f_allreduce(src, dst, dst, (int)size);
  // Accumulate values from `src` into `dst` on the CPU.
  //for(size_t i = 0; i < size; i++) {
   //dst[i] += src[i];
  //}
}


