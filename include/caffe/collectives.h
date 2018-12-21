#ifndef BAIDU_ALLREDUCE_COLLECTIVES_H_
#define BAIDU_ALLREDUCE_COLLECTIVES_H_

#include <cstddef>
#include <mpi.h>

#define NO_DEVICE -1

/*
 * This file contains the implementation of the baidu-allreduce communication
 * collectives, and provides the following functions:
 *
 *    void InitCollectives(int device);
 *    void RingAllreduce(float* data, size_t length, float** output);
 *    void RingAllgather(float* data, size_t length, float** output);
 *
 */

// Initialize the library, including MPI and if necessary the CUDA device.
// If device == -1, no GPU is used; otherwise, the device specifies which CUDA
// device should be used. All data passed to other functions must be on that device.
void InitCollectives(int device);
void Report(void) ;
void ResetCounters(void) ;

// The ring allreduce. The lengths of the data chunks passed to this function
// must be the same across all MPI processes. The output memory will be
// allocated and written into `output`.
void RingAllreduce(float* data, size_t length, bool istimer);
void TreeAllreduce(float* data, size_t length, bool istimer);

// The ring allgather. The lengths of the data chunks passed to this function
// may differ across different devices. The output memory will be allocated and
// written into `output`.
void RingAllgather(float* data, size_t length, float** output);

// custom allocator; huge pages
float* alloc(size_t size) ;
void dealloc(float* buffer);
// Internal buffers
void Alloc(size_t blen);
void Dealloc(void);
extern "C" {
void sw_add_f_allreduce(float* src1,float *src2, float* dst,const int count);
void thread_init();
}

// Copy data from one memory buffer to another on CPU or GPU.
// Both buffers must resize on the same device.
void copy(float* dst, float* src, size_t size);


// Copy data from one memory buffer to another on CPU or GPU.
// Both buffers must resize on the same device.
void reduce(float* dst, float* src, size_t size);


#endif /* ifndef BAIDU_ALLREDUCE_COLLECTIVES_H_ */
