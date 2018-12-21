//#define HUGE
#include <stdexcept>
#include <cassert>
#include <cstring>
#include <iostream>
#include <vector>
#include <mpi.h>
#include <math.h>
#include <fcntl.h>
#include <unistd.h>
#include <limits.h>
#include <sys/mman.h>
#include <cstdlib>

#include "caffe/collectives.h"
//#include "timer.h"

/***
 * Reduce-scatter and Allgather for number of nodes being power of 2
 *
 */
void TreeAllreduce(float* data, size_t length, bool istimer) {
  //timer::Timer timer, comm_timer, reduce_timer;
  //double global_time = 0.0, comm_time = 0.0, reduce_time = 0.0;
  //timer.start();

  // Get MPI size and rank.
  //MPI_Barrier(MPI_COMM_WORLD);
  int rank;
  int mpi_error = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if(mpi_error != MPI_SUCCESS)
      throw std::runtime_error("MPI_Comm_rank failed with an error");

  int contextSize;
  mpi_error = MPI_Comm_size(MPI_COMM_WORLD, &contextSize);
  if(mpi_error != MPI_SUCCESS)
      throw std::runtime_error("MPI_Comm_size failed with an error");
  int steps = log2(contextSize);
  int chunks = 1 << steps;
  int chunkSize = (length + chunks - 1) / chunks;
  int chunkBytes = chunkSize * sizeof(float);

  size_t bitmask = 1;
	size_t stepChunkSize = chunkSize << (steps - 1);
	size_t sendOffset = 0;
	size_t recvOffset = 0;
	size_t bufferOffset = 0; // offset into recvBuf_
  std::vector<size_t> sendOffsets(steps, 0);
  std::vector<size_t> recvOffsets(steps, 0);
  std::vector<size_t> sendCounts(steps, 0);
  std::vector<size_t> recvCounts(steps, 0);
  float* buffer = (float*)malloc((length/2 + 1) * sizeof(float));
  if(!buffer)
    throw std::runtime_error("allocate buffer error");

  MPI_Status recv_status;
  std::vector<int> destBuff(steps, 0);
	for(int i = 0; i < steps; ++i) {
    const int destRank = rank ^ bitmask;
    destBuff[i] = destRank;
    sendOffsets[i] = sendOffset + ((destRank & bitmask) ? stepChunkSize : 0);
    recvOffsets[i] = recvOffset + ((rank & bitmask) ? stepChunkSize : 0);
		if (sendOffsets[i] < length) {
			// specifies number of elements to send in each step
		  if (sendOffsets[i] + stepChunkSize > length) {
		  	sendCounts[i] = length - sendOffsets[i];
		  } else {
		  	sendCounts[i] = stepChunkSize;
		  }
		}
		if (recvOffsets[i] < length) {
      // specifies number of elements received in each step
      if (recvOffsets[i] + stepChunkSize > length) {
        recvCounts[i] = length - recvOffsets[i];
      } else {
        recvCounts[i] = stepChunkSize;
      }
    }

    float* segment_send = &(data[sendOffsets[i]]);
    float* segment_update = &(data[recvOffsets[i]]);
    MPI_Sendrecv(segment_send, sendCounts[i],
            MPI_FLOAT, destRank, 1, buffer,
            recvCounts[i], MPI_FLOAT, destRank,
            1, MPI_COMM_WORLD, &recv_status);

    reduce(segment_update, buffer, recvCounts[i]);
    if (rank & bitmask) {
      sendOffset += stepChunkSize;
      recvOffset += stepChunkSize;
    }

    bitmask <<= 1;
    stepChunkSize >>= 1;
  }
  for(int i = steps - 1; i >= 0; --i) {
    const int destRank = destBuff[i];
    float* segment_send = &(data[recvOffsets[i]]);
    float* segment_recv = &(data[sendOffsets[i]]);
    MPI_Sendrecv(segment_send, recvCounts[i],
            MPI_FLOAT, destRank, 0, segment_recv,
            sendCounts[i], MPI_FLOAT, destRank,
            0, MPI_COMM_WORLD, &recv_status);
    bitmask >>= 1;
  }



  //debug
//  for(int ii = 0; ii < steps; ++ii) {
//    std::cout << " step " << ii << ", rank: " << rank << ": send : " <<
//      destBuff[ii] << " (" << sendOffsets[ii] << ", "
//      << sendOffsets[ii] + sendCounts[ii] << "); recv : ("
//      << recvOffsets[ii] << ", "
//      << recvOffsets[ii] + recvCounts[ii] << ")" << std::endl;
//    MPI_Barrier(MPI_COMM_WORLD);
//  }
  free(buffer);
}

