#include "caffe/common.hpp"
#include "caffe/util/mpi.hpp"

#include <execinfo.h>
namespace caffe {

template<>
int caffe_mpi_send<float>(void *buf, int count, int dest, int tag,
                    MPI_Comm comm) {
/*
int j, nptrs;
 void *buffer[100];
   char **strings;
nptrs = backtrace(buffer, 3);
strings = backtrace_symbols(buffer, nptrs);
for (j = 0; j < nptrs; j++)

       printf("%s\n", strings[j]);

   free(strings);
*/
//	LOG(INFO)<<"MPI_SEND "<<buf<<" "<<count<<" "<<dest<<" "<<tag<<" ";
//	int size=1024*1024*1024*1;
//	char * bbuf= new char[size];
//        MPI_Buffer_attach((void*)bbuf,size);
	int ret=MPI_Send(buf, count, MPI_FLOAT, dest, tag,
                    comm);
//	MPI_Buffer_detach((void*)bbuf,&size);
	return ret;
}

template<>
int caffe_mpi_send<double>(void *buf, int count,  int dest, int tag,
                    MPI_Comm comm) {
	return MPI_Send(buf, count, MPI_DOUBLE, dest, tag,
                    comm);
}

int caffe_mpi_send(void *buf, int count, MPI_Datatype datatype, int dest, int tag,
                    MPI_Comm comm) {
	return MPI_Send(buf, count, datatype, dest, tag,
                    comm);
}
template<>
int caffe_mpi_recv<float>(void *buf, int count, int dest, int tag,
                    MPI_Comm comm, MPI_Status *status) {
	//LOG(INFO)<<"MPI_RECV "<<buf<<" "<<count<<" "<<dest<<" "<<tag<<" ";
	int ret=MPI_Recv(buf, count, MPI_FLOAT, dest, tag,
                    comm, status);
	return ret;
}

template<>
int caffe_mpi_recv<double>(void *buf, int count,  int dest, int tag,
                    MPI_Comm comm, MPI_Status *status) {
	return MPI_Recv(buf, count, MPI_DOUBLE, dest, tag,
                    comm, status);
}

int caffe_mpi_recv(void *buf, int count, MPI_Datatype datatype, int dest, int tag,
                    MPI_Comm comm, MPI_Status *status) {
	return MPI_Recv(buf, count, datatype, dest, tag,
                    comm, status);
}

template <>
int caffe_mpi_isend<float>(void *buf, int count, int dest, int tag,
                    MPI_Comm comm, MPI_Request *req) {
	return MPI_Isend(buf, count, MPI_FLOAT, dest, tag,comm, req);
}

template <>
int caffe_mpi_isend<double>(void *buf, int count, int dest, int tag,
                    MPI_Comm comm, MPI_Request *req) {
	return MPI_Isend(buf, count, MPI_DOUBLE, dest, tag,comm, req);
}

int caffe_mpi_isend(void *buf, int count, MPI_Datatype datatype, int dest, int tag,
                    MPI_Comm comm, MPI_Request *req) {
	return MPI_Isend(buf, count, datatype, dest, tag,comm, req);
}
template <>
int caffe_mpi_ssend<float>(void *buf, int count, int dest, int tag,
                    MPI_Comm comm) {
	return MPI_Ssend(buf, count, MPI_FLOAT, dest, tag,comm);
}

template <>
int caffe_mpi_ssend<double>(void *buf, int count, int dest, int tag,
                    MPI_Comm comm) {
	return MPI_Ssend(buf, count, MPI_DOUBLE, dest, tag,comm);
}

int caffe_mpi_ssend(void *buf, int count, MPI_Datatype datatype, int dest, int tag,
                    MPI_Comm comm) {
	return MPI_Ssend(buf, count, datatype, dest, tag,comm);
}


template <>
int caffe_mpi_allreduce<float>( void *sendbuf, void *recvbuf, int count,
    MPI_Op op, MPI_Comm comm  ){
  return MPI_Allreduce(sendbuf, recvbuf, count, MPI_FLOAT, op, comm);
}

template <>
int caffe_mpi_allreduce<double>( void *sendbuf, void *recvbuf, int count,
    MPI_Op op, MPI_Comm comm  ){
  return MPI_Allreduce(sendbuf, recvbuf, count, MPI_DOUBLE, op, comm);
}

template <>
int caffe_mpi_reduce<float>( void *sendbuf, void *recvbuf, int count,
    MPI_Op op, int root, MPI_Comm comm  ){
  return MPI_Reduce(sendbuf, recvbuf, count, MPI_FLOAT, op, root, comm);
}

template <>
int caffe_mpi_reduce<double>( void *sendbuf, void *recvbuf, int count,
    MPI_Op op, int root, MPI_Comm comm  ){
  return MPI_Reduce(sendbuf, recvbuf, count, MPI_DOUBLE, op, root, comm);
}

template <>
int caffe_mpi_ireduce<float>( void *sendbuf, void *recvbuf, int count,
    MPI_Op op, int root, MPI_Comm comm, MPI_Request *req ){
  return MPI_Ireduce(sendbuf, recvbuf, count, MPI_FLOAT, op, root, comm, req);
}

template <>
int caffe_mpi_ireduce<double>( void *sendbuf, void *recvbuf, int count,
    MPI_Op op, int root, MPI_Comm comm, MPI_Request *req  ){
  return MPI_Ireduce(sendbuf, recvbuf, count, MPI_DOUBLE, op, root, comm, req);
}

template <>
int caffe_mpi_bcast<float>( void *buffer, int count, int root,
    MPI_Comm comm ) {
  return MPI_Bcast(buffer, count, MPI_FLOAT, root, comm);
}

template <>
int caffe_mpi_bcast<double>( void *buffer, int count, int root,
    MPI_Comm comm ) {
  return MPI_Bcast(buffer, count, MPI_DOUBLE, root, comm);
}

}  // namespace caffe
