#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>
//#include <assert.h>

#include <vector>

//#include "caffe/data_transformer.hpp"
#include "caffe/layers/data_layer.hpp"
//#include "caffe/util/benchmark.hpp"

namespace caffe {

template <typename Dtype>
DataLayer<Dtype>::DataLayer(const LayerParameter& param):offset_(0),
	n_rows(0), n_cols(0), number_of_images(0),
	Layer<Dtype>(param)
{
  file.open(this->layer_param_.data_param().data_source().c_str(), std::ios::in | std::ios::binary);
	label_file.open(this->layer_param_.data_param().label_source().c_str(), std::ios::in | std::ios::binary );
	if(!file.is_open() || !label_file.is_open())
		DLOG(FATAL) << "MNIST Read failed";
  start = true;
}

template <typename Dtype>
DataLayer<Dtype>::~DataLayer() {
	file.close();
	label_file.close();
}

template <typename Dtype>
void DataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  //if(Caffe::solver_rank() == 0) {
    int magic_number=0;
    number_of_images=0;
    n_rows=0;
    n_cols=0;
    file.read((char*)&magic_number,sizeof(magic_number));
    magic_number= reverseInt(magic_number);
	  DLOG(INFO) << "magic_number is " << magic_number;
    file.read((char*)&number_of_images,sizeof(number_of_images));
    number_of_images= reverseInt(number_of_images);
	  DLOG(INFO) << "number_of_images is " << number_of_images;
    file.read((char*)&n_rows,sizeof(n_rows));
    n_rows= reverseInt(n_rows);
	  DLOG(INFO) << "n_rows is " << n_rows;
    file.read((char*)&n_cols,sizeof(n_cols));
    n_cols= reverseInt(n_cols);
	  DLOG(INFO) << "n_cols is " << n_cols;

    batch_size = this->layer_param_.data_param().batch_size();
	  DLOG(INFO) << "number_of_images is " << number_of_images << " batch_size is " <<
	  	batch_size;

    label_file.read(reinterpret_cast<char*>(&magic_number), 4);
    magic_number = reverseInt(magic_number);
    CHECK_EQ(magic_number, 2049) << "Incorrect label file magic.";
	  int num_labels = 0;
    label_file.read(reinterpret_cast<char*>(&num_labels), 4);
    num_labels = reverseInt(num_labels);
    CHECK_EQ(num_labels, number_of_images);

  //}
/*
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Bcast(&n_rows,            1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Bcast(&n_cols,            1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Bcast(&number_of_images,  1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Bcast(&batch_size,        1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
*/
	vector<int> top_shape;

#ifdef SEQ_MNIST
    top_shape.push_back(28);
    top_shape.push_back(batch_size);
    top_shape.push_back(28);
    top[0]->Reshape(top_shape);
    top_shape.clear();
    top_shape.push_back(28);
    top_shape.push_back(batch_size);
    top[1]->Reshape(top_shape);
    top_shape.clear();
    top_shape.push_back(batch_size);
    top_shape.push_back(1);
    top[2]->Reshape(top_shape);

#else
	  top_shape.push_back(batch_size);
	  top_shape.push_back(1);
	  top_shape.push_back(n_rows);
	  top_shape.push_back(n_cols);
    top[0]->Reshape(top_shape);
	  top_shape.clear();
	  top_shape.push_back(batch_size);
	  top_shape.push_back(1);
	  top[1]->Reshape(top_shape);

#endif
  LOG_IF(INFO, Caffe::root_solver())
      << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
}

static double comm_lapes = 0.0;
static int times = 0;
template <typename Dtype>
void DataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top){
    if(this->layer_param_.phase() == 1) {
    #ifdef SEQ_MNIST
      if (start){
        start = false;
        for (int i=0; i<n_rows*batch_size; i++)
          top[1]->mutable_cpu_data()[i] = 0;
      }
      else
        for (int i=0; i<n_rows*batch_size; i++)
          top[1]->mutable_cpu_data()[i] = 1;
    #endif

      for( int img = 0; img < batch_size; ++img ){
        for( int c = 0; c < n_cols; ++c )
          for( int r = 0; r < n_rows; ++r ) {
            unsigned char temp;
            file.read((char*) &temp, sizeof(temp));
            #ifdef SEQ_MNIST    // n_rows * batch_size * n_cols
              int temp_offset = r * batch_size * n_cols + img * n_cols + c;
            #else
              int temp_offset = img * n_rows * n_cols + c * n_rows + r;
            #endif
            top[0]->mutable_cpu_data()[temp_offset] = 1.0/255 * static_cast<Dtype>(temp);
          }

        //read label
        char tmp_label = 0;
        label_file.read(&tmp_label, 1);
        #ifdef SEQ_MNIST
          top[2]->mutable_cpu_data()[img] = static_cast<Dtype>(tmp_label);
        #else
          top[1]->mutable_cpu_data()[img] = static_cast<Dtype>(tmp_label);
        #endif

        offset_++;
        if( offset_ >= number_of_images ) {
          offset_ = 0;
          file.clear();
          file.seekg( 0, std::ios_base::beg );

          int dump;
          file.read((char*)&dump,sizeof(dump)); 
          file.read((char*)&dump,sizeof(dump));
          file.read((char*)&dump,sizeof(dump));
          file.read((char*)&dump,sizeof(dump));

          label_file.clear();
          label_file.seekg( 0, std::ios_base::beg );

          label_file.read(reinterpret_cast<char*>(&dump), 4);
          label_file.read(reinterpret_cast<char*>(&dump), 4);

        }
      }//img
    }
    else {

    double begin_time = 0;
    double end_time   = 0;
    begin_time = MPI_Wtime();
    if(Caffe::root_solver()){
        #ifdef SEQ_MNIST
          if (start){
            start = false;
            for (int i=0; i<n_rows*batch_size; i++)
              top[1]->mutable_cpu_data()[i] = 0;
          }
          else
            for (int i=0; i<n_rows*batch_size; i++)
              top[1]->mutable_cpu_data()[i] = 1;
        #endif


        // read data for rank 0
        for( int img = 0; img < batch_size; ++img ){
        for( int c = 0; c < n_cols; ++c )
          for( int r = 0; r < n_rows; ++r ) {
            unsigned char temp;
            file.read((char*) &temp, sizeof(temp));
            #ifdef SEQ_MNIST    // n_rows * batch_size * n_cols
              int temp_offset = r * batch_size * n_cols + img * n_cols + c;
            #else
              int temp_offset = img * n_rows * n_cols + c * n_rows + r;
            #endif
            top[0]->mutable_cpu_data()[temp_offset] = 1.0/255 * static_cast<Dtype>(temp);
          }

        //read label
        char tmp_label = 0;
        label_file.read(&tmp_label, 1);
        #ifdef SEQ_MNIST
          top[2]->mutable_cpu_data()[img] = static_cast<Dtype>(tmp_label);
        #else
          top[1]->mutable_cpu_data()[img] = static_cast<Dtype>(tmp_label);
        #endif

        offset_++;
        if( offset_ >= number_of_images ) {
          offset_ = 0;
          file.clear();
          file.seekg( 0, std::ios_base::beg );

          int dump;
          file.read((char*)&dump,sizeof(dump)); 
          file.read((char*)&dump,sizeof(dump));
          file.read((char*)&dump,sizeof(dump));
          file.read((char*)&dump,sizeof(dump));

          label_file.clear();
          label_file.seekg( 0, std::ios_base::beg );

          label_file.read(reinterpret_cast<char*>(&dump), 4);
          label_file.read(reinterpret_cast<char*>(&dump), 4);

        }
      }
      // read data for remote node
      int send_data_buff_count = batch_size*n_cols*n_rows;
      Dtype* send_data_buff = (Dtype*)malloc(sizeof(Dtype)*send_data_buff_count);
      int top1buff_size = batch_size;
      Dtype* top1buff = (Dtype*)malloc(sizeof(Dtype)*top1buff_size);

      for( int proc = 1; proc < Caffe::solver_count(); proc++ ){
        for( int img = 0; img < batch_size; ++img ){
          for( int c = 0; c < n_cols; ++c )
            for( int r = 0; r < n_rows; ++r ) {
              unsigned char temp;
              file.read((char*) &temp, sizeof(temp));
              #ifdef SEQ_MNIST    // n_rows * batch_size * n_cols
                int temp_offset = r * batch_size * n_cols + img * n_cols + c;
              #else
                int temp_offset = img * n_rows * n_cols + c * n_rows + r;
              #endif
              //top[0]->mutable_cpu_data()[temp_offset] = 1.0/255 * static_cast<Dtype>(temp);
              send_data_buff[temp_offset] = 1.0/255 * static_cast<Dtype>(temp);
            }

          //read label
          char tmp_label = 0;
          label_file.read(&tmp_label, 1);
          #ifdef SEQ_MNIST
            top[2]->mutable_cpu_data()[img] = static_cast<Dtype>(tmp_label);
          #else
            //top[1]->mutable_cpu_data()[img] = static_cast<Dtype>(tmp_label);
            top1buff[img] = static_cast<Dtype>(tmp_label);
          #endif

          offset_++;
          if( offset_ >= number_of_images ) {
            offset_ = 0;
            file.clear();
            file.seekg( 0, std::ios_base::beg );

            int dump;
            file.read((char*)&dump,sizeof(dump)); 
            file.read((char*)&dump,sizeof(dump));
            file.read((char*)&dump,sizeof(dump));
            file.read((char*)&dump,sizeof(dump));

            label_file.clear();
            label_file.seekg( 0, std::ios_base::beg );

            label_file.read(reinterpret_cast<char*>(&dump), 4);
            label_file.read(reinterpret_cast<char*>(&dump), 4);

          }
        }//img

        caffe_mpi_send<Dtype>(send_data_buff, send_data_buff_count, proc, 0, MPI_COMM_WORLD);
        caffe_mpi_send<Dtype>(top1buff,       top1buff_size,        proc, 0, MPI_COMM_WORLD);
      }//for proc
      free(send_data_buff);
      free(top1buff);

    } else {
      int send_data_buff_count = batch_size*n_cols*n_rows;
      int top1buff_size = batch_size;
      MPI_Status status;
      caffe_mpi_recv<Dtype>(top[0]->mutable_cpu_data(), send_data_buff_count, 0, 0, MPI_COMM_WORLD, &status);
      caffe_mpi_recv<Dtype>(top[1]->mutable_cpu_data(), top1buff_size,        0, 0, MPI_COMM_WORLD, &status);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    comm_lapes += end_time - begin_time;
    times++;
    if(times == 100) {
      DLOG_IF(INFO, Caffe::root_solver()) << " DataLayer Commmunication Time is " << comm_lapes;
      comm_lapes = 0.0;
      times = 0;
    }
  }
}

INSTANTIATE_CLASS(DataLayer);
REGISTER_LAYER_CLASS(Data);

}  // namespace caffe
