#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>
//#include <assert.h>

#include <vector>

#include "caffe/layers/imagenet_data_layer.hpp"

namespace caffe {

template <typename Dtype>
IMAGENETDataLayer<Dtype>::IMAGENETDataLayer(const LayerParameter& param):offset_(0),
	n_rows(0), n_cols(0), number_of_images(0),
	DataLayer<Dtype>(param)
{
  file.open(this->layer_param_.data_param().data_source().c_str(), std::ios::in | std::ios::binary);
	label_file.open(this->layer_param_.data_param().label_source().c_str(), std::ios::in | std::ios::binary );
  mean_file.open(this->layer_param_.data_param().mean_source().c_str(), std::ios::in | std::ios::binary);
	if(!file.is_open() || !label_file.is_open())
		DLOG(FATAL) << "Imagenet Read failed";
	DLOG(INFO) << "read Imagenet data OK";
  start = true;
}

template <typename Dtype>
IMAGENETDataLayer<Dtype>::~IMAGENETDataLayer() {
	file.close();
	label_file.close();
}

template <typename Dtype>
void IMAGENETDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  number_of_images=0;
  n_rows=0;
  n_cols=0;
  file.read((char*)&n_rows,sizeof(n_rows));
  DLOG(INFO) << "n_rows is " << n_rows;
  file.read((char*)&n_cols,sizeof(n_cols));
  DLOG(INFO) << "n_cols is " << n_cols;
  file.read((char*)&number_of_images,sizeof(number_of_images));
	DLOG(INFO) << "number_of_images is " << number_of_images;
  
  batch_size = this->layer_param_.data_param().batch_size();
	DLOG(INFO) << "number_of_images is " << number_of_images << " batch_size is " <<
		batch_size;

	int num_labels = 0;
  label_file.read((char*)&num_labels, 4);
  CHECK_EQ(num_labels, number_of_images);

	vector<int> top_shape;

	top_shape.push_back(batch_size);
	top_shape.push_back(3);
	top_shape.push_back(n_rows);
	top_shape.push_back(n_cols);
  top[0]->Reshape(top_shape);
	top_shape.clear();
	top_shape.push_back(batch_size);
	top_shape.push_back(1);
	top[1]->Reshape(top_shape);

  LOG_IF(INFO, Caffe::root_solver())
      << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();

  mean = new Dtype[3*n_rows*n_cols];
  //if (Caffe::root_solver()){
    unsigned char temp;
    for (int ch=0; ch<3; ch++)
      for (int c=0; c<n_cols; c++)
        for (int r=0; r<n_rows; r++){
          mean_file.read((char*)&temp, sizeof(temp));
          mean[ch*n_cols*n_rows+c*n_rows+r] = (Dtype)temp;
        }
  //}
  //if(this->layer_param_.phase() == 1) {   // test phase, necessary ?
  //  DLOG(INFO) << "bcast Begin";
  //  caffe_mpi_bcast<Dtype>(mean, 3*n_rows*n_cols, 0, MPI_COMM_WORLD);
  //  MPI_Barrier(MPI_COMM_WORLD);
  //  DLOG(INFO) << "bcast OK";
  //}
}

static double comm_lapes = 0.0;
static int times = 0;
template <typename Dtype>
void IMAGENETDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top){
  if(this->layer_param_.phase() == 1) {   // test phase, necessary ?
    unsigned char temp;
    for( int img = 0; img < batch_size; ++img ){
      for (int ch = 0; ch < 3; ch++)
        for( int c = 0; c < n_cols; ++c )
          for( int r = 0; r < n_rows; ++r ) {
            file.read((char*) &temp, sizeof(temp));
            int temp_offset = img * n_rows * n_cols * 3 + ch * n_rows * n_cols + c * n_rows + r;
            top[0]->mutable_cpu_data()[temp_offset] = static_cast<Dtype>(temp) - mean[ch*n_cols*n_rows+c*n_rows+r];
          }

      //read label
      char tmp_label = 0;
      label_file.read(&tmp_label, 1);

      top[1]->mutable_cpu_data()[img] = static_cast<Dtype>(tmp_label);

      offset_++;
      if( offset_ >= number_of_images ) {
        offset_ = 0;
        file.clear();
        file.seekg( 0, std::ios_base::beg );

        int dump;
        file.read((char*)&dump,sizeof(dump));
        file.read((char*)&dump,sizeof(dump));
        file.read((char*)&dump,sizeof(dump));
        
        label_file.clear();
        label_file.seekg( 0, std::ios_base::beg );

        label_file.read((char*)&dump,sizeof(dump));
      }
    }
  }
  else {
#ifdef MYMPI
    double begin_time = 0;
    double end_time   = 0;
    begin_time = MPI_Wtime();
#endif
    if(Caffe::root_solver()){
      // read data for rank 0
      unsigned char temp;
      for( int img = 0; img < batch_size; ++img ){
        for (int ch = 0; ch < 3; ch++)
          for( int c = 0; c < n_cols; ++c )
            for( int r = 0; r < n_rows; ++r ) {
              file.read((char*) &temp, sizeof(temp));
              int temp_offset = img * n_rows * n_cols * 3 + ch * n_rows * n_cols + c * n_rows + r;
              top[0]->mutable_cpu_data()[temp_offset] = static_cast<Dtype>(temp) - mean[ch*n_cols*n_rows+c*n_rows+r];
            }

        //read label
        char tmp_label = 0;
        label_file.read(&tmp_label, 1);

        top[1]->mutable_cpu_data()[img] = static_cast<Dtype>(tmp_label);

        offset_++;
        if( offset_ >= number_of_images ) {
          offset_ = 0;
          file.clear();
          file.seekg( 0, std::ios_base::beg );

          int dump;
          file.read((char*)&dump,sizeof(dump));
          file.read((char*)&dump,sizeof(dump));
          file.read((char*)&dump,sizeof(dump));
          
          label_file.clear();
          label_file.seekg( 0, std::ios_base::beg );

          label_file.read((char*)&dump,sizeof(dump));
        }
      }
#ifdef MYMPI
      // read data for remote node
      int send_data_buff_count = batch_size*3*n_cols*n_rows;
      Dtype* send_data_buff = (Dtype*)malloc(sizeof(Dtype)*send_data_buff_count);
      int top1buff_size = batch_size;
      Dtype* top1buff = (Dtype*)malloc(sizeof(Dtype)*top1buff_size);

      for( int proc = 1; proc < Caffe::solver_count(); proc++ ){
        for( int img = 0; img < batch_size; ++img ){
          for (int ch = 0; ch < 3; ch++)
            for( int c = 0; c < n_cols; ++c )
              for( int r = 0; r < n_rows; ++r ) {
                file.read((char*) &temp, sizeof(temp));
                int temp_offset = img * n_rows * n_cols * 3 + ch * n_rows * n_cols + c * n_rows + r;
                send_data_buff[temp_offset] = static_cast<Dtype>(temp) - mean[ch*n_cols*n_rows+c*n_rows+r];
              }



          //read label
          char tmp_label = 0;
          label_file.read(&tmp_label, 1);
          top1buff[img] = static_cast<Dtype>(tmp_label);

          offset_++;
          if( offset_ >= number_of_images ) {
            offset_ = 0;
            file.clear();
            file.seekg( 0, std::ios_base::beg );

            int dump;
            file.read((char*)&dump,sizeof(dump));
            file.read((char*)&dump,sizeof(dump));
            file.read((char*)&dump,sizeof(dump));
            
            label_file.clear();
            label_file.seekg( 0, std::ios_base::beg );

            label_file.read((char*)&dump,sizeof(dump));

          }
        }//img

        caffe_mpi_send<Dtype>(send_data_buff, send_data_buff_count, proc, 0, MPI_COMM_WORLD);
        caffe_mpi_send<Dtype>(top1buff,       top1buff_size,        proc, 0, MPI_COMM_WORLD);
      }//for proc
      MPI_Barrier(MPI_COMM_WORLD);
      free(send_data_buff);
      free(top1buff);
#endif
    }
#ifdef MYMPI
    else {
      int send_data_buff_count = batch_size*3*n_cols*n_rows;
      MPI_Status status;
      caffe_mpi_recv<Dtype>(top[0]->mutable_cpu_data(), send_data_buff_count, 0, 0, MPI_COMM_WORLD, &status);
      int top1buff_size = batch_size;
      caffe_mpi_recv<Dtype>(top[1]->mutable_cpu_data(), top1buff_size,        0, 0, MPI_COMM_WORLD, &status);
      MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    comm_lapes += end_time - begin_time;
    times++;
    if(times == 100) {
      LOG_IF(INFO, Caffe::root_solver()) << " DataLayer Commmunication Time is " << comm_lapes;
      comm_lapes = 0.0;
      times = 0;
    }
#endif
  }
}

INSTANTIATE_CLASS(IMAGENETDataLayer);
REGISTER_LAYER_CLASS(IMAGENETData);

}  // namespace caffe
