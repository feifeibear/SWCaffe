#ifndef CAFFE_DATA_LAYER_HPP_
#define CAFFE_DATA_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include <iostream>
#include <fstream>
#include "caffe/layer.hpp"

namespace caffe {


template <typename Dtype>
class Batch {
 public:
  Blob<Dtype> data_, label_;
};


template <typename Dtype>
class DataLayer : public Layer<Dtype> {
 public:
  explicit DataLayer(const LayerParameter& param);
  virtual ~DataLayer();
  void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top); 
  // DataLayer uses DataReader instead for sharing for parallelism
  virtual inline bool ShareInParallel() const { return false; }
  virtual inline const char* type() const { return "Data"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 3; }

  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top){

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

 protected:

  int n_cols;
  int n_rows;
  int number_of_images;
  int batch_size;
  std::ifstream file;
  std::ifstream label_file;
  std::ifstream mean_file;

  long offset_;

  bool start;

  vector<int> mean;
};

}  // namespace caffe

#endif  // CAFFE_DATA_LAYER_HPP_
