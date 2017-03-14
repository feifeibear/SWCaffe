#ifndef CAFFE_DATA_LAYER_HPP_
#define CAFFE_DATA_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
//#include "caffe/data_transformer.hpp"
//#include "caffe/internal_thread.hpp"
#include <iostream>
#include <fstream>
#include "caffe/layer.hpp"
//#include "caffe/layers/base_data_layer.hpp"
//#include "caffe/proto/caffe.pb.h"
//#include "caffe/util/db.hpp"

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
  //virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
  //    const vector<Blob<Dtype>*>& top);
  void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top); 
  // DataLayer uses DataReader instead for sharing for parallelism
  virtual inline bool ShareInParallel() const { return false; }
  virtual inline const char* type() const { return "Data"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }

  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top){
    //if(offset_ > number_of_images)
    //  offset_ = 0;
    for( int img = 0; img < batch_size; ++img ){
      for( int c = 0; c < n_cols; ++c )
        for( int r = 0; r < n_rows; ++r ) {
          unsigned char temp;
          file.read((char*) &temp, sizeof(temp));
          int temp_offset = img * n_rows * n_cols + c * n_cols + r;
          top[0]->mutable_cpu_data()[temp_offset] = 1.0/255 * static_cast<Dtype>(temp);
        }

      //read label
      char tmp_label = 0;
      label_file.read(&tmp_label, 1);
      //TODO
      top[1]->mutable_cpu_data()[img*10+ static_cast<int>(tmp_label)] = static_cast<Dtype>(1);

      offset_++;
      if( offset_ > number_of_images ) {
        offset_ = 0;
        file.seekg( 0, std::ios_base::beg );
        label_file.seekg( 0, std::ios_base::beg );
      }
    }
    //top[1]->fjr_print_data();
    //top[0]->fjr_print_data();
  }

 protected:

  int reverseInt (int i) {
      unsigned char c1, c2, c3, c4;
      c1 = i & 255;
      c2 = (i >> 8) & 255;
      c3 = (i >> 16) & 255;
      c4 = (i >> 24) & 255;
      return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
  }
  int n_cols;
  int n_rows;
  int number_of_images;
  int batch_size;
  std::ifstream file;
  std::ifstream label_file;
  //void Next();
  //bool Skip();
  //virtual void load_batch(Batch<Dtype>* batch);

  //shared_ptr<db::DB> db_;
  //shared_ptr<db::Cursor> cursor_;
  uint64_t offset_;
};

}  // namespace caffe

#endif  // CAFFE_DATA_LAYER_HPP_
