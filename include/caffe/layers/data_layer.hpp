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
    if(offset_ > number_of_images)
      offset_ = 0;
    for( int img = 0; img < batch_size; ++img ){
      for( int r = 0; r < n_rows; ++r )
        for( int c = 0; c < n_cols; ++c ){
          unsigned char temp;
          file.read((char*) &temp, sizeof(temp));
          //top[0]->mutable_cpu_data()[img][0][r][c] = static_cast<Dtype>(temp);
        }

    }
    offset_+=batch_size;

  }

 protected:
  int n_cols;
  int n_rows;
  int number_of_images;
  int batch_size;
  std::ifstream file;
  //void Next();
  //bool Skip();
  //virtual void load_batch(Batch<Dtype>* batch);

  //shared_ptr<db::DB> db_;
  //shared_ptr<db::Cursor> cursor_;
  uint64_t offset_;
};

}  // namespace caffe

#endif  // CAFFE_DATA_LAYER_HPP_
