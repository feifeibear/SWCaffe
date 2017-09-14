#ifndef CAFFE_IMAGENET_DATA_LAYER_HPP_
#define CAFFE_IMAGENET_DATA_LAYER_HPP_

#include <vector>
#include "caffe/layers/data_layer.hpp"

namespace caffe {

template <typename Dtype>
class IMAGENETDataLayer : public DataLayer<Dtype> {
 public:
  explicit IMAGENETDataLayer(const LayerParameter& param);
  virtual ~IMAGENETDataLayer();
  void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top); 
  // DataLayer uses DataReader instead for sharing for parallelism
  virtual inline bool ShareInParallel() const { return false; }
  virtual inline const char* type() const { return "IMAGENETDataLayer"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 3; }

  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}

  void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

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

  Dtype* mean;
};

}  // namespace caffe

#endif  // CAFFE_DATA_LAYER_HPP_
