#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>
//#include <assert.h>

#include <vector>

#include "caffe/layers/data_layer.hpp"

namespace caffe {

template <typename Dtype>
DataLayer<Dtype>::DataLayer(const LayerParameter& param):offset_(0),
	n_rows(0), n_cols(0), number_of_images(0),
	Layer<Dtype>(param)
{
  file.open(this->layer_param_.data_param().data_source().c_str(), std::ios::in | std::ios::binary);
	label_file.open(this->layer_param_.data_param().label_source().c_str(), std::ios::in | std::ios::binary );
  mean_file.open("../data/imagenet/mean.bin", std::ios::in | std::ios::binary);
	if(!file.is_open() || !label_file.is_open())
		DLOG(FATAL) << "Imagenet Read failed";
	DLOG(INFO) << "read Imagenet data OK";
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

  mean.resize(3*n_rows*n_cols);
  unsigned char temp;
  for (int ch=0; ch<3; ch++)
    for (int c=0; c<n_cols; c++)
      for (int r=0; r<n_rows; r++){
        mean_file.read((char*)&temp, sizeof(temp));
        mean[ch*n_cols*n_rows+c*n_rows+r] = (int)temp;
      }
}

INSTANTIATE_CLASS(DataLayer);
REGISTER_LAYER_CLASS(Data);

}  // namespace caffe
