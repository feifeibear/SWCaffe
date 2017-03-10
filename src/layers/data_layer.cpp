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

int reverseInt (int i) 
{
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

template <typename Dtype>
DataLayer<Dtype>::DataLayer(const LayerParameter& param):offset_(0),
	n_rows(0), n_cols(0), number_of_images(0),
	Layer<Dtype>(param)
{
  file.open("../data/train-images-idx3-ubyte",std::ios::binary);
	if(!file.is_open()) DLOG(FATAL) << "MNIST Read failed";
	DLOG(INFO) << "fjrdebug read mnist data OK";
}

template <typename Dtype>
DataLayer<Dtype>::~DataLayer() {
	file.close();
}

template <typename Dtype>
void DataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

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
	//assert(number_of_images%batch_size == 0);

	vector<int> top_shape;
	top_shape.push_back(batch_size);
	top_shape.push_back(1);
	top_shape.push_back(n_rows);
	top_shape.push_back(n_cols);

  top[0]->Reshape(top_shape);

  LOG_IF(INFO, Caffe::root_solver())
      << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
}

INSTANTIATE_CLASS(DataLayer);
REGISTER_LAYER_CLASS(Data);

}  // namespace caffe
