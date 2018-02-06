#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>
//#include <assert.h>

#include "caffe/layers/imagenet_data_layer.hpp"

namespace caffe {

template <typename Dtype>
IMAGENETDataLayer<Dtype>::IMAGENETDataLayer(const LayerParameter& param)
: BaseDataLayer<Dtype>(param), offset_(){
  fetch_.reset(new Batch<Dtype>());
  db_.reset(db::GetDB(param.data_param().backend()));
  db_->Open(param.data_param().source(), db::READ);
  cursor_.reset(db_->NewCursor());
}

template <typename Dtype>
IMAGENETDataLayer<Dtype>::~IMAGENETDataLayer() {}

template <typename Dtype>
void IMAGENETDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  BaseDataLayer<Dtype>::LayerSetUp(bottom, top);
  //call memory alloc
  DLOG(INFO) << "Initializing fetch";
  this->data_transformer_->InitRand();
  fetch_->data_.mutable_cpu_data();
  DLOG(INFO) << "Fetch initialized.";
}

template <typename Dtype>
void IMAGENETDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int batch_size = this->layer_param_.data_param().batch_size();
  // Read a data point, and use it to initialize the top blob.
  Datum datum;
  datum.ParseFromString(cursor_->value());

  // Use data_transformer to infer the expected blob shape from datum.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
  this->transformed_data_.Reshape(top_shape);
  // Reshape top[0] and prefetch_data according to the batch_size.
  top_shape[0] = batch_size;
  top[0]->Reshape(top_shape);
  this->fetch_->data_.Reshape(top_shape);
  LOG_IF(INFO, Caffe::root_solver())
      << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  if (this->output_labels_) {
    vector<int> label_shape(1, batch_size);
    top[1]->Reshape(label_shape);
    this->fetch_->label_.Reshape(label_shape);
  }
#ifdef DATAPREFETCH
  if (this->layer_param_.phase() == TRAIN && !Caffe::mpi_root_solver()){
    nbatch = 0;
    batchidx = 0;
    pre_load_batch(20);
  }
#endif
}

template <typename Dtype>
void IMAGENETDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top){
#ifdef DATAPREFETCH
  if (this->layer_param_.phase() == TRAIN && !Caffe::mpi_root_solver()){
    // Reshape to loaded data.
    top[0]->ReshapeLike(fetch_ptr_[batchidx]->data_);
    top[0]->set_cpu_data(fetch_ptr_[batchidx]->data_.mutable_cpu_data());
    if (this->output_labels_) {
      // Reshape to loaded labels.
      top[1]->ReshapeLike(fetch_ptr_[batchidx]->label_);
      top[1]->set_cpu_data(fetch_ptr_[batchidx]->label_.mutable_cpu_data());
    }
    batchidx ++;
    if(batchidx == nbatch){
      batchidx=0; 
    }
  }else{
    load_batch(fetch_.get());
    // Reshape to loaded data.
    top[0]->ReshapeLike(fetch_->data_);
    top[0]->set_cpu_data(fetch_->data_.mutable_cpu_data());
    if (this->output_labels_) {
      // Reshape to loaded labels.
      top[1]->ReshapeLike(fetch_->label_);
      top[1]->set_cpu_data(fetch_->label_.mutable_cpu_data());
    }
  }
#else
  load_batch(fetch_.get());
  // Reshape to loaded data.
  top[0]->ReshapeLike(fetch_->data_);
  top[0]->set_cpu_data(fetch_->data_.mutable_cpu_data());
  if (this->output_labels_) {
    // Reshape to loaded labels.
    top[1]->ReshapeLike(fetch_->label_);
    top[1]->set_cpu_data(fetch_->label_.mutable_cpu_data());
  }
#endif
}
template <typename Dtype>
bool IMAGENETDataLayer<Dtype>::Skip() {
#ifdef SWMPI
  int size = Caffe::mpi_count()-1;
  int rank = Caffe::mpi_rank()-1;
  bool keep = (offset_ % size) == rank ||
#ifndef SWMPITEST
              // In test mode, only rank 0 runs, so avoid skipping
              this->layer_param_.phase() == TEST ||
#endif
              // For the train iteration after optimization in server
              rank == -1;
  return !keep;

#else
  int size = Caffe::solver_count();
  int rank = Caffe::solver_rank();
  bool keep = (offset_ % size) == rank ||
              // In test mode, only rank 0 runs, so avoid skipping
              this->layer_param_.phase() == TEST;
  return !keep;
#endif
}

template<typename Dtype>
void IMAGENETDataLayer<Dtype>::Next() {
  cursor_->Next();
  offset_++;
  if (!cursor_->valid()) {
    LOG_IF(INFO, Caffe::root_solver())
        << "Restarting data prefetching from start.";
    cursor_->SeekToFirst();
    offset_=0;
  }
}

#ifdef DATAPREFETCH
template<typename Dtype>
void IMAGENETDataLayer<Dtype>::pre_load_batch(int num_batch) {
  DLOG(INFO) <<" Rank "<< Caffe::mpi_rank()  << " Entering pre load batch..";
  CPUTimer batch_timer;
  CPUTimer timer;
  const int batch_size = this->layer_param_.data_param().batch_size();
  CHECK(this->transformed_data_.count());
  CHECK(this->fetch_ptr_.size()==0);

  int size = Caffe::mpi_count()-1;
  int rank = Caffe::mpi_rank()-1;
  if(rank == -1){
    //int total_samples = 0;
    //cursor_->SeekToFirst();
    //while(cursor_->valid()){
      //total_samples ++;
      //cursor_->Next();
    //}
    //DLOG(INFO) <<" MPIRoot : "<< " Total number of samples is "<<total_samples
      //<<". Number of workers is "<<size
      //<<". Number of prefetched batch is "<<total_samples/(batch_size*size)
      //<<". (Max at "<<num_batch<<").";
    return;
  }
  offset_ = 0;
  nbatch = 0;
  cursor_->SeekToFirst();

  while(cursor_->valid() && nbatch<num_batch){
    if((offset_ % batch_size) != 0 || ((offset_/batch_size) % size) != rank){
      cursor_->Next();
      offset_++;
      continue;
    }
    //load a batch
    //--init a new batch
    double read_time = 0;
    double trans_time = 0;
    batch_timer.Start();
    Batch<Dtype>* batch =  new Batch<Dtype>();
    batch->data_.ReshapeLike(this->fetch_->data_);
    batch->data_.mutable_cpu_data();
    if(this->output_labels_){
      batch->label_.ReshapeLike(this->fetch_->label_);
      batch->label_.mutable_cpu_data();
    }
    CHECK(batch->data_.count());
    //-- load a batch 
    Datum datum;
    bool full = true;
    for (int item_id = 0; item_id < batch_size; ++item_id) {
      if(!cursor_->valid()){
        full = false;
        cursor_->SeekToFirst();
        offset_ = 0;
      }

      timer.Start();
      datum.ParseFromString(cursor_->value());
      read_time += timer.MicroSeconds();

      if (item_id == 0) {
        // Reshape according to the first datum of each batch
        // on single input batches allows for inputs of varying dimension.
        // Use data_transformer to infer the expected blob shape from datum.
        vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
        this->transformed_data_.Reshape(top_shape);
        // Reshape batch according to the batch_size.
        top_shape[0] = batch_size;
        batch->data_.Reshape(top_shape);
      }

      // Apply data transformations (mirror, scale, crop...)
      timer.Start();
      int offset = batch->data_.offset(item_id);
      Dtype* top_data = batch->data_.mutable_cpu_data();
      this->transformed_data_.set_cpu_data(top_data + offset);
      this->data_transformer_->Transform(datum, &(this->transformed_data_));
      // Copy label.
      if (this->output_labels_) {
        Dtype* top_label = batch->label_.mutable_cpu_data();
        top_label[item_id] = datum.label();
      }
      trans_time += timer.MicroSeconds();
      cursor_->Next();
      offset_++; 
    }
    if(full){
      timer.Stop();
      batch_timer.Stop();
      //save to fetch_ptr_
      fetch_ptr_.push_back(batch);
      nbatch++;
      DLOG(INFO) <<" Rank "<< Caffe::mpi_rank()  << " : PreFetch batch ("
        << nbatch <<"/" <<num_batch<<" @ "<<offset_ - batch_size << "~"<<offset_
        <<" ): "  << batch_timer.MilliSeconds() << " ms.";
      DLOG(INFO) <<" Rank "<< Caffe::mpi_rank()  << " : Read time: " << read_time / 1000 << " ms.";
      DLOG(INFO) <<" Rank "<< Caffe::mpi_rank()  << " : Transform time: " << trans_time / 1000 << " ms.";
      
    }else{
      timer.Stop();
      batch_timer.Stop();
      break;
    }
  }
  CHECK(nbatch==fetch_ptr_.size());
  DLOG(INFO) <<" Rank "<< Caffe::mpi_rank()  << " : PreFetch batch done! "
    << nbatch << " batches are prefeched.";
  offset_ = 0;
  cursor_->SeekToFirst();
}

#endif

template<typename Dtype>
void IMAGENETDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());
  const int batch_size = this->layer_param_.data_param().batch_size();

  Datum datum;
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    timer.Start();
    while (Skip()) {
      Next();
    }
    datum.ParseFromString(cursor_->value());
    read_time += timer.MicroSeconds();

    if (item_id == 0) {
      // Reshape according to the first datum of each batch
      // on single input batches allows for inputs of varying dimension.
      // Use data_transformer to infer the expected blob shape from datum.
      vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
      this->transformed_data_.Reshape(top_shape);
      // Reshape batch according to the batch_size.
      top_shape[0] = batch_size;
      batch->data_.Reshape(top_shape);
    }

    // Apply data transformations (mirror, scale, crop...)
    timer.Start();
    int offset = batch->data_.offset(item_id);
    Dtype* top_data = batch->data_.mutable_cpu_data();
    this->transformed_data_.set_cpu_data(top_data + offset);
    this->data_transformer_->Transform(datum, &(this->transformed_data_));
    // Copy label.
    if (this->output_labels_) {
      Dtype* top_label = batch->label_.mutable_cpu_data();
      top_label[item_id] = datum.label();
    }
    trans_time += timer.MicroSeconds();
    Next();
  }
  timer.Stop();
  batch_timer.Stop();
#ifdef SWMPI
  DLOG(INFO) <<" Rank "<< Caffe::mpi_rank()  << " : Fetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) <<" Rank "<< Caffe::mpi_rank()  << " : Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) <<" Rank "<< Caffe::mpi_rank()  << " : Transform time: " << trans_time / 1000 << " ms.";
#else
  DLOG(INFO) << "Fetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
#endif
}

INSTANTIATE_CLASS(IMAGENETDataLayer);
REGISTER_LAYER_CLASS(IMAGENETData);

}  // namespace caffe
