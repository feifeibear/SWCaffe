#ifndef _NETPARAMETER_H_
#define _NETPARAMETER_H_

#include "caffe/protohpp/LayerParameter.hpp"
#include <vector>
#include <string>

namespace caffe {

class NetState
{
public:
  NetState(){
    phase_ = TEST;
    stage_.resize(0);
    level_ = 0;
  }

  NetState(const NetState& other){
    this->CopyFrom(other);
  }

  inline NetState& operator=(const NetState& other) {
    this->CopyFrom(other);
    return *this;
  }

  inline std::string name() const { return name_; } 
  inline Phase phase() const { return phase_; }
  inline int level() const { return level_; }
  inline std::vector<std::string> stage() const { return stage_; }
  int stage_size() const { return stage_.size(); }
  std::string stage( int id ) const { return stage_[id]; }
  
  inline void set_name(std::string x) {name_=x;}
  inline void set_phase(Phase p) { phase_ = p; }
  inline void set_level(int x) {level_=x;}
  inline void add_stage(std::string x) {stage_.push_back(x);}

  void CopyFrom(const NetState& other) {
    name_ = other.name();
    phase_ = other.phase();
    level_ = other.level();
    int stage_size = other.stage_size();
    stage_.resize(stage_size);
    for( int i = 0; i < stage_size; ++i )
      stage_[i] = other.stage(i);
  }

private:
  std::string name_;
  Phase phase_;
  int level_;
  std::vector<std::string> stage_;
};

class NetParameter {
  public:
    NetParameter():
      force_backward_(false), debug_info_(false) {}

    NetParameter(const NetParameter& other){
      this->CopyFrom(other);
    }

    ~NetParameter(){
      clear_layer();
    }

    inline NetParameter& operator=(const NetParameter& other) {
      this->CopyFrom(other);
      return *this;
    }

    inline std::string name() const { return name_; }
    inline bool force_backward() const { return force_backward_; }
    inline bool debug_info() const { return debug_info_; }

    inline void set_name(std::string x) {name_=x;}
    inline void set_force_backward(bool x) {force_backward_=x;}
    inline void set_debug_info(bool x) {debug_info_=x;}

    inline const NetState& state() const { return state_; }
    inline NetState* mutable_state() { return &state_; }
    
    inline int layer_size() const { return layer_.size(); }
    inline const LayerParameter& layer(int id) const { return *(layer_[id]); }
    inline void add_layer(const LayerParameter& l){
      LayerParameter* lptr = new LayerParameter(l);
      layer_.push_back(lptr);
    }
    inline LayerParameter* add_layer() {
      LayerParameter* layer = new LayerParameter;
      layer_.push_back(layer);
      return layer_[layer_.size()-1];
    }
    inline LayerParameter* mutable_layer(int id) { return layer_[id]; }
    void clear_layer() {
      for (int i=0; i<layer_.size(); i++)
        delete layer_[i];
      layer_.clear();
    }
    
    std::string DebugString() const { return "TODO in NetParameter.hpp DebugString"; }

    void CopyFrom(const NetParameter& other) {
      this->name_ = other.name();
      force_backward_ = other.force_backward();
      debug_info_ = other.debug_info();
      state_.CopyFrom(other.state());
      clear_layer();
      int layer_size = other.layer_size();
      layer_.resize(layer_size);
      for( int i = 0; i < layer_size; ++i ){
        layer_[i] = new LayerParameter;
        layer_[i]->CopyFrom(other.layer(i));
      }
    }

  private:
    std::string name_;
    bool force_backward_;
    bool debug_info_;

    NetState state_;
    std::vector<LayerParameter*> layer_;
};

}

#endif
