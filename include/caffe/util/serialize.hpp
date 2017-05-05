#ifndef _SERIALIZE_HPP_
#define _SERIALIZE_HPP_

#include <iostream>
#include <fstream>

class Serial_Blob
{
public:
  std::vector<float> data;

  void serialize_out(std::ofstream& out){
    int data_size = data.size();
    out.write((char*)&data_size, sizeof(int));
    for (int i=0; i<data.size(); i++)
      out.write((char*)&data[i], sizeof(float));
  }

  void serialize_in(std::ifstream& in){
    int data_size;
    in.read((char*)&data_size, sizeof(int));
    data.resize(data_size);
    for (int i=0; i<data_size; i++)
      in.read((char*)&data[i], sizeof(float));
  }
};

class Serial_Layer
{
public:
  std::string name;
  std::vector<Serial_Blob> blobs;

  void serialize_out(std::ofstream& out){
    int name_size = name.size();
    out.write((char*)&name_size, sizeof(int));
    for (int i=0; i<name_size; i++)
      out.write((char*)&name[i], sizeof(name[i]));
    int blobs_size = blobs.size();
    out.write((char*)&blobs_size, sizeof(int));
    for (int i=0; i<blobs.size(); i++)
      blobs[i].serialize_out(out);
  }

  void serialize_in(std::ifstream& in){
    int name_size;
    in.read((char*)&name_size, sizeof(int));
    name.resize(name_size);
    for (int i=0; i<name_size; i++)
      in.read((char*)&name[i], sizeof(name[i]));
    int blobs_size;
    in.read((char*)&blobs_size, sizeof(int));
    blobs.resize(blobs_size);
    for (int i=0; i<blobs_size; i++)
      blobs[i].serialize_in(in);
  }
};

class Serial_Net
{
public:
  std::vector<Serial_Layer> layers;

  void serialize_out(std::ofstream& out){
    int layer_size = layers.size();
    out.write((char*)&layer_size, sizeof(int));
    for (int i=0; i<layers.size(); i++)
      layers[i].serialize_out(out);
  }

  void serialize_in(std::ifstream& in){
    int layer_size;
    in.read((char*)&layer_size, sizeof(int));
    layers.resize(layer_size);
    for (int i=0; i<layer_size; i++)
      layers[i].serialize_in(in);
  }
};

#endif