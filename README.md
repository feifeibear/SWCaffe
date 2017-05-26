## Brief Introduction :
<img src="https://github.com/feifeibear/SWCaffe/blob/master/swdnnlogo.png" width = "300" height = "200" alt="swdnnlogo" align=center />
This is a caffe customized for Deep Nerual Network Training on Enbedded Devices.

### Dependencies:
1. blas
http://www.netlib.org/blas/

### Features
1. No protocbuf
2. boost with only hpp headers
3. No database for data storage, read from binary file
4. Support swBLAS and swDNN
https://github.com/THUHPGC/swDNN.git
large image channels, swDNN is used and small channels swBLAS is used

### Usage
modify makefile to decide using Sunway or Intel.
./models/src contains network models
make (vgg) for compling and make run(vgg) for runing scripts in ./scripts/

Sunway:
support vgg-16, alexnet, lstm, mnist, solver.
solver is defined by myself.
uncomment USE_SWDNN in Makefile.sw to use swDNN

Intel:
support vgg-16, alexnet, lstm, mnist, solver.

### Bugs
1. DataLayer is customized for mnist and imagenet
2. Only support double for Sunway

### Developer
Jiaui Fang, Zheyu Zhang
Liandeng Li

### Contact
fang_jiarui@163.com
