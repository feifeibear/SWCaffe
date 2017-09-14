#include "caffe/caffe.hpp"
//#include <mpi.h>
//#include "caffe/util/serialize.hpp"
#include "boost/system/error_code.hpp"
#include "caffe/net.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <sys/time.h>
using namespace cv;
using namespace std;
using namespace caffe;


struct  RESULT
{
	Rect_<float> boudingboxes;
	Point2f points[5];
	float score;

	float reg[4];
};
struct  PAD_RESULT
{
	int dy;
	int edy;
	int dx;
	int edx;
	int y;
	int ey;
	int x;
	int ex;
	int tmpw;
	int tmph;
};
typedef struct _st_image_data
{
	_st_image_data()
	{
		data = NULL;
		width = 0;
		height = 0;
		num_channels = 0;
	}

	_st_image_data(int img_width,
				   int img_height,
				   int img_num_channels = 1)
	{
		data = NULL;
		width = img_width;
		height = img_height;
		num_channels = img_num_channels;
	}

	unsigned char* data;
	int width;
	int height;
	int num_channels;
} ST_IMAGE_DATA;


typedef struct _st_face_data {
	_st_face_data() {
		data = NULL;
		data_size = 0;
		x_pos = 0;
		y_pos = 0;
		width = 0;
		height = 0;
		num_channels = 0;
		memset(landmark, 0, sizeof(landmark));
	}

	_st_face_data(int face_x_pos, int face_y_pos, int face_width, int face_height,
		int img_num_channels = 1) {
		data = NULL;
		data_size = 0;
		x_pos = face_x_pos;
		y_pos = face_y_pos;
		width = face_width;
		height = face_height;
		num_channels = img_num_channels;
	}

	unsigned char* data;
	int data_size;
	float landmark[10];
	int x_pos;
	int y_pos;
	int width;
	int height;
	int num_channels;
} ST_FACE_DATA;
void  bbreg(vector<RESULT>& result)
{
	for (unsigned int i = 0; i < result.size(); i++)
	{
		float xend = result[i].boudingboxes.x + result[i].boudingboxes.width;
		float yend = result[i].boudingboxes.y + result[i].boudingboxes.height;
		result[i].boudingboxes.x = result[i].boudingboxes.x + result[i].reg[1] * result[i].boudingboxes.width;
		result[i].boudingboxes.y = result[i].boudingboxes.y + result[i].reg[0] * result[i].boudingboxes.height;
		result[i].boudingboxes.width = xend + result[i].reg[3] * result[i].boudingboxes.width - result[i].boudingboxes.x;
		result[i].boudingboxes.height = yend + result[i].reg[2] * result[i].boudingboxes.height - result[i].boudingboxes.y;

	}

}
typedef pair<int, float> PAIR;
int cmp(const PAIR& x, const PAIR& y)
{
	return x.second < y.second;
}
vector<int> nms(vector<RESULT> boxes, float threshold, string type)
{
	vector<int>pick;
	if (boxes.size() == 0)
		return pick;

	vector<int> x1;
	vector<int> y1;
	vector<int> x2;
	vector<int> y2;
	vector<PAIR> s;
	vector<int> area;
	vector<int> is_suppressed;
	for (unsigned int i = 0; i < boxes.size(); i++)
	{
		x1.push_back(boxes[i].boudingboxes.x);
		y1.push_back(boxes[i].boudingboxes.y);
		x2.push_back(boxes[i].boudingboxes.x + boxes[i].boudingboxes.width);
		y2.push_back(boxes[i].boudingboxes.y + boxes[i].boudingboxes.height);

		s.push_back(make_pair(i, boxes[i].score));

		area.push_back((boxes[i].boudingboxes.width + 1) * (boxes[i].boudingboxes.height + 1));

		is_suppressed.push_back(0);
	}

	sort(s.begin(), s.end(), cmp);

	vector<PAIR> s_copy;
	while (s.size() > 0)
	{
		int last = s.size();
		int i = s[last - 1].first;
		s[last - 1].first = -1;
		pick.push_back(i);
		last = last - 1;
		vector<float> o(last), xx1(last), yy1(last), xx2(last), yy2(last), w(last), h(last), inter(last);
		for (int m = last - 1; m >= 0; m--)
		{
			int idx = s[m].first;
			xx1[m] = max(x1[i], x1[idx]);
			yy1[m] = max(y1[i], y1[idx]);
			xx2[m] = min(x2[i], x2[idx]);
			yy2[m] = min(y2[i], y2[idx]);
			w[m] = max(0.0, xx2[m] - xx1[m] + 1.0);
			h[m] = max(0.0, yy2[m] - yy1[m] + 1.0);
			inter[m] = w[m] * h[m];

			if (type == "Min")
			{
				o[m] = inter[m] / min(area[i], area[idx]);
			}
			else
			{
				o[m] = inter[m] / (area[i] + area[idx] - inter[m]);
			}

			if (o[m] > threshold)
			{
				//printf("s[%d] = %d, del %d\n", m, s[m].first, m);
				s[m].first = -1;
			}
		}

		s_copy.clear();
		vector<PAIR>::iterator itr = s.begin();
		for (; itr != s.end(); itr++)
		{
			if (itr->first != -1)
			{
				s_copy.push_back(*itr);
			}
		}
		s = s_copy;
	}

	return pick;
}
vector<RESULT> generateBoundingBox(Mat map, vector<Mat> reg, float scale, float t)
{
	//map为CV32FC1，有1 个，reg为CV32FC1，有4个
	//use heatmap to generate bounding boxes
	vector<RESULT> result;
	
	int stride = 2;
	int cellsize = 12;

	Mat dx1 = reg[0];
	Mat dy1 = reg[1];
	Mat dx2 = reg[2];
	Mat dy2 = reg[3];
	
	vector<Point2f> xy;
	for (int i = 0; i < map.rows;i++)
	{
		for (int j = 0; j < map.cols; j++)
		{	
			if (map.at<float>(i, j) >= t)
			{
				xy.push_back(Point2f(j, i));
			}
		}
	}
	for (unsigned int i = 0; i < xy.size(); i++)
	{
		RESULT resulttmp;
		resulttmp.reg[0] = dx1.at<float>(Point2f(xy[i]));
		resulttmp.reg[1] = dy1.at<float>(Point2f(xy[i]));
		resulttmp.reg[2] = dx2.at<float>(Point2f(xy[i]));
		resulttmp.reg[3] = dy2.at<float>(Point2f(xy[i]));

		resulttmp.score = map.at<float>(Point2f(xy[i]));

		resulttmp.boudingboxes.x = floor((stride*(xy[i].x) + 1) / scale) - 1;
		resulttmp.boudingboxes.y = floor((stride*(xy[i].y) + 1) / scale) - 1;
		resulttmp.boudingboxes.width = floor((stride*(xy[i].x) + cellsize ) / scale) - 1 - resulttmp.boudingboxes.x;
		resulttmp.boudingboxes.height = floor((stride*(xy[i].y) + cellsize ) / scale) - 1 - resulttmp.boudingboxes.y;

		result.push_back(resulttmp);
		
	}
	
	return result;

}
vector<PAD_RESULT> pad(vector<RESULT> result, int w, int h)
{
	//compute the padding coordinates(pad the bounding boxes to square)
	vector<PAD_RESULT> pad_result(result.size());
	for (unsigned int i = 0; i < result.size();i++)
	{
		

		pad_result[i].x = floor(result[i].boudingboxes.x);
		pad_result[i].y = floor(result[i].boudingboxes.y);
		pad_result[i].ex = floor(result[i].boudingboxes.x + result[i].boudingboxes.width);
		pad_result[i].ey = floor(result[i].boudingboxes.y + result[i].boudingboxes.height);

		pad_result[i].tmpw = pad_result[i].ex - pad_result[i].x+1;//floor(result[i].boudingboxes.width);
		pad_result[i].tmph = pad_result[i].ey - pad_result[i].y+1;//floor(result[i].boudingboxes.height);

		pad_result[i].edx = pad_result[i].tmpw;
		pad_result[i].edy = pad_result[i].tmph;

		if (pad_result[i].ex > w)
		{
			pad_result[i].edx = pad_result[i].ex*(-1) + w + pad_result[i].tmpw;
			pad_result[i].ex = w;
		}
		if (pad_result[i].ey> h)
		{
			pad_result[i].edy = pad_result[i].ey*(-1) + h + pad_result[i].tmph;
			pad_result[i].ey = h;
		}
		if (pad_result[i].x<0)
		{
			pad_result[i].dx = 1 - pad_result[i].x;
			pad_result[i].x=0;
		}else
			pad_result[i].dx = 0;

		if (pad_result[i].y < 0)
		{
			pad_result[i].dy = 1 - pad_result[i].y;
			pad_result[i].y = 0;
		}else
			pad_result[i].dy = 0;

	}
	return pad_result;

}
void  rerec(vector<RESULT>& result)
{
	vector <float>l(result.size());
	for (unsigned int i = 0; i < result.size(); i++)
	{
		l[i] = std::max(result[i].boudingboxes.width, result[i].boudingboxes.height);
		result[i].boudingboxes.x = result[i].boudingboxes.x + result[i].boudingboxes.width*0.5 - l[i] * 0.5;
		result[i].boudingboxes.y = result[i].boudingboxes.y + result[i].boudingboxes.height*0.5 - l[i] * 0.5;
		result[i].boudingboxes.width = l[i];
		result[i].boudingboxes.height = l[i];
	}
}

inline unsigned long rpcc()
{
        unsigned long rpcc;
        asm volatile("rtc %0":"=r"(rpcc));
        return rpcc;
}
inline clock_t start()
{
	return clock();
}
inline double end(clock_t st)
{
	return (double)(1000 * (clock() - st) / CLOCKS_PER_SEC);
}

void WrapInputLayer(caffe::shared_ptr<Net<float> >& net_,std::vector<cv::Mat>* input_channels)
{
	Blob<float>* input_layer = net_->input_blobs()[0];

	int width = input_layer->width();
	int height = input_layer->height(); 
	float* input_data = input_layer->mutable_cpu_data();
	for (int i = 0; i < input_layer->channels(); ++i) 
	{
		cv::Mat channel(height, width, CV_32FC1, input_data);
		input_channels->push_back(channel);
		input_data += width * height;
	}
}

void Preprocess(caffe::shared_ptr<Net<float> >& net_, 
	int num_channels_, 
	cv::Size input_geometry_,
	const cv::Mat& img, 
	std::vector<cv::Mat>* input_channels)
{
	/* Convert the input image to the input image format of the network. */

	cv::Mat sample;
	cv::cvtColor(img, sample, CV_BGR2RGB);
	
	cv::Mat sample_float;
	if (num_channels_ == 3)
		sample.convertTo(sample_float, CV_32FC3);
	else
		sample.convertTo(sample_float, CV_32FC1);


	cv::Mat sample_resized;
	if (sample.size() != input_geometry_)
		cv::resize(sample_float, sample_resized, input_geometry_, 0.0, 0.0, INTER_AREA);
	else
		sample_resized = sample_float;


	subtract(sample_resized, 127.5, sample_resized);
	sample_resized = sample_resized.mul(0.0078125);

	/* This operation will write the separate BGR planes directly to the
	* input layer of the network because it is wrapped by the cv::Mat
	* objects in input_channels. */
	cv::split(sample_resized, *input_channels);

	CHECK(reinterpret_cast<float*>(input_channels->at(0).data)== net_->input_blobs()[0]->cpu_data())
		<< "Input channels are not wrapping the input layer of the network.";
}


void WrapInputLayerRNetONet(caffe::shared_ptr<Net<float> >& net_, std::vector<cv::Mat>* input_channels)
{
	Blob<float>* input_layer = net_->input_blobs()[0];

	int width = input_layer->width();
	int height = input_layer->height();
	int num = input_layer->num();
	float* input_data = input_layer->mutable_cpu_data();
	for (int k = 0; k < num; k++)
	{
		for (int i = 0; i < input_layer->channels(); ++i)
		{
			cv::Mat channel(height, width, CV_32FC1, input_data);
			input_channels->push_back(channel);
			input_data += width * height;
		}

	}
	
}
void PreprocessRNetONet(caffe::shared_ptr<Net<float> >& net_,
	vector<Mat>& temping,
	std::vector<cv::Mat>* input_channels)
{
	Blob<float>* input_layer = net_->input_blobs()[0];		

	for (unsigned int i = 0; i < temping.size();i++)
	{
		Mat tmp = temping[i];
		cv::cvtColor(tmp, tmp, CV_BGR2RGB);

		subtract(tmp, 127.5, tmp);
		tmp = tmp.mul(0.0078125);

		vector<Mat>temp_inputChannels;
		cv::split(tmp, temp_inputChannels);
		temp_inputChannels[0].copyTo(input_channels->at(i * input_layer->channels() + 0));
		temp_inputChannels[1].copyTo(input_channels->at(i * input_layer->channels() + 1));
		temp_inputChannels[2].copyTo(input_channels->at(i * input_layer->channels() + 2));
		
	}

	CHECK(reinterpret_cast<float*>(input_channels->at(0).data) == net_->input_blobs()[0]->cpu_data())
		<< "Input channels are not wrapping the input layer of the network.";
}

/*

vector<RESULT>  detect_face(Mat& img,
	int minsize,
	caffe::shared_ptr<Net<float> >& PNet,
	caffe::shared_ptr<Net<float> >& RNet,
	caffe::shared_ptr<Net<float> >& ONet,
	float threshold[3],
	bool fastresize,
	float factor)
{
	img = img.t();
	
	vector<RESULT> total_result;
	vector<RESULT> total_resultmid;
	vector<RESULT> total_resultend;

	int factor_count = 0;

	int h = img.rows;
	int w = img.cols;

	float minl = std::min(h,w);

	
	float m = 12.0 / minsize;
	minl = minl*m;
	
	//creat scale pyramid
	vector<float>scales;
	while (minl >= 12)
	{
		scales.push_back(m*powf(factor ,factor_count));
		minl = minl*factor;
		factor_count = factor_count + 1;
	}
	
	//first stage
	vector<RESULT> total_result1;
	for (unsigned int j = 0; j < scales.size(); j++)
	{
		float scale = scales[j];
		int hs = ceil(h*scale);
		int ws = ceil(w*scale);
		int PNet_num_channels_ = 3;
		cv::Size PNet_input_geometry_ = Size(ws, hs);


		Blob<float>* PNet_input_layer = PNet->input_blobs()[0];
		// Forward dimension change to all layers. 
		
		PNet_input_layer->Reshape(1, 3, hs, ws);
		//PNet->Reshape();

		std::vector<cv::Mat> PNet_input_channels;
		WrapInputLayer(PNet, &PNet_input_channels);
		Preprocess(PNet, PNet_num_channels_, PNet_input_geometry_, img, &PNet_input_channels);		
		
		PNet->Forward();
		// Copy the output layer to a std::vector 
		Blob<float>* output_layer0 = PNet->output_blobs()[0];
		Blob<float>* output_layer1 = PNet->output_blobs()[1];
		
		vector<Mat> reg;
		for (int kkk = 0; kkk < 4; kkk++)
		{
		Mat regtmp(output_layer0->height(), output_layer0->width(), CV_32FC1);
		memcpy(regtmp.data, output_layer0->cpu_data() + output_layer0->width()*output_layer0->height()*kkk, output_layer0->width()*output_layer0->height()*sizeof(float));
		reg.push_back(regtmp);
		}

		cv::Mat map(output_layer1->height(), output_layer1->width(), CV_32FC1);
		memcpy(map.data, output_layer1->cpu_data() + output_layer1->width()*output_layer1->height(), output_layer1->width()*output_layer1->height()*sizeof(float));

		vector<RESULT>result1;
		result1=generateBoundingBox(map,reg, scale, threshold[0]);
	
		//inter - scale nms
		vector<int> pick;
		pick = nms(result1, 0.5, "Union");

		vector<RESULT>result;
		for (unsigned int pick_iter = 0; pick_iter < pick.size(); pick_iter++)
		{
			result.push_back(result1[pick[pick_iter]]);
		}
		if (result.size() != 0)
		{
			for (unsigned int i = 0; i < result.size(); i++)
			{
				total_result1.push_back(result[i]);
			}
		}
		
	}

	
	vector<PAD_RESULT> pad_result;
	if (total_result1.size() != 0)
	{
		vector<int> picktmp;
		picktmp = nms(total_result1, 0.7, "Union");
		for (unsigned int pick_iter = 0; pick_iter < picktmp.size(); pick_iter++)
		{
			total_result.push_back(total_result1[picktmp[pick_iter]]);
		}
		for (unsigned int i = 0; i < total_result.size(); i++)
		{
			int regw = total_result[i].boudingboxes.width;
			int regh = total_result[i].boudingboxes.height;
			int x2 = total_result[i].boudingboxes.x + total_result[i].boudingboxes.width;
			int y2 = total_result[i].boudingboxes.y + total_result[i].boudingboxes.height;

			total_result[i].boudingboxes.x = total_result[i].boudingboxes.x + total_result[i].reg[1] * regw;
			total_result[i].boudingboxes.y = total_result[i].boudingboxes.y + total_result[i].reg[0] * regh;
			
			total_result[i].boudingboxes.width = x2 + total_result[i].reg[3] * regw - total_result[i].boudingboxes.x;
			total_result[i].boudingboxes.height = y2 + total_result[i].reg[2] * regh - total_result[i].boudingboxes.y;

		}
		rerec(total_result);
		pad_result = pad(total_result, w, h);
	}
	int numbox = (int)total_result.size();
	
	//return total_result;
	
// second stage
	if (numbox > 0)
	{
		vector<Mat>temping;
		for (int k = 0; k < numbox;k++)
		{
			Mat tmp(pad_result[k].tmph, pad_result[k].tmpw, CV_32FC3);
			
			img(Range(pad_result[k].y, pad_result[k].ey), Range(pad_result[k].x, pad_result[k].ex)).copyTo(tmp);
			tmp.convertTo(tmp, CV_32FC3);
			resize(tmp, tmp, Size(24, 24), 0.0, 0.0, INTER_CUBIC);

// 			subtract(tmp, 127.5, tmp);
// 			tmp = tmp.mul(0.0078125);

			temping.push_back(tmp);
		}
		
		Blob<float>* RNet_input_layer = RNet->input_blobs()[0];
		// Forward dimension change to all layers. 

		RNet_input_layer->Reshape(numbox, 3, 24, 24);
		//RNet->Reshape();

		std::vector<cv::Mat> RNet_input_channels;
		WrapInputLayerRNetONet(RNet, &RNet_input_channels);
		PreprocessRNetONet(RNet, temping, &RNet_input_channels);
		RNet->Forward();
		Blob<float>* output_layer0 = RNet->output_blobs()[0];
		Blob<float>* output_layer1 = RNet->output_blobs()[1];
		
		Mat regtmp(output_layer0->num(), 4, CV_32FC1);
		memcpy(regtmp.data, output_layer0->cpu_data(), 4 * output_layer0->num()*sizeof(float));
		Mat reg = regtmp.t();
		
		vector <float>score;
		for (int i = 0; i < output_layer1->count(); i++)
		{
			if (i % 2 != 0)
				score.push_back(*(output_layer1->cpu_data() + i));
		}
		vector<int>pass;
		for (unsigned int i = 0; i < score.size();i++)
		{
			if (score[i]>threshold[1])
			{
				pass.push_back(i);
			}
		}
		vector<RESULT> total_resulttmp(pass.size());
		for (unsigned int i = 0; i < pass.size();i++)
		{
			total_resulttmp[i].boudingboxes=total_result[pass[i]].boudingboxes;
			total_resulttmp[i].score = score[pass[i]];

			total_resulttmp[i].reg[0] = reg.at<float>(0, pass[i]);
			total_resulttmp[i].reg[1] = reg.at<float>(1, pass[i]);
			total_resulttmp[i].reg[2] = reg.at<float>(2, pass[i]);
			total_resulttmp[i].reg[3] = reg.at<float>(3, pass[i]);

#ifdef _DRAW_DET
			rectangle(img, total_resulttmp[i].boudingboxes, Scalar(0, 0, 255), 2);
#endif
		}

		
		
		if (total_resulttmp.size()>0)
		{
			vector<int> picktmp;
			picktmp = nms(total_resulttmp, 0.7, "Union");
			for (unsigned int i = 0; i < picktmp.size();i++)
			{
				total_resultmid.push_back(total_resulttmp[picktmp[i]]);
			}
			bbreg(total_resultmid);
			rerec(total_resultmid);
		}
		unsigned int numbox = total_resultmid.size();

		//return total_resultmid;
	
	
		
		//third stage
		if (numbox>0)
		{
			pad_result = pad(total_resultmid, w, h);

			vector<Mat>temping;
			for (unsigned int k = 0; k < numbox; k++)
			{
				Mat tmp(pad_result[k].tmph, pad_result[k].tmpw, CV_32FC3);

				img(Range(pad_result[k].y, pad_result[k].ey), Range(pad_result[k].x, pad_result[k].ex)).copyTo(tmp);
				tmp.convertTo(tmp, CV_32FC3);
				resize(tmp, tmp, Size(48, 48), 0.0, 0.0, INTER_CUBIC);

				temping.push_back(tmp);
			}

			Blob<float>* ONet_input_layer = ONet->input_blobs()[0];
			// Forward dimension change to all layers. 

			ONet_input_layer->Reshape(numbox, 3, 48, 48);
			//ONet->Reshape();

			std::vector<cv::Mat> ONet_input_channels;
			WrapInputLayerRNetONet(ONet, &ONet_input_channels);
			PreprocessRNetONet(ONet, temping, &ONet_input_channels);
			ONet->Forward();

			Blob<float>* output_layer0 = ONet->output_blobs()[0];
			Blob<float>* output_layer1 = ONet->output_blobs()[1];
			Blob<float>* output_layer2 = ONet->output_blobs()[2];


			Mat regtmp(output_layer0->num(), 4, CV_32FC1);
			memcpy(regtmp.data, output_layer0->cpu_data(), 4 * output_layer0->num()*sizeof(float));
			Mat reg = regtmp.t();

			Mat pointstmp(output_layer1->num(), 10, CV_32FC1);
			memcpy(pointstmp.data, output_layer1->cpu_data(), 10 * output_layer1->num()*sizeof(float));
			Mat points = pointstmp.t();

			vector <float>score;
			for (int i = 0; i < output_layer2->count(); i++)
			{
				if (i % 2 != 0)
					score.push_back(*(output_layer2->cpu_data() + i));
			}
			
			vector<int>pass;
			for (unsigned int i = 0; i<score.size();i++)
			{
				if (score[i]>threshold[2])
				{
					pass.push_back(i);
				}
			}
			
			vector<RESULT> total_resulttmp2(pass.size());
			for (unsigned int i = 0; i < pass.size(); i++)
			{
				total_resulttmp2[i].boudingboxes=total_resultmid[pass[i]].boudingboxes;
				total_resulttmp2[i].score = score[pass[i]];
				total_resulttmp2[i].points[0] = Point2f(points.at<float>(0, pass[i]), points.at<float>(5, pass[i]));
				total_resulttmp2[i].points[1] = Point2f(points.at<float>(1, pass[i]), points.at<float>(6, pass[i]));
				total_resulttmp2[i].points[2] = Point2f(points.at<float>(2, pass[i]), points.at<float>(7, pass[i]));
				total_resulttmp2[i].points[3] = Point2f(points.at<float>(3, pass[i]), points.at<float>(8, pass[i]));
				total_resulttmp2[i].points[4] = Point2f(points.at<float>(4, pass[i]), points.at<float>(9, pass[i]));

				total_resulttmp2[i].reg[0] = reg.at<float>(0, pass[i]);
				total_resulttmp2[i].reg[1] = reg.at<float>(1, pass[i]);
				total_resulttmp2[i].reg[2] = reg.at<float>(2, pass[i]);
				total_resulttmp2[i].reg[3] = reg.at<float>(3, pass[i]);
			}
			for (unsigned int i = 0; i<total_resulttmp2.size();i++)
			{
				int w = total_resulttmp2[i].boudingboxes.width;
				int h = total_resulttmp2[i].boudingboxes.height;
				for (int j = 0; j < 5; j++)
				{
					int x = w*total_resulttmp2[i].points[j].y;
					int y = h*total_resulttmp2[i].points[j].x;
					total_resulttmp2[i].points[j].x = x + total_resulttmp2[i].boudingboxes.x - 1;
					total_resulttmp2[i].points[j].y = y + total_resulttmp2[i].boudingboxes.y - 1;
				}			
			}

			if (total_resulttmp2.size()>0)
			{
				bbreg(total_resulttmp2);
				vector<int>pick;
				pick = nms(total_resulttmp2, 0.7, "Min");
				for (unsigned int i = 0; i < pick.size();i++)
				{
					total_resultend.push_back(total_resulttmp2[pick[i]]);
				}
			}
		}
	}
	
	return total_resultend;
}
*/
vector<RESULT>  detect_face(Mat& img,
	int minsize,
	caffe::shared_ptr<Net<float> >& PNet,
	caffe::shared_ptr<Net<float> >& RNet,
	caffe::shared_ptr<Net<float> >& ONet,
	float threshold[3],
	bool fastresize,
	float factor)
{
#ifdef DEBUG_PRINT_TIME  
   	unsigned long lTime=0,lTmp=0;
		unsigned long d1=0,d2=0,d3=0,d4=0;
		double dTime=0;
   	clock_t lStart = start(),lMid=0;

	lStart = start();
#endif
  img = img.t();
#ifdef DEBUG_PRINT_TIME
  dTime = end(lStart);
	cout << "image transepose run time= " << dTime << endl;	
#endif
	vector<RESULT> total_result;
	vector<RESULT> total_resultmid;
	vector<RESULT> total_resultend;

	int factor_count = 0;

	int h = img.rows;
	int w = img.cols;

	float minl = std::min(h,w);
  const float min_pixes = 12.0; 
	float m = min_pixes / minsize;
	minl = minl*m;
	
	//creat scale pyramid
	vector<float>scales;
	while (minl >= min_pixes)
	{
		minl = minl*factor;
		
		scales.push_back(m*powf(factor ,factor_count));
		factor_count = factor_count + 1;
	}
#ifdef DEBUG_PRINT_TIME  
  dTime = end(lStart);
	cout << "creat scale pyramid run time= " << dTime << endl;
#endif
    int PNet_num_channels_ = 3;
	cv::Mat sample;
#ifdef DEBUG_PRINT_TIME	
	lStart = start();	
#endif	
  cv::cvtColor(img, sample, CV_BGR2RGB);	
#ifdef DEBUG_PRINT_TIME	
  dTime = end(lStart);
	cout << "image cvtColor run time= " << dTime <<" w="<<sample.rows<<" h="<<sample.cols<<" channels="<<sample.channels()<<" depth ="<<sample.depth()<< endl;
#endif
	cv::Mat sample_float,img_float;
#ifdef DEBUG_PRINT_TIME	
	lStart = start();	
#endif
	sample.convertTo(img_float, CV_32FC3,0.0078125,0.99609375);	
#ifdef DEBUG_PRINT_TIME
  dTime = end(lStart);
	cout << "image convertTo run time= " << dTime << "old value ="<< sample.at<Vec3b>(1,1)<<" new value=" << img_float.at<Vec3f>(1,1)<< endl;
#endif	
	
	//first stage
	cv::Mat sample_resized;
	vector<RESULT> total_result1;
	for (int j = 0; j < scales.size(); j++)
	{
		float scale = scales[j];
		int hs = ceil(h*scale);
		int ws = ceil(w*scale);
		
		float minv = std::min(hs,ws);
		//if( minv < minsize) break;
		// Forward dimension change to all layers. 
		cv::Size PNet_input_geometry_ = Size(ws, hs);
		Blob<float>* PNet_input_layer = PNet->input_blobs()[0];
		PNet_input_layer->Reshape(1, 3, hs, ws);
		std::vector<cv::Mat> PNet_input_channels;		
		float* input_data = PNet_input_layer->mutable_cpu_data();
		for (int i = 0; i < PNet_input_layer->channels(); ++i) 
		{
			cv::Mat channel(hs, ws, CV_32FC1, input_data);
			PNet_input_channels.push_back(channel);
			input_data += hs * ws;
		}
#ifdef DEBUG_PRINT_TIME	
    lMid = start();
#endif		
		//sample.convertTo(sample_resized, CV_32FC3);	
	  cv::resize(img_float, sample_resized, PNet_input_geometry_, 0.0, 0.0,INTER_AREA);//INTER_AREA,INTER_NEAREST
		//subtract(sample_resized, 127.5, sample_resized);
	  //sample_resized = sample_resized.mul(0.0078125);	
		cv::split(sample_resized, PNet_input_channels);
#ifdef DEBUG_PRINT_TIME		
    dTime = end(lMid);	
		d4 += dTime;
       			
		lMid = start();
#endif
		PNet->Forward();
#ifdef DEBUG_PRINT_TIME    
    dTime = end(lMid);	
		d1 += dTime;
    cout << "First stage PNET run time= " << dTime <<" scale"<<scale<< " hs =" << hs << " ws =" << ws << endl;
#endif		
		// Copy the output layer to a std::vector 
		
		Blob<float>* output_layer0 = PNet->output_blobs()[0];
		Blob<float>* output_layer1 = PNet->output_blobs()[1];
		
		vector<Mat> reg;
		for (int kkk = 0; kkk < 4; kkk++)
		{
			Mat regtmp(output_layer0->height(), output_layer0->width(), CV_32FC1);
			memcpy(regtmp.data, output_layer0->cpu_data() + output_layer0->width()*output_layer0->height()*kkk, output_layer0->width()*output_layer0->height()*sizeof(float));
			reg.push_back(regtmp);
		}

		cv::Mat map(output_layer1->height(), output_layer1->width(), CV_32FC1);
		memcpy(map.data, output_layer1->cpu_data() + output_layer1->width()*output_layer1->height(), output_layer1->width()*output_layer1->height()*sizeof(float));

		vector<RESULT>result1;
		result1=generateBoundingBox(map,reg, scale, threshold[0]);
	
		//inter - scale nms
		vector<int> pick;
		pick = nms(result1, 0.5, "Union");

		vector<RESULT>result;
		for (unsigned int pick_iter = 0; pick_iter < pick.size(); pick_iter++)
		{
			result.push_back(result1[pick[pick_iter]]);
		}
		if (result.size() != 0)
		{
			for (unsigned int i = 0; i < result.size(); i++)
			{
				total_result1.push_back(result[i]);
			}
		}
	}

	vector<PAD_RESULT> pad_result;
	if (total_result1.size() != 0)
	{
		vector<int> picktmp;
		picktmp = nms(total_result1, 0.7, "Union");
		for (unsigned int pick_iter = 0; pick_iter < picktmp.size(); pick_iter++)
		{
			total_result.push_back(total_result1[picktmp[pick_iter]]);
		}
		for (unsigned int i = 0; i < total_result.size(); i++)
		{
			int regw = total_result[i].boudingboxes.width;
			int regh = total_result[i].boudingboxes.height;
			int x2 = total_result[i].boudingboxes.x + total_result[i].boudingboxes.width;
			int y2 = total_result[i].boudingboxes.y + total_result[i].boudingboxes.height;

			total_result[i].boudingboxes.x = total_result[i].boudingboxes.x + total_result[i].reg[1] * regw;
			total_result[i].boudingboxes.y = total_result[i].boudingboxes.y + total_result[i].reg[0] * regh;
			
			total_result[i].boudingboxes.width = x2 + total_result[i].reg[3] * regw - total_result[i].boudingboxes.x;
			total_result[i].boudingboxes.height = y2 + total_result[i].reg[2] * regh - total_result[i].boudingboxes.y;

		}
		rerec(total_result);
		h = img.rows;
		w = img.cols;
		pad_result = pad(total_result, w, h);
	}

	int numbox = (int)total_result.size();
#ifdef DEBUG_PRINT_TIME  
 	dTime = end(lStart);	
	cout << "First stage total time= " << dTime << " Forward time= " <<d1<< " Ratio= "<<d1/dTime<< " Image Ratio= "<<d4/dTime<<" num="<<numbox<<endl;
#endif
	//return total_result;
	
// second stage
	if (numbox > 0)
	{
#ifdef DEBUG_PRINT_TIME
    lStart = start();
#endif
		Blob<float>* RNet_input_layer = RNet->input_blobs()[0];
		// Forward dimension change to all layers. 
		int width = 24,height=24;
		std::vector<cv::Mat> RNet_input_channels;
		RNet_input_layer->Reshape(numbox, 3, width, height);
		
		
		int len = width*height;
		int num = RNet_input_layer->num();
		float* input_data = RNet_input_layer->mutable_cpu_data();
		for (int k = 0; k < num; k++)
		{
			for (int i = 0; i < RNet_input_layer->channels(); ++i)
			{
				cv::Mat channel(width, height, CV_32FC1, input_data);
				RNet_input_channels.push_back(channel);
				input_data += len;
			}

		}		
	
		for (int k = 0; k < numbox;k++)
		{
			Mat tmp(pad_result[k].tmph, pad_result[k].tmpw, CV_32FC3);
		
			img_float(Range(pad_result[k].y, pad_result[k].ey), Range(pad_result[k].x, pad_result[k].ex)).copyTo(tmp);
			//tmp.convertTo(tmp, CV_32FC3);
			resize(tmp, tmp, Size(width, height), 0.0, 0.0,INTER_CUBIC);

 			//subtract(tmp, 127.5, tmp);
 			//tmp = tmp.mul(0.0078125);

			vector<Mat>temp_inputChannels;
			cv::split(tmp, temp_inputChannels);
			temp_inputChannels[0].copyTo(RNet_input_channels.at(k * RNet_input_layer->channels() + 0));
			temp_inputChannels[1].copyTo(RNet_input_channels.at(k * RNet_input_layer->channels() + 1));
			temp_inputChannels[2].copyTo(RNet_input_channels.at(k * RNet_input_layer->channels() + 2));
		}

#ifdef DEBUG_PRINT_TIME
		lMid = start();
#endif
		RNet->Forward();
#ifdef DEBUG_PRINT_TIME
    dTime = end(lMid);	
		d2 += dTime;
#endif
	    //cout << "second stage RNet run time= " << dTime << endl;
		
		Blob<float>* output_layer0 = RNet->output_blobs()[0];
		Blob<float>* output_layer1 = RNet->output_blobs()[1];
		
		Mat regtmp(output_layer0->num(), 4, CV_32FC1);
		memcpy(regtmp.data, output_layer0->cpu_data(), 4 * output_layer0->num()*sizeof(float));
		Mat reg = regtmp.t();
		
		vector <float>score;
		for (int i = 0; i < output_layer1->count(); i++)
		{
			if (i % 2 != 0)
				score.push_back(*(output_layer1->cpu_data() + i));
		}
		vector<int>pass;
		for (unsigned int i = 0; i < score.size();i++)
		{
			if (score[i]>threshold[1])
			{
				pass.push_back(i);
			}
		}
		vector<RESULT> total_resulttmp(pass.size());
		for (unsigned int i = 0; i < pass.size();i++)
		{
			total_resulttmp[i].boudingboxes=total_result[pass[i]].boudingboxes;
			total_resulttmp[i].score = score[pass[i]];

			total_resulttmp[i].reg[0] = reg.at<float>(0, pass[i]);
			total_resulttmp[i].reg[1] = reg.at<float>(1, pass[i]);
			total_resulttmp[i].reg[2] = reg.at<float>(2, pass[i]);
			total_resulttmp[i].reg[3] = reg.at<float>(3, pass[i]);

#ifdef _DRAW_DET
			rectangle(img, total_resulttmp[i].boudingboxes, Scalar(0, 0, 255), 2);
#endif
		}
		
		
		if (total_resulttmp.size()>0)
		{
			vector<int> picktmp;
			picktmp = nms(total_resulttmp, 0.7, "Union");
			for (unsigned int i = 0; i < picktmp.size();i++)
			{
				total_resultmid.push_back(total_resulttmp[picktmp[i]]);
			}
			bbreg(total_resultmid);
			rerec(total_resultmid);
		}
		unsigned int numbox = total_resultmid.size();

		//return total_resultmid;
		
#ifdef DEBUG_PRINT_TIME  
 	dTime = end(lStart);	
	cout << "Second stage total time= " << dTime << " Forward time= " <<d2<< " Ratio= "<<d2/dTime<<" num="<<numbox<<endl;
	lStart = start();
#endif		
		//third stage
		if (numbox>0)
		{
			h = img.rows;
			w = img.cols;
			pad_result = pad(total_resultmid, w, h);
           
			// Forward dimension change to all layers. 
			Blob<float>* ONet_input_layer = ONet->input_blobs()[0];
			int width = 48,height=48;
			std::vector<cv::Mat> ONet_input_channels;
			ONet_input_layer->Reshape(numbox, 3, width, height);			
			
			int len = width*height;
			int num = ONet_input_layer->num();
			float* input_data = ONet_input_layer->mutable_cpu_data();
			for (int k = 0; k < num; k++)
			{
				for (int i = 0; i < ONet_input_layer->channels(); ++i)
				{
					cv::Mat channel(width, height, CV_32FC1, input_data);
					ONet_input_channels.push_back(channel);
					input_data += len;
				}

			}		
		
			for (int k = 0; k < numbox;k++)
			{
				Mat tmp(pad_result[k].tmph, pad_result[k].tmpw, CV_32FC3);

				img_float(Range(pad_result[k].y, pad_result[k].ey), Range(pad_result[k].x, pad_result[k].ex)).copyTo(tmp);
				//tmp.convertTo(tmp, CV_32FC3);
				resize(tmp, tmp, Size(width, height), 0.0, 0.0,INTER_CUBIC);

				//subtract(tmp, 127.5, tmp);
				//tmp = tmp.mul(0.0078125);

				vector<Mat>temp_inputChannels;
				cv::split(tmp, temp_inputChannels);
				temp_inputChannels[0].copyTo(ONet_input_channels.at(k * ONet_input_layer->channels() + 0));
				temp_inputChannels[1].copyTo(ONet_input_channels.at(k * ONet_input_layer->channels() + 1));
				temp_inputChannels[2].copyTo(ONet_input_channels.at(k * ONet_input_layer->channels() + 2));
			}
#ifdef DEBUG_PRINT_TIME		
			lMid = start();
#endif
		  ONet->Forward();
#ifdef DEBUG_PRINT_TIME      
      dTime = end(lMid);		
			d3 += dTime;
#endif
     //cout << "Third stage ONet run time= " << dTime << endl;
		
			Blob<float>* output_layer0 = ONet->output_blobs()[0];
			Blob<float>* output_layer1 = ONet->output_blobs()[1];
			Blob<float>* output_layer2 = ONet->output_blobs()[2];


			Mat regtmp(output_layer0->num(), 4, CV_32FC1);
			memcpy(regtmp.data, output_layer0->cpu_data(), 4 * output_layer0->num()*sizeof(float));
			Mat reg = regtmp.t();

			Mat pointstmp(output_layer1->num(), 10, CV_32FC1);
			memcpy(pointstmp.data, output_layer1->cpu_data(), 10 * output_layer1->num()*sizeof(float));
			Mat points = pointstmp.t();

			vector <float>score;
			for (int i = 0; i < output_layer2->count(); i++)
			{
				if (i % 2 != 0)
					score.push_back(*(output_layer2->cpu_data() + i));
			}
			
			vector<int>pass;
			for (unsigned int i = 0; i<score.size();i++)
			{
				if (score[i]>threshold[2])
				{
					pass.push_back(i);
				}
			}
			
			vector<RESULT> total_resulttmp2(pass.size());
			for (unsigned int i = 0; i < pass.size(); i++)
			{
				total_resulttmp2[i].boudingboxes=total_resultmid[pass[i]].boudingboxes;
				total_resulttmp2[i].score = score[pass[i]];
				total_resulttmp2[i].points[0] = Point2f(points.at<float>(0, pass[i]), points.at<float>(5, pass[i]));
				total_resulttmp2[i].points[1] = Point2f(points.at<float>(1, pass[i]), points.at<float>(6, pass[i]));
				total_resulttmp2[i].points[2] = Point2f(points.at<float>(2, pass[i]), points.at<float>(7, pass[i]));
				total_resulttmp2[i].points[3] = Point2f(points.at<float>(3, pass[i]), points.at<float>(8, pass[i]));
				total_resulttmp2[i].points[4] = Point2f(points.at<float>(4, pass[i]), points.at<float>(9, pass[i]));

				total_resulttmp2[i].reg[0] = reg.at<float>(0, pass[i]);
				total_resulttmp2[i].reg[1] = reg.at<float>(1, pass[i]);
				total_resulttmp2[i].reg[2] = reg.at<float>(2, pass[i]);
				total_resulttmp2[i].reg[3] = reg.at<float>(3, pass[i]);
			}
			for (unsigned int i = 0; i<total_resulttmp2.size();i++)
			{
				int w = total_resulttmp2[i].boudingboxes.width;
				int h = total_resulttmp2[i].boudingboxes.height;
				for (int j = 0; j < 5; j++)
				{
					int x = w*total_resulttmp2[i].points[j].y;
					int y = h*total_resulttmp2[i].points[j].x;
					total_resulttmp2[i].points[j].x = x + total_resulttmp2[i].boudingboxes.x - 1;
					total_resulttmp2[i].points[j].y = y + total_resulttmp2[i].boudingboxes.y - 1;
				}			
			}

			if (total_resulttmp2.size()>0)
			{
				bbreg(total_resulttmp2);
				vector<int>pick;
				pick = nms(total_resulttmp2, 0.7, "Min");
				for (unsigned int i = 0; i < pick.size();i++)
				{
					total_resultend.push_back(total_resulttmp2[pick[i]]);
				}
			}
		}
#ifdef DEBUG_PRINT_TIME  
    dTime = end(lStart);	
    cout << "Third stage total time= " << dTime << " Forward time= " <<d3<< " Ratio= "<<d3/dTime <<endl;
#endif
	}
	
	return total_resultend;
}

bool InitModels(std::string caffe_model_path,
      caffe::shared_ptr<caffe::Net<float> >& PNet,
        caffe::shared_ptr<caffe::Net<float> >& RNet,
          caffe::shared_ptr<caffe::Net<float> >& ONet)
{
    int PNet_num_channels_, RNet_num_channels_, ONet_num_channels_;
    cv::Size PNet_input_geometry_, RNet_input_geometry_, ONet_input_geometry_;

    // Load the network
    PNet.reset(new caffe::Net<float>(caffe_model_path + "/pbx.prototxt", TEST));
    PNet->CopyTrainedLayersFrom(caffe_model_path + "/pbx.caffemodel");
    if (PNet->num_inputs() != 1 || PNet->num_outputs() != 2)
    {
        return false;
    }
    
    caffe::Blob<float>* PNet_input_layer = PNet->input_blobs()[0];
    PNet_num_channels_ = PNet_input_layer->channels();
    PNet_input_geometry_ = cv::Size(PNet_input_layer->width(), PNet_input_layer->height());
    if (PNet_num_channels_ != 1 && PNet_num_channels_ != 3)
    {
        return false;
    }
    RNet.reset(new caffe::Net<float>(caffe_model_path + "/dvr.prototxt", TEST));
    RNet->CopyTrainedLayersFrom(caffe_model_path + "/dvr.caffemodel");
    if (RNet->num_inputs() != 1 || RNet->num_outputs() != 2)
    {
        return false;
    }
    caffe::Blob<float>* RNet_input_layer = RNet->input_blobs()[0];
    RNet_num_channels_ = RNet_input_layer->channels();
    RNet_input_geometry_ = cv::Size(RNet_input_layer->width(), RNet_input_layer->height());
    if (RNet_num_channels_ != 1 && RNet_num_channels_ != 3)
    {
        return false;
    }
    caffe::Blob<float>* ONet_input_layer = NULL;
    ONet.reset(new caffe::Net<float>(caffe_model_path + "/ldm.prototxt", TEST));
    ONet->CopyTrainedLayersFrom(caffe_model_path + "/ldm.caffemodel");
    if (ONet->num_inputs() != 1 || ONet->num_outputs() != 3)
    {
        return false;
    }
        
    ONet_input_layer = ONet->input_blobs()[0];
    ONet_num_channels_ = ONet_input_layer->channels();
    ONet_input_geometry_ = cv::Size(ONet_input_layer->width(), ONet_input_layer->height());
    if (ONet_num_channels_ != 1 && ONet_num_channels_ != 3)
    {
        return false;
    }
        
    return true;
}


int main (int argc, char ** argv) {
  caffe::shared_ptr<caffe::Net<float> >* PNet = new caffe::shared_ptr<caffe::Net<float> >;
  caffe::shared_ptr<caffe::Net<float> >* RNet = new caffe::shared_ptr<caffe::Net<float> >;
  caffe::shared_ptr<caffe::Net<float> >* ONet = new caffe::shared_ptr<caffe::Net<float> >;

  bool bRet = InitModels("./models/prototxt", *PNet, *RNet, *ONet);
  if (!bRet)
  {
	return false;
  }
  
  //steps's threshold
  float threshold[3] = { 0.650, 0.700, 0.700 };
  //scale factor
  float factor = 0.7;
  vector<RESULT> result;
  int minFaceSize = 40;
  
  cv::Mat img,srcImg;
  //img = cv::imread("./image/peter1920.jpg");
  img = cv::imread(argv[1]);
  srcImg = img.clone();
  double dTime=0;
  clock_t lStart = start();
  result = detect_face(img, minFaceSize, *PNet, *RNet, *ONet, threshold, false, factor);
  dTime = end(lStart);
  int ifacecount = result.size();
  std::vector<ST_FACE_DATA> vctFaces;
  // get faces data
  for (unsigned int u = 0; u < result.size(); u++)
   {
         ST_FACE_DATA stFaceData;
         stFaceData.x_pos = result[u].boudingboxes.y;
         stFaceData.y_pos = result[u].boudingboxes.x;
         stFaceData.width = result[u].boudingboxes.height;
         stFaceData.height = result[u].boudingboxes.width;
         stFaceData.num_channels = img.channels();
         for (int i = 0; i < 5; ++i) {
            stFaceData.landmark[i * 2] = result[u].points[i].y;
            stFaceData.landmark[i * 2 + 1] = result[u].points[i].x;
         }
  
         vctFaces.push_back(stFaceData);
  }
		
  for (unsigned int u = 0; u < vctFaces.size(); u++)
  {
		rectangle(srcImg, cv::Rect(vctFaces[u].x_pos, vctFaces[u].y_pos, vctFaces[u].width, vctFaces[u].height), Scalar(255, 0, 0), 2);
		for (int k = 0; k < 5; k++)
		{
			circle(srcImg, cv::Point(vctFaces[u].landmark[k * 2], vctFaces[u].landmark[k * 2 + 1]), 2, Scalar(255, 255, 0), 4);
		}
  }
  
  cv::imwrite("1.jpg", srcImg);
  cout <<"time=" << dTime << " num= " << ifacecount <<endl;
  return 0;
}
