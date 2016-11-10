/*
fast clustering by find peak density.
input: C(N,2) of distance of each samples.
output:the label of each samples,and whether belongs to a cluster center.

Author£ºfandy
Date£º2016.1
*/

#ifndef FASTCLUSTER_H__
#define FASTCLUSTER_H__

#include <iostream>
#include <vector>
#include <array>
#include <opencv/cxcore.hpp>
#include "common/common.hpp"

using namespace common;

namespace arecog {

struct node
{
	double rho;
	double delta;
	double gamma;
	int idx;
};

struct cluster
{
	int classid;
	int centerid;
	int nelement;
	int ncore;
	int nhalo;
	double centerrho;
	std::vector<int> elements;
};

struct androidpoint
{
	int label;
	std::string filename;
	double x_width;
	double y_height;
};

struct picture
{
	std::array<int, 6> date;
	//Image img;
	std::string filepath;
	std::string filename;
};

//struct AImage
//{
//	cv::Mat img;
//	int ori;
//	std::array<int, 6> date;
//	std::array<int, 3> locate;
//	std::vector<FaceInfo> faces;
//};

struct picsInoneTime
{
	std::vector<picture> pic;
};

struct datapoint
{
	int label;
	bool clustcenter;
};

bool comp(node x, node y);
bool comppics(picture x, picture y);
bool comprho(cluster x, cluster y);
bool compgamma(node x, node y);
bool descend(double x, double y);

struct CFace
{
	std::string imgpath;
	std::string srcpath;
	FaceInfo faceinfo;
	int facelabel;
	bool isclustCenter;
};

class Cluster{

public:
	Cluster(int N);
	int AClustProcess_Init(std::vector<CFace>& cfaces, double dc,double rth,double dth);
	int AClustProcess_GetDist(std::vector<CFace>& cfaces, cv::Mat &thisdist);
	int AClustProcess_Clust(std::vector<CFace> &cfaces);

	void fillval(std::vector<double> &a, double &val);
	void fillval(std::vector<int> &a, int &val);

	void getnrate(cv::Mat &dist);
	void getDc(cv::Mat &dist, double& percent);
	void calculateRho(cv::Mat &dist, double &dc, std::vector<double>& rho);
	void sortRho(const std::vector<double>& rho, std::vector<double>& sorted_rho,
	std::vector<int>& ordrho);
	void sortgamma(std::vector<double>& gamma, std::vector<double>& sorted_gamma,
	std::vector<int>& ordgamma);
	void calculateDelta(cv::Mat& dist, std::vector<double>& rho,
	std::vector<double>& sorted_rho, std::vector<int>& ordrho,
	std::vector<double>& delta, std::vector<int>& nneigh,
	std::vector<double>& gamma, std::vector<int>& ordgamma,
	std::vector<double> &loggamma);

	void normrd(std::vector<double> &r, std::vector<double> &d);

	void getrdth(std::vector<double> &r,std::vector<double> &d,int split);
	void fastClust(cv::Mat &dist, std::vector<cluster>& clustResult);
	
	bool isSameOne(cv::Mat &dist, cluster& A, cluster& B);
	cluster mergeCluster(cluster& A, cluster& B);
	void mergeCls(std::vector<cluster> & cls,std::string & Gfile,std::vector<cluster> &newcls);


	void getmeanstd(std::vector<CFace>& cfaces);
	void getPandM(std::vector<cluster>& P,std::vector<cluster>& M);

	void getPuredist(std::vector<CFace> &cfaces);

	void savecluster(std::vector<CFace>& cfaces, std::vector<cluster> &cls, std::string &folderpath);

public:
  //std::ofstream data_analyse;
	int t_sN;
	cv::Mat t_dist;
	double t_dc;
	double t_neighbor_rate;
	std::vector<double> t_rho;
	std::vector<double> t_delta;
	std::vector<double> t_sorted_rho;  //descending 
	std::vector<int> t_rho_ind;      //the indice of descending serials.
	std::vector<double> t_sorted_gamma;
	std::vector<int> t_gamma_ind;
	std::vector<double> t_sorted_loggamma;

	std::vector<int> t_neighbor;    //the neighbor of each one 

	double t_maxrho;
	double t_minrho;
	double t_maxdelta;
	double t_mindelta;
	double t_rhoth;
	double t_deltath;

	int t_clusterN;
	std::vector<int> t_icl;          //i-th cluster 's center id.
	std::vector<datapoint> t_result;
	std::vector<cluster> t_clustresult;
	std::vector<double> t_mean;
	std::vector<double> t_std;
};

}

#endif //FASTCLUSTER_H__