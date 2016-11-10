#include <iostream>
#include <fstream>
#include "ARecog/fastCluster.h"
#include "ARecog/MRECOG.h"
#include "common/fs.hpp"
#include "ARecog/utils.h"

using namespace std;
using namespace cv;
using namespace common;
using namespace fs;

namespace arecog {

bool comp(node x, node y)
{
  return x.rho > y.rho;
}
bool compgamma(node x, node y)
{
  return x.gamma > y.gamma;
}
bool comppics(picture x, picture y)
{

  for (int i = 0; i < x.date.size();)
  {
    if (x.date.at(i) == y.date.at(i))
    {
      i++;
    }
    else
    {
      return x.date.at(i) < y.date.at(i);
    }
  }
  return false;
}
bool comprho(cluster x, cluster y)
{
  return x.centerrho > y.centerrho;
}
bool descend(double x, double y)
{
	return x > y;
}
Cluster::Cluster(int N)
{
	t_sN = N;
	t_dist = Mat::eye(t_sN, t_sN, CV_64FC1);
	t_dc = 0;
	t_neighbor_rate = 0;
	t_rho = {};
	t_delta = {};
	t_sorted_rho = {};  //descending 
	t_rho_ind = {};      //the indice of descending serials.
	t_sorted_gamma = {};
	t_gamma_ind = {};
	t_sorted_loggamma = {};
	t_neighbor = {};    //the neighbor of each one 
	t_maxrho = 0;
	t_minrho = 0;
	t_maxdelta = 0;
	t_rhoth = 0;
	t_deltath = 0;

	t_clusterN = 0;
	t_result = {};
}

int Cluster::AClustProcess_Init(std::vector<CFace>& cfaces, double dc, double rth, double dth)
{
	AClustProcess_GetDist(cfaces, t_dist);  //gen dist
	//getnrate(t_dist);                       //gen neibor rate
	//getDc(t_dist, t_neighbor_rate);           //gen dc
	t_dc = dc;
	t_rhoth = rth;
	t_deltath = dth;
	t_sN = cfaces.size();
	calculateRho(t_dist, t_dc, t_rho);      //gen rho
	sortRho(t_rho, t_sorted_rho, t_rho_ind);         //gen sorted rho and sorted indice
	calculateDelta(t_dist, t_rho, t_sorted_rho,
		t_rho_ind, t_delta, t_neighbor, t_sorted_gamma, t_gamma_ind, t_sorted_loggamma);

	normrd(t_rho, t_delta);
	return 0;
}

int Cluster::AClustProcess_GetDist(vector<CFace>& cfaces, Mat &dist)
{
	int Numface = cfaces.size();
	dist = Mat::zeros(Numface, Numface, CV_64FC1);

	for (int i = 0; i < Numface - 1; i++)
	{
		for (int j = i + 1; j < Numface; j++)
		{
			double disij;
			featureCompare(cfaces[i].faceinfo.feature, cfaces[j].faceinfo.feature, disij);

			dist.at<double>(i, j) = 1 - disij;
			dist.at<double>(j, i) = 1 - disij;  //  cos distance the bigger disij is,the similar ij are.
		}
	}
	return 0;
}

int Cluster::AClustProcess_Clust(std::vector<CFace> &cfaces)
{
	fastClust(t_dist, t_clustresult);
	return 0;
}

void Cluster::fillval(vector<int> &a, int &val)
{
  for (int i = 0; i < a.size(); i++)
  {
    a[i] = val;
  }
}

void Cluster::fillval(vector<double> &a, double &val)
{
  for (int i = 0; i < a.size(); i++)
  {
    a[i] = val;
  }
}

void Cluster::getnrate(Mat &dist)
{
	ofstream dataana("data.txt", ios::app);
	int N = dist.rows;
	int nneigh = 0;
	double averdist = 0.0;
	Mat mean, stddev;
	meanStdDev(dist, mean, stddev);
	double aver = mean.at<double>(0, 0);
	cout << "average distance in C(N,2) " << aver << endl;

	double dc = aver / 2;
	cout << "预估的距离dc为 ：" << dc << endl;
	//data_analyse << "预估的距离dc " << dc << endl;

	for (int i = 0; i < N - 1; i++)
	{
	for (int j = i + 1; j < N; j++)
	{
		if (dist.at<double>(i, j) < dc)
		{
		nneigh++;
		}
	}
	}
	double percents = 100.0*nneigh / (double)(N*(N - 1) / 2);
	cout << "邻居率为 " << percents << " %." << endl;
	//data_analyse << "邻居率 " << percents << endl;
	t_neighbor_rate = percents;
}

void Cluster::getDc(cv::Mat &dist, double& percent)
{
	int Num = dist.rows;
	int N = Num*(Num - 1) / 2;  //所有距离的总数
	if (((percent) >= 100) || ((percent) <= 0))
	{
	cerr << "the percent must be [1-100],2-5 maybe ok." << endl;
	}
	cout << "average percentage of neighbours(hard code) :  " << percent << "% ." << endl;

	int position = (int)(N*percent / 100);
	cv::Mat alldis(1, N, CV_64FC1);

	for (int i = 0; i < Num - 1; i++)
	{
	for (int j = i + 1; j < Num; j++)
	{
		alldis.at<double>(0, (1 + 2 * (Num - 1) - i)*i / 2 + (j - i) - 1) = dist.at<double>(i, j);
	}
	}
	vector<double> sda = (vector<double>)alldis;
	sort(sda.begin(), sda.end());  //ascending
	t_dc = sda[position];
	cout << "Computing Rho with gaussian kernal of radius(dc) : " << t_dc << endl;
}

void Cluster::calculateRho(cv::Mat &dist, double &dc, std::vector<double>& rho)
{
	rho.resize(t_sN);
	int Num = dist.rows;
	double bdouble = 0.0;
	fillval(rho, bdouble);
	//Gaussian kernal
	for (int i = 0; i < Num - 1; i++)
	{
		for (int j = i + 1; j < Num; j++)
		{
			double distmp = dist.at<double>(i, j);
			rho[i] = rho[i] + exp(-pow(distmp / dc, 2));
			rho[j] = rho[j] + exp(-pow(distmp / dc, 2));
		}
	}
}

void Cluster::sortRho(const vector<double>& rho, vector<double>& sorted_rho, vector<int>& ordrho)
{
	sorted_rho.resize(t_sN);
	ordrho.resize(t_sN);
	//sort with descending ,record the indice
	vector<node> _rho(rho.size());
	for (int i = 0; i < rho.size(); i++)
	{
		_rho[i].rho = rho[i];
		_rho[i].idx = i;
	}
	sort(_rho.begin(), _rho.end(), comp);//comp function indicate the descending

	for (int i = 0; i < _rho.size(); i++)
	{
		sorted_rho[i] = _rho[i].rho;
		ordrho[i] = _rho[i].idx;
	}
	t_maxrho = sorted_rho[0];
	t_minrho = sorted_rho[t_sN - 1];
}
void Cluster::sortgamma(vector<double>& gamma, vector<double>& sorted_gamma,
  vector<int>& ordgamma)
{
	assert((sorted_gamma.size() == gamma.size()) && (ordgamma.size() == gamma.size()));
	//将局部密度rho[Num] 按照降序排列，并保留下标号的次序。
	vector<node> _gamma(gamma.size());
	for (int i = 0; i < gamma.size(); i++)
	{
	_gamma[i].gamma = gamma[i];
	_gamma[i].idx = i;
	}
	sort(_gamma.begin(), _gamma.end(), compgamma);//comp函数指明升序还是降序。

	for (int i = 0; i < _gamma.size(); i++)
	{
	sorted_gamma[i] = _gamma[i].gamma;
	ordgamma[i] = _gamma[i].idx;
	}
}

void Cluster::calculateDelta(Mat& dist, vector<double>& rho,vector<double>& sorted_rho, 
							vector<int>& ordrho, vector<double>& delta,vector<int>& nneigh, 
							vector<double>& sorted_gamma,vector<int>& ordgamma,
							vector<double> &loggamma)
{
	
	delta.resize(t_sN);
	nneigh.resize(t_sN);
	sorted_gamma.resize(t_sN);
	ordgamma.resize(t_sN);
	loggamma.resize(t_sN);

	delta[ordrho[0]] = -1.0;
	nneigh[ordrho[0]] = -1; //the biggest point has no neighbor
	double mind;
	minMaxIdx(dist, &mind, &t_maxdelta);  //the biggest distance in dist is maxd.
	for (int ii = 1; ii < t_sN; ii++)
	{
		delta[ordrho[ii]] = t_maxdelta;
		for (int jj = 0; jj < ii; jj++)  //ordrho[jj] has bigger rho than ordrho[ii]
		{
			if (dist.at<double>(ordrho[ii], ordrho[jj]) < delta[ordrho[ii]])
			{
				delta[ordrho[ii]] = dist.at<double>(ordrho[ii], ordrho[jj]); 
				// at the end,delta[ordrho[ii]] get the smallest dist(i,j).
				nneigh[ordrho[ii]] = ordrho[jj];
			}
		}
	}

	delta[ordrho[0]] = t_maxdelta;  //delta all done.

	/*由rho[Num]  和 delta[Num] 计算gamma[Num].*/
	vector<int> ind(rho.size());
	vector<double> gamma(rho.size());
	//double minrho = sorted_rho[sorted_rho.size() - 1];
	//double maxrho = sorted_rho[0];
	//double difrho = maxrho - minrho;
	//double difdelta = maxdel - mindel;
	for (int i = 0; i < t_sN; i++)
	{
		ind[i] = i;
		gamma[i] = 1e-6 + rho[i] * delta[i];
		//gamma[i] = 1e-6 + (rho[i] - minrho) * (delta[i] - mindel) / (difrho*difdelta);
	}
	sorted_gamma = gamma;
	sortgamma(gamma, sorted_gamma, ordgamma);
	for (int i = 0; i < t_sN; i++)
	{
		loggamma[i] = log(sorted_gamma[i]);
	}
}

void Cluster::normrd(std::vector<double> &r, std::vector<double> &d)
{
	vector<double> t_sorted_delta = d;
	sort(t_sorted_delta.begin(),t_sorted_delta.end());
	t_mindelta = t_sorted_delta[0];
	t_maxdelta = t_sorted_delta[t_sN - 1];
	double diffr = t_maxrho - t_minrho;
	double diffd = t_maxdelta - t_mindelta;
	for (int i = 0; i < r.size(); i++)
	{
		//r[i] = (r[i] - t_minrho) / (diffr + 1e-6);
		d[i] = (d[i] - t_mindelta) / (diffd + 1e-6);
	}
}

void Cluster::getrdth(vector<double> &r, vector<double> &d, int split)
{
	vector<vector<double>> rdy(split + 1);
	vector<vector<double>> rdx(split + 1);
	vector<double> rdyth, rdxth;
	for (int i = 0; i < t_sN; i++)
	{
		rdy[int(split*r[i])].push_back(d[i]);         //bullet
		rdx[int(split*d[i])].push_back(r[i]);         //bullet
	}
	for (int i = 0; i < rdy.size(); i++)
	{
		//sort descending
		sort(rdy[i].begin(), rdy[i].end(), descend);
		int Nele = rdy[i].size();
		if (Nele == 0)
			continue;
		Mat _mrdy = Mat(rdy[i]);
		Mat _mean, _std;
		meanStdDev(_mrdy, _mean, _std);
		double orimean = _mean.at<double>(0, 0);
		double oristd = _std.at<double>(0, 0);
		int j = 0;
		if (Nele == 1)
		{
			if (rdy[i][0] > rdyth[rdyth.size() - 1])
			{
				rdyth.push_back(rdy[i][0]);
				//cout << "rdyth " << rdy[i][0] << endl;
			}
			continue;
		}

		while (j < Nele && rdy[i].size()>1)
		{
			rdy[i].erase(rdy[i].begin());//delete the biggest ,jurge the change
			Mat _mrdy = Mat(rdy[i]);
			Mat _mean, _std;
			meanStdDev(_mrdy, _mean, _std);
			double mean = _mean.at<double>(0, 0);
			double std = _std.at<double>(0, 0);
			//cout << "orimean - mean= " << orimean - mean << endl;
			//cout << "(0.02 + 3 * mean) / Nele= " << (0.02 + 3 * mean) / Nele << endl;
			if (orimean - mean < (0.02 + 3 * mean) / (Nele - j))
			{
				rdyth.push_back(rdy[i][0]);
				//cout << "rdyth " << rdy[i][0] << endl;
				//cout << "i****" << i << endl;
				break;
			}
			orimean = mean;
			j++;
		}
		//cout << "i----" << i << endl;
	}

	//
	for (int i = 0; i < rdx.size(); i++)
	{
		//sort descending
		sort(rdx[i].begin(), rdx[i].end(), descend);
		int Nele = rdx[i].size();
		if (Nele == 0)
			continue;
		Mat _mrdx = Mat(rdx[i]);
		Mat _mean, _std;
		meanStdDev(_mrdx, _mean, _std);
		double orimean = _mean.at<double>(0, 0);
		double oristd = _std.at<double>(0, 0);

		if (Nele == 1)
		{
			if (rdx[i][0] > rdxth[rdxth.size() - 1])
			{
				rdxth.push_back(rdx[i][0]);
				//cout << "rdxth " << rdx[i][0] << endl;
			}
			continue;
		}

		int j = 0;
		while (j < Nele && rdx[i].size()>1)
		{
			if (rdx[i][0] < 0.02)
			{
				break;
			}
			rdx[i].erase(rdx[i].begin());//delete the biggest ,jurge the change
			Mat _mrdx = Mat(rdx[i]);
			Mat _mean, _std;
			meanStdDev(_mrdx, _mean, _std);
			double mean = _mean.at<double>(0, 0);
			double std = _std.at<double>(0, 0);

			if (orimean - mean < (0.02 + 3 * mean) / (Nele - j))
			{
				rdxth.push_back(rdx[i][0]);
				cout << "rdyth " << rdx[i][0] << endl;
				cout << "i****" << i << endl;
				break;
			}
			orimean = mean;
			j++;
		}
	}

	sort(rdyth.begin(), rdyth.end());
	sort(rdxth.begin(), rdxth.end());
	for (int i = 0; i < rdyth.size(); i++)
	{
		cout << rdyth[i] << " - ";
	}
	for (int i = 0; i < rdxth.size(); i++)
	{
		cout << rdxth[i] << " + ";
	}
	t_rhoth = rdyth[0];
	t_deltath = max(rdxth[0], 0.02);
}
void Cluster::fastClust(cv::Mat &dist, vector<cluster>& clustResult)
{
	/*start clustering*/
	string rdpath = "rhodelta_" + to_string(rand()) + ".txt";
	ofstream rhodeltatxt(rdpath);
	string loggammapath = "loggamma_" + to_string(rand()) + ".txt";
	ofstream loggammatxt(loggammapath);
	for (int i = 0; i < t_sN; i++)
	{
		rhodeltatxt << i << " " << t_rho[i] << " " << t_delta[i] << endl;
		loggammatxt << t_sorted_loggamma[i] << " " << t_sorted_loggamma[i] << endl;

	}
	rhodeltatxt.close();
	loggammatxt.close();

	t_clusterN = 0;
	vector<int> cl(t_sN);  //初始化cl[Num]，全为-1
	int bint = -1;
	fillval(cl, bint);

	for (int i = 0; i < t_sN; i++)
	{
		if ((t_rho[i] > t_rhoth) && (t_delta[i] > t_deltath))
		{
			cl[i] = t_clusterN;  //第i个样点属于类别NCLUST
			t_icl.push_back(i);  //第NCLUST个类别的中心是样点i.
			t_clusterN++;
		}
	}
	//have found all cluster center
	cout << "NUMBER OF CLUSTERS : " << t_clusterN << endl;
	//data_analyse << "聚类中心总数 " << NCLUST << endl;
	/*开始进行所有样点的分配，归入所有的中心*/
	cout << "Performing assignation" << endl;
	for (int i = 0; i < t_sN; i++)
	{
		if (cl[t_rho_ind[i]] == -1)
		{
			cl[t_rho_ind[i]] = cl[t_neighbor[t_rho_ind[i]]];
		}
	}

	vector<int> halo(t_sN);
	for (int i = 0; i < t_sN; i++)
	{
		halo[i] = cl[i];	
	}

	vector<double> bord_rho(t_clusterN);
	if (t_clusterN > 0)
	{
		for (int i = 0; i < t_clusterN; i++)
		{
			bord_rho[i] = 0;
		}
		for (int i = 0; i < t_sN - 1; i++)
		{
			for (int j = i + 1; j < t_sN; j++)
			{
				if ((cl[i] != cl[j]) && (dist.at<double>(i, j) <= t_dc))
				{
					//i，j的类别不同，且两个点的距离小于截断距离。
					double rho_aver = (t_rho[i] + t_rho[j]) / 2.0;
					if (rho_aver > bord_rho[cl[i]])
					{
						//i所属类别的bord_rho比平均局部密度小,则将bord_rho扩张到rho_aver.
						bord_rho[cl[i]] = rho_aver;
					}
					if (rho_aver > bord_rho[cl[j]])
					{
						//同样，扩张j所属类别的bord_rho.
						bord_rho[cl[j]] = rho_aver;
					}
				}
			}
		}

		for (int i = 0; i < t_sN; i++)
		{
			if (t_rho[i] < bord_rho[cl[i]])
			{
				halo[i] = -1;  
			}
		}
	}

	if (t_clusterN > 0)
	{
		//计算每个类别的nc和nh
		vector<cluster> allclust(t_clusterN);
		vector<int> nc(t_clusterN);
		vector<int> nh(t_clusterN);
		int ling = 0;
		fillval(nc, ling);
		fillval(nh, ling);
		//data_analyse << "类簇大小 ";
		for (int i = 0; i < t_clusterN; i++)
		{
			int ncc = 0;
			int nhh = 0;
			for (int j = 0; j < t_sN; j++)
			{
				if (cl[j] == i)
				{
					ncc++;
				}
				if (halo[j] == i)
				{
					nhh++;
				}
			}
			nc[i] = ncc;
			nh[i] = nhh;
			cout << "CLUSTER : " << i << "  CENTER: " << t_icl[i] << " ELEMENT: " << nc[i]
			<< "  CORE: " << nh[i] << "  HALO: " << nc[i] - nh[i] << endl;
			//data_analyse << nc[i] << " ";
		}
		//data_analyse << endl;
		for (int i = 0; i < t_clusterN; i++)
		{
			allclust[i].classid = i;
			allclust[i].centerid = t_icl[i];
			allclust[i].nelement = nc[i];
			allclust[i].ncore = nh[i];
			allclust[i].nhalo = nc[i] - nh[i];
			allclust[i].centerrho = t_rho[t_icl[i]];
		}
		for (int i = 0; i < t_sN; i++)
		{
			int nclass = cl[i];
			allclust[nclass].elements.push_back(i);
		}

		ofstream dist_center_ij("dist_center_ij.txt");

		//dist_center_ij << "dc = " << dc << "\n";
		for (int i = 0; i < t_clusterN; i++)
		{
			for (int j = 0; j < t_clusterN; j++)
			{
				double dist_clust_ij = dist.at<double>(allclust[i].centerid, allclust[j].centerid);
				dist_center_ij << dist_clust_ij;
				dist_center_ij << "  ";
			}
			dist_center_ij << "\n";
		}
		dist_center_ij.close();
		clustResult = allclust;
	}
	else
	{
		//没有聚类中心 各自为一个类别
		for (int i = 0; i < t_sN; i++)
		{
			cluster cresult = { i, i, 1, 1, 1, t_rho[i], {i} };
			clustResult.push_back(cresult);
		}
	}


}

void Cluster::getmeanstd(vector<CFace>& cfaces)
{
	t_mean.resize(t_clusterN);
	t_std.resize(t_clusterN);
	for (int i = 0; i < t_clusterN; i++)
	{
		cluster tmpc = t_clustresult[i];
		int M = tmpc.elements.size();
		vector<double> disteach;
		for (int p = 0; p < M; p++)
		{
			for (int q = p + 1; q < M; q++)
			{
				double tmpd = 0.;
				featureCompare(cfaces[tmpc.elements[p]].faceinfo.feature, cfaces[tmpc.elements[q]].faceinfo.feature, tmpd);
				disteach.push_back(1 - tmpd);
			}
		}
		Mat _d = Mat(disteach);
		Mat _mean, _std;
		meanStdDev(_d, _mean, _std);
		t_mean[i] = _mean.at<double>(0, 0);
		t_std[i] = _std.at<double>(0, 0);
	}
}

void Cluster::getPandM(vector<cluster>& P,vector<cluster>& M)
{
	P.clear();
	M.clear();

	for (int i = 0; i < t_clusterN; i++)
	{
		if (t_clustresult[i].elements.size() == 1)
		{
			P.push_back(t_clustresult[i]);
		}
		else
		{
			if (t_mean[i] < 0.3&&t_std[i] < 0.2)
			{
				P.push_back(t_clustresult[i]);
			}
			else
			{
				M.push_back(t_clustresult[i]);
			}
		}
	}
}

void Cluster::savecluster(vector<CFace>& cfaces, vector<cluster> &cls, string &folderpath)
{
	if (!fs::IsExists(folderpath))
		fs::MakeDir(folderpath);
	for (int i = 0; i < cls.size(); i++)
	{
		cluster tmpc = cls[i];
		string dirinclass = folderpath + "/" + to_string(tmpc.classid)
			+ "_num= " + to_string(tmpc.nelement)
			+ "_mean= " + to_string(t_mean[tmpc.classid])
			+ "_std= " + to_string(t_std[tmpc.classid]);
		if (!fs::IsExists(dirinclass))
		{
			fs::MakeDir(dirinclass);
		}
		for (int j = 0; j < tmpc.elements.size(); j++)
		{
			string facepath = cfaces[tmpc.elements[j]].srcpath;
			Mat face = imread(facepath);
			string savepath;
			if (tmpc.centerid == tmpc.elements[j])
			{
				string name = facepath.substr(facepath.find_last_of("/") + 1);
				name = name.substr(0, max(name.find_last_of(".jpg"), name.find_last_of(".JPG")) - 3) + "_center.jpg";
				savepath = dirinclass + "/" + name;
			}
			else
			{
				savepath = dirinclass + "/" +
					facepath.substr(facepath.find_last_of("/") + 1);
			}
			cv::imwrite(savepath, face);
		}
	}
}

void Cluster::getPuredist(vector<CFace> &cfaces)
{
	Mat pdist = Mat::zeros(cfaces.size(), cfaces.size(), CV_64FC1);
	for (int i = 0; i < cfaces.size(); i++)
	{
		for (int j = i + 1; j < cfaces.size(); j++)
		{
			double tmpd;
			featureCompare(cfaces[i].faceinfo.feature, cfaces[j].faceinfo.feature, tmpd);
			pdist.at<double>(i, j) = 1 - tmpd;
			pdist.at<double>(j, i) = 1 - tmpd;
			//if (tmpd > 0.7)
			//{
			//	cout << i << " should combine " << j << endl;
			//}
		}
	}

	ofstream pd("pd.txt");
	if (pd.is_open())
	{
		for (int i = 0; i < pdist.rows; i++)
		{
			for (int j = 0; j < pdist.cols; j++)
			{
				pd << pdist.at<double>(i, j) << " ";
			}
			pd << endl;
		}
	}
	pd.close();
}
bool Cluster::isSameOne(cv::Mat &dist, cluster& A, cluster& B)
{
  int M = A.nelement;
  int N = B.nelement;
  double ABsim = 0;
  for (int i = 0; i < M; i++)
  {
    for (int j = 0; j < N; j++)
    {
      double simij = 1 - dist.at<double>(A.elements[i], B.elements[j]);
      ABsim += simij;
    }
  }
  if (ABsim / (M*N) > 0.8)
  {
    return true;
  }
  return false;
}

cluster Cluster::mergeCluster(cluster& A, cluster& B)
{
  if (A.nelement == 0)
	  return B;
  if (B.nelement == 0)
	  return A;
  A.classid = B.classid;
  A.ncore = A.ncore + B.ncore;
  A.nhalo = A.nhalo + B.nhalo;
  A.nelement = A.nelement + B.nelement;

  A.elements.insert(A.elements.end(), B.elements.begin(), B.elements.end());
  return A;
}

void Cluster::mergeCls(vector<cluster> & cls,string & Gfile,vector<cluster> &newcls)
{
	int N = cls.size();
	vector<vector<bool>> manner(N, vector<bool>(N, false));
	for (int i = 0; i < N; i++)
	{
		for (int j = i+1; j < N; j++)
		{
			bool ans = isSameOne(t_dist, cls[i], cls[j]);
			manner[i][j] = ans;
			manner[j][i] = ans;
		}
	}
	vector<set<int>> G(cls.size());
	for (int i = 0; i < N; )
	{
		for (int j = 0; j < N; j++)
		{
			if (manner[i][j])
			{
				G[i].insert(j);
			}
		}
		i++;
	}
	//find all circle with 3-length.
	vector<set<int>> circles = findcircleswithlen(G, G.size(), 3);   
	ofstream Gnet(Gfile);
	for (int i = 0; i < circles.size(); i++)
	{
		Gnet << "merge cluster : ";
		for (set<int>::iterator it = circles[i].begin(); it != circles[i].end(); it++)
		{
			Gnet << *it << " ";
		}
		Gnet << endl;
	}
	Gnet.close();
	vector<set<int>> mergedcircle;
	
	//merge node 0;node i which nor merged.
	vector<bool> mergedflag(cls.size(), false);
	for (int i = 0; i < cls.size(); i++)
	{
		int ind = i;
		//int nodeid = cls[ind].classid;
		if (!mergedflag[i])
		{
			//may it's neighbor has been taged.
			for (int j = 0; j < circles.size(); j++)
			{
				if (circles[j].find(ind) != circles[j].end())
				{
					for (set<int>::iterator it = circles[j].begin(); it != circles[j].end(); it++)
					{
						if (mergedflag[*it])
						{
							mergedflag[i] = true;
							break;
						}
					}
				}
				if (mergedflag[i])
				{
					break;
				}
			}
			if (mergedflag[i])
			{
				i--;
				continue;
			}

			mergedcircle.push_back({ i });
			int len = mergedcircle.size();
			//merge all circles which has ind-node into mergedcircle[len-1].
			for (int j = 0; j < circles.size(); j++)
			{
				if (circles[j].find(ind) != circles[j].end())
				{
					//all nodes in circles[j] should merge in ind-cluster
					for (set<int>::iterator it = circles[j].begin(); it != circles[j].end(); it++)
					{
						mergedcircle[len - 1].insert(*it);
					}
					circles[j].clear();
				}
			}


			//tag all node in mergedcircle[len-1] to true. no circle mean mergedcircle[len-1] has i only
			for (set<int>::iterator it = mergedcircle[len - 1].begin(); it != mergedcircle[len - 1].end(); it++)
			{
				mergedflag[*it] = true;
			}

		}
		else
		{
			//1.find which circle ind belong
			int id_belong = 0;
			for (int i = 0; i < mergedcircle.size(); i++)
			{
				if (mergedcircle[i].find(ind) != mergedcircle[i].end())
				{
					id_belong = i;
					break;
				}
			}

			//find the neighbor node,that dosent been tag.
			for (int j = 0; j < circles.size(); j++)
			{
				if (circles[j].find(ind) != circles[j].end())
				{
					//all nodes in circles[j] should merge in ind-cluster
					for (set<int>::iterator it = circles[j].begin(); it != circles[j].end(); it++)
					{
						if (!mergedflag[*it])
						{
							mergedcircle[id_belong].insert(*it);
							mergedflag[*it] = true;
						}
						else
						{
							//has been taged,check if equal to id_belong.
							int id_belong2 = 0;
							for (int i = 0; i < mergedcircle.size(); i++)
							{
								if (mergedcircle[i].find(*it) != mergedcircle[i].end())
								{
									id_belong2 = i;
									break;
								}
							}
							if (id_belong2 != id_belong)
							{
								//merge mergedcircle[id_belong] and mergedcircle[id_belong2]
								if (id_belong < id_belong2)
								{
									mergedcircle[id_belong].insert(mergedcircle[id_belong2].begin(), mergedcircle[id_belong2].end());
									mergedcircle.erase(mergedcircle.begin() + id_belong2);
								}
								else
								{
									//remove one mergedcircle,but impact the id_belong.we should update the id_belong.
									mergedcircle[id_belong2].insert(mergedcircle[id_belong].begin(), mergedcircle[id_belong].end());
									mergedcircle.erase(mergedcircle.begin() + id_belong);
									id_belong = id_belong2;
								}
							}
						}
					}
					circles[j].clear();
				}
			}
		}
	}

	//for (int i = 0; i < circles.size(); i++)
	//{
	//	set<int> tmpcircle = {};
	//	for (set<int>::iterator it = circles[i].begin(); it != circles[i].end(); it++)
	//	{
	//		tmpcircle.insert(*it);
	//	}
	//	set<int> tmpp = tmpcircle;
	//	for (set<int>::iterator it = tmpp.begin(); it != tmpp.end(); it++)
	//	{
	//		for (int j = i + 1; j < circles.size(); j++)
	//		{
	//			if (circles[j].find(*it) != circles[j].end())
	//			{
	//				//find same node in circles[j].merge all node in circles[j] and delate it
	//				for (set<int>::iterator it = circles[j].begin(); it != circles[j].end(); it++)
	//				{
	//					tmpcircle.insert(*it);
	//				}
	//				circles[j].clear();         //logitic delete.
	//			}
	//		}
	//	}
	//	if (!tmpcircle.empty())
	//	{
	//		mergedcircle.push_back(tmpcircle);
	//	}
	//}

	//assign newcls via mergedcircles
	for (int i = 0; i < mergedcircle.size(); i++)
	{
		cluster tmp = {};
		for (set<int>::iterator it = mergedcircle[i].begin(); it != mergedcircle[i].end(); it++)
		{
			tmp = mergeCluster(tmp,cls[*it]);
		}
		newcls.push_back(tmp);
	}
	
}



}  // namespace arecog
