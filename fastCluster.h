/*
fast clustering by find peak density.
input: C(N,2) of distance of each samples.
output:the label of each samples,and whether belongs to a cluster center.

we load 2-d data from aggregation dataset, which is wide-used unbalance dataset.
see here of https://cs.joensuu.fi/sipu/datasets/Aggregation.txt

Author£ºfandy
Date£º2016.1
*/

#ifndef FASTCLUSTER_H__
#define FASTCLUSTER_H__

#include <iostream>
#include <vector>
#include <array>

using namespace std;

struct pt
{
	double x;
	double y;
	int reallabel;
	int label;
	int id;
	bool isCenter;
};

struct node
{
	int id;
	double rho;
	double delta;
};

bool comp(node x, node y);


class Cluster{

public:
	//load the dataset to clust.
	Cluster(string dataset_filepath);
	int init();
	//use the 2-D points to get distance matrix
	int fclust();
	int getdist();
	int getdc();
	int getrho();
	int getdelta();
	int assign();

private:
	string dataset_file;
	int dataSize;                      //dataset's size 
	vector<pt> data;
	double dc;                         //the dc found
	vector<vector<double>> t_dist;     //
	double t_neighbor_rate;            //2%
	double maxdist;

	vector<double> t_rho;
	vector<node> t_orderrho;
	vector<double> t_delta;

	vector<int> t_neighbor;
	double t_maxrho;
	double t_minrho;

	int t_clusterNum;
	vector<int> cl;
	vector<int> icenter_of_class;
};

#endif //FASTCLUSTER_H__