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

class Cluster{

public:
	//load the dataset to clust.
	Cluster(string dataset_filepath);
	
	int init();
	int fclust();

private:
	string dataset_file;
	int dataSize;                      //dataset's size 
	double dc;                         //the dc found
	vector<vector<double>> t_dist;     //
	double t_neighbor_rate;            //2%
	std::vector<double> t_rho;
	std::vector<double> t_delta;

};

#endif //FASTCLUSTER_H__