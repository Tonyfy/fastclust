#include "fastCluster.h"
#include <fstream>
#include <string>
#include <assert.h>

using namespace std;

bool comp(node x, node y)
{
	return x.rho > y.rho;
}

Cluster::Cluster(string dataset_filepath)
{
	dataset_file = dataset_filepath;
	t_neighbor_rate = 0.03;
	init();
}

int Cluster::init()
{
	ifstream datafile(dataset_file,ios::in);
	if(datafile.is_open())
	{
		data.clear();
		string line;
		int seq = 0;
		while (getline(datafile, line))
		{
			//get the pt struct.
			int split1 = line.find_first_of("	");
			string line1 = line.substr(split1+1);
			int split2 = line1.find_first_of("	");
			string line2 = line1.substr(split2+1);

			string sx = line.substr(0, split1);
			string sy = line1.substr(0, split2);
			string sreallabel = line2;

			pt tmp_pt;

			tmp_pt.x = atof(sx.c_str());
			tmp_pt.y = atof(sy.c_str());
			tmp_pt.reallabel = atoi(sreallabel.c_str());
			tmp_pt.id = seq++;
			tmp_pt.label = -1;
			tmp_pt.isCenter = false;
			data.push_back(tmp_pt);
		}
		dataSize = seq;
	}
	else {
		cerr << "dataset_file not exists." << endl;
	}
	datafile.close();
	return 0;
}

int Cluster::getdist()
{
	t_dist.resize(dataSize,vector<double>(dataSize,0.0));
	for (int i = 0; i < data.size(); i++)
	{
		for (int j = i; j < data.size(); j++)
		{
			double distij = sqrt(pow(data[i].x - data[j].x, 2) + pow(data[i].y - data[j].y, 2));
			t_dist[i][j] = distij;
			t_dist[j][i] = distij;
		}
	}
	return 0;
}

int Cluster::getdc()
{
	assert(t_dist.size() == dataSize&&t_dist[0].size() == dataSize&&dataSize >= 2);
	vector<double> alldist;
	for(int i=0;i<t_dist.size()-1;i++)
	{
		for(int j=i+1;j<t_dist.size();j++)
		{
			alldist.push_back(t_dist[i][j]);
		}
	}
	sort(alldist.begin(), alldist.end());
	dc = alldist[static_cast<int>(alldist.size())*t_neighbor_rate];
	maxdist = alldist[alldist.size() - 1];
	return 0;
}

int Cluster::getrho()
{
	t_rho.resize(dataSize,0);
	for(int i=0;i<t_dist.size()-1;i++)
	{
		for(int j =i+1;j<t_dist.size();j++)
		{
			double distij = t_dist[i][j];
			t_rho[i] = t_rho[i] + exp(-pow(distij / dc, 2));
			t_rho[j] = t_rho[j] + exp(-pow(distij / dc, 2));
		}
	}
	//get ordered rho.
	t_orderrho.resize(dataSize);
	for(int i =0;i<t_rho.size();i++)
	{
		t_orderrho[i].id = i;
		t_orderrho[i].rho = t_rho[i];
	}
	sort(t_orderrho.begin(), t_orderrho.end(), comp);

	return 0;
}

int Cluster::getdelta()
{
	t_delta.resize(dataSize, 0);
	t_neighbor.resize(dataSize, -1);
	t_delta[t_orderrho[0].id] = -1.0;
	t_neighbor[t_orderrho[0].id] = -1;  //the most center point has no neighbor
	
	for (int i=0;i<dataSize;i++)
	{
		t_delta[t_orderrho[i].id] = maxdist;
		for(int j =0;j<i;j++)
		{
			if(t_dist[t_orderrho[i].id][t_orderrho[j].id]<t_delta[t_orderrho[i].id])
			{
				t_delta[t_orderrho[i].id] = t_dist[t_orderrho[i].id][t_orderrho[j].id];
				t_neighbor[t_orderrho[i].id] = t_orderrho[j].id;
			}
		}
	}
	t_delta[t_orderrho[0].id] = maxdist;

	return 0;
}

int Cluster::assign()
{
	//start clustering
	double t_rhoTH = t_maxrho / 2;
	double t_deltaTH = maxdist / 8;
	
	t_clusterNum = 0;
	cl.resize(dataSize, -1);
	icenter_of_class.clear();
	for(int i=0;i<dataSize;i++)
	{
		if (t_rho[i] > t_rhoTH&&t_delta[i] > t_deltaTH)
		{
			cl[i] = t_clusterNum;
			icenter_of_class.push_back(i);
			t_clusterNum++;
		}
	}

	cout << "Performing assignment of each points." << endl;
	for(int i =0;i<dataSize;i++)
	{
		if(cl[t_orderrho[i].id]==-1)
		{
			cl[t_orderrho[i].id] = cl[t_neighbor[t_orderrho[i].id]];
		}
	}


	for(int i=0;i<dataSize;i++)
	{
		data[i].label = cl[i];
	}
	for(int i=0;i<icenter_of_class.size();i++)
	{
		int centerIDofclsI = icenter_of_class[i];
		data[centerIDofclsI].isCenter = true;
	}

	ofstream result("../../test/result.txt", ios::out);
	if(result.is_open())
	{
		for(int i=0;i<data.size();i++)
		{
			result << data[i].x << " " << data[i].y << " " << data[i].reallabel << " " << data[i].label << endl;
		}
	}
	else { cerr << "save data failed." << endl; }
	result.close();
	return 0;
}

int Cluster::fclust()
{
	getdist();
	getdc();
	getrho();
	getdelta();
	assign();
	return 0;
}
