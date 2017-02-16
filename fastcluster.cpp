#include "fastCluster.h"
#include <fstream>
#include <string>

using namespace std;

Cluster::Cluster(string dataset_filepath)
{
	dataset_file = dataset_filepath;
}

int Cluster::init()
{
	ifstream data(dataset_file,ios::in);
	if(data.is_open())
	{
		string pt;
		while (getline(data,pt))
		{
			//todo 
			//get the pt struct.
		}
	}
	else {
		cerr << "dataset_file not exists." << endl;
	}
	return 0;
}
