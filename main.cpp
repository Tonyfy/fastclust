#include "fastCluster.h"

int main(int argc, char* argv[])
{
	string dataset_filepath = "../aggregation.txt";
	Cluster cler(dataset_filepath);

	cler.fclust();
	return 0;	
}
