#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <vector>
#include <utility>

#include <omp.h>
#include <mpi.h>

using namespace std;

#include "queue.hpp"
#include "element.cpp"

void parse_inputs(int argc, char *argv[]);
void print_result(vector <int> BestTour, double BestTourCost);
pair<vector <int>, double> tsp();

PriorityQueue<QueueElem> split_work(int num_processes);
vector<pair<double,double>> get_mins();
double initialLB(vector<pair<double,double>> &mins);
double calculateLB(vector<pair<double,double>> &mins, int f, int t, double LB);

// global variables
double BestTourCost;
int numCities, numRoads;
vector <vector <double>> distances;