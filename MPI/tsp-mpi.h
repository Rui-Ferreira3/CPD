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
void share_inputs(int rank);
void create_tasks(int num_processes, PriorityQueue<QueueElem> &startQueue);
void split_tasks(int rank, PriorityQueue<QueueElem>&startElems, PriorityQueue<QueueElem> &myQueue, MPI_Datatype elem_type, int elementPerProcess);

pair<vector <int>, double> tsp(PriorityQueue<QueueElem> &myQueue, int rank, MPI_Datatype elem_type);
void create_children(QueueElem &myElem, PriorityQueue<QueueElem> &myQueue, vector<pair<double,double>> &mins);

MPI_Datatype create_MPI_type();
void send_element(int dest, int tag, QueueElem elem, MPI_Datatype elem_type);
QueueElem recv_element(int source, int tag, MPI_Datatype elem_type);

void send_BestTourCost(int rank);
double update_BestTour(int rank, vector <int> &BestTour);

void redistribute_elements(PriorityQueue<QueueElem> &myQueue, int rank, MPI_Datatype elem_type);
void get_elements(PriorityQueue<QueueElem> &myQueue, int rank, MPI_Datatype elem_type);

vector<pair<double,double>> get_mins();
double initialLB(vector<pair<double,double>> &mins);
double calculateLB(vector<pair<double,double>> &mins, int f, int t, double LB);

pair<vector <int>, double> get_results(int rank, pair<vector<int>, double> results);
void print_result(vector <int> BestTour, double BestTourCost);

// global variables
double BestTourCost;
int numCities, numRoads;
vector <vector <double>> distances;
int num_processes;