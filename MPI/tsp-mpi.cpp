#include "tsp-mpi.h"

#define NUM_SWAPS 3
#define NUM_ITERATIONS 30

int main(int argc, char *argv[]) {
    double exec_time;

    // omp_set_num_threads(2);
    
    int num_processes, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if(rank == 0) {
        parse_inputs(argc, argv);
    }

    //MPI_Bcast sends the message from the root process to all other processes
    MPI_Bcast(&numCities, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&numRoads, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&BestTourCost, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if(rank != 0) {
        distances.resize(numCities);
        for(int i=0; i<numCities; i++) {
            distances[i].resize(numCities);
        }
    }

    MPI_Bcast(&distances[0][0], distances.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    vector<QueueElem> startElems;
    if(rank == 0)
        startElems = split_work(num_processes);

    // create an MPI data type for QueueElem
    QueueElem elem = startElems[0];
    MPI_Datatype elem_type;
    int block_lengths[5] = { 1, 1, 1, 1, 1 };
    MPI_Datatype types[5] = { MPI_INT, MPI_DOUBLE, MPI_DOUBLE, MPI_INT, MPI_INT };
    MPI_Aint displacements[5];
    MPI_Aint start_address, address;
    MPI_Get_address(&elem, &start_address);
    MPI_Get_address(&elem.tour[0], &address);
    displacements[0] = address - start_address;
    MPI_Get_address(&elem.cost, &address);
    displacements[1] = address - start_address;
    MPI_Get_address(&elem.bound, &address);
    displacements[2] = address - start_address;
    MPI_Get_address(&elem.length, &address);
    displacements[3] = address - start_address;
    MPI_Get_address(&elem.node, &address);
    displacements[4] = address - start_address;
    MPI_Type_create_struct(5, block_lengths, displacements, types, &elem_type);
    MPI_Type_commit(&elem_type);

    if (rank == 0) {
        // send the array of QueueElem data to process 1
        MPI_Send(&elem, 1, elem_type, 1, 0, MPI_COMM_WORLD);
    } else if (rank == 1) {
        // receive the array of QueueElem data from process 0
        QueueElem myElem;
        MPI_Recv(&recv_elems, 1, elem_type, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printQueueElem(myElem);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // calculate tsp
    double start_time = MPI_Wtime();
    pair<vector<int>, double> results = tsp();
    double end_time = MPI_Wtime();


   //  print_result(best_result.first, best_result.second);

    MPI_Finalize();
    return 0;
}

void parse_inputs(int argc, char *argv[]) {
    string line;
    ifstream myfile (argv[1]);

    int row, col;
    double val;

    if(argc-1 != 2)
        exit(-1);

    if (myfile.is_open()){
        getline(myfile, line);
        sscanf(line.c_str(), "%d %d", &numCities, &numRoads);
    }else
        exit(-1);

    for(int i=0; i<numCities; i++) {
        vector <double> ones(numCities, -1.0);
        distances.push_back(ones);
    }
    
    if (myfile.is_open()){
        while (getline(myfile, line)) {
            sscanf(line.c_str(), "%d %d %lf", &row, &col, &val);
            distances[row][col] = val;
            distances[col][row] = val;
        }
        myfile.close();
    }else 
        exit(-1);

    BestTourCost = atof(argv[2]);
}

void print_result(vector <int> BestTour, double BestTourCost) {
    if(BestTour.size() != numCities+1) {
        cout << "NO SOLUTION" << endl;
    } else {
        cout.precision(1);
        cout << fixed << BestTourCost << endl;
        for(int i=0; i<numCities+1; i++) {
            cout << BestTour[i] << " ";
        }
        cout << endl;
    }
}

pair<vector <int>, double> tsp() {
    vector <int> BestTour = {0};
    
    return make_pair(BestTour, BestTourCost);
}

vector<QueueElem> split_work(int num_processes) {
    vector<pair<double,double>> mins = get_mins();

    PriorityQueue<QueueElem> startQueue;
    startQueue.push({{0}, 0.0, initialLB(mins), 1, 0});

    while(startQueue.size() < num_processes) {
        QueueElem myElem = startQueue.pop();

        bool visitedCities[numCities] = {false};
        for (int city : myElem.tour) {
            visitedCities[city] = true;
        }

        for(int v=0; v<numCities; v++) {
            double dist = distances[myElem.node][v];
            if(dist>0 && !visitedCities[v]) {
                double newBound = calculateLB(mins, myElem.node, v, myElem.bound);
                vector <int> newTour = myElem.tour;
                newTour.push_back(v);
                startQueue.push({newTour, myElem.cost + dist, newBound, myElem.length+1, v});
            }
        }
    }

    vector<QueueElem> startElems;
    startElems.reserve(startQueue.size());
    while (!startQueue.empty()){
        startElems.push_back(startQueue.pop());
    }
    
    return startElems;
}

vector<pair<double,double>> get_mins() {
    vector<pair<double,double>> mins;
    mins.reserve(numCities);

    for (int i=0; i<numCities; i++) {
        double min1 = BestTourCost;
        double min2 = BestTourCost;
        for (int j=0; j<numCities; j++) {
            double dist = distances[i][j];
            if(dist > 0) {
                if(dist <= min1) {
                    min2 = min1;
                    min1 = dist;
                }else if(dist <= min2) {
                    min2 = dist;
                }
            }
        }
        mins[i] = make_pair(min1, min2);
    }

    return mins;
}

double initialLB(vector<pair<double,double>> &mins) {
    double LB=0;

    for(int i=0; i<numCities; i++) {
        LB += (mins[i].first + mins[i].second)/2;
    }
    return LB;
}

double calculateLB(vector<pair<double,double>> &mins, int f, int t, double LB) {
    double cf, ct;
    double directCost = distances[f][t];

    if(distances[f][t] <= 0)
        exit(-1);

    if(directCost >= mins[f].second) {
        cf = mins[f].second;
    }else
        cf = mins[f].first;

    if(directCost >= mins[t].second) {
        ct = mins[t].second;
    }else
        ct = mins[t].first;

    return LB + directCost - (cf+ct)/2;
}