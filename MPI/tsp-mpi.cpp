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

    // split initial workload
    int position = 0;
    int bufSize = sizeof(QueueElem);
    char* buffer = new char[bufSize];
    vector<QueueElem> startElems;
    QueueElem myElem;
    if(rank == 0) {
        startElems = split_work(num_processes);
        
        myElem = startElems[0];
        MPI_Pack(&myElem, sizeof(QueueElem), MPI_BYTE, buffer, bufSize, &position, MPI_COMM_WORLD);

        // Send the buffer to the receiving process
        /*for loop to send to all processes*/
        MPI_Send(&buffer, bufSize, MPI_PACKED, 1, 0, MPI_COMM_WORLD);
    }else {
        MPI_Recv(&buffer, bufSize, MPI_PACKED, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Unpack(myElem, bufSize, &position, &buffer[0], bufSize, MPI_BYTE, MPI_COMM_WORLD);
        
        printQueueElem(myElem);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // int elemsPerProcess = startElems.size() / num_processes;
    // vector<QueueElem> myElems;

    // QueueElem data[]

    // MPI_Scatter(&startElems[0], elemsPerProcess, MPI_QueueElem,
    //             &myElems[0], elemsPerProcess, MPI_QueueElem,
    //             0, MPI_COMM_WORLD);

    // PriorityQueue<QueueElem> myQueue;
    // while(!myElems.empty()) {
    //     myQueue.push(myElems[-1]);
    //     myElems.pop_back();
    // }
    // if(rank == 1)
    //     myQueue.print(printQueueElem);

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