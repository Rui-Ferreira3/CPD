#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <utility>

using namespace std;

struct QueueElem{
  vector<int> tour;
  double cost;
  double bound;
  int length;
  int node;
  friend bool operator>(const QueueElem &left, const QueueElem &right) {
    if(left.bound == right.bound)
      return left.node > right.node;
    return left.bound > right.bound;
  }
};

void printQueueElem(QueueElem myElem) {
    cout << "Node: " << myElem.node << endl;
    cout << "Tour: ";
    for(int i=0; i<myElem.length; i++) {
        cout << myElem.tour.at(i) << " ";
    }
    cout << endl;
    cout << "Cost: " << myElem.cost << endl;
    cout << "Bound: " << myElem.bound << endl;
    cout << "Lenght: " << myElem.length << endl;
}