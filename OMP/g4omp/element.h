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

void printQueueElem(QueueElem myElem);