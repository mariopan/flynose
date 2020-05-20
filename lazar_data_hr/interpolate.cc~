#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

using namespace std;

int main(int argc, char *argv[])
{
  char buf[1024];
  double tmp;
  vector<double> x,y;
  double xn,yn,sum,wght;
  int steps, pos;
  
  if (argc != 6) {
    cerr << "usage: interpolate <infile> <xmin> <xmax> <dt> <outfile>" << endl;
    exit(1);
  }

  ifstream is(argv[1]);
  double xmin= atof(argv[2]);
  double xmax= atof(argv[3]);
  double dt= atof(argv[4]);
  if (!is.good()) {
    cerr << "error opening input file" << endl;
    exit(1);
  }
  ofstream os(argv[5]);
  is.getline(buf, 1024, '\n');
  is >> tmp;
  while (is.good()) {
    x.push_back(tmp);
    is >>tmp;
    y.push_back(tmp);
    is >> tmp;
  }
  pos= 1;
  steps= static_cast<int>((xmax-xmin)/dt);
  for (int i= 0; i < steps; i++) {
    xn= i*dt;
    while (pos < x.size()-1 && x[pos] <= xn) pos++;
    yn= y[pos-1]+(y[pos]-y[pos-1])/(x[pos]-x[pos-1])*(xn-x[pos-1]);
    os << xn << " " << yn << endl;
  }
  os.close();
}

	
