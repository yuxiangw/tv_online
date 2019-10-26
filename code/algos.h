#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include<vector>
#include<cmath>
#include<algorithm>

#include<fstream>
#include<iomanip>


using namespace std;

namespace py = pybind11;


typedef unsigned long long int t_ulong;

class Arrows{
 public:
  unsigned long long int sim_steps;
  float sigma;
  std::vector<double> haar_coeff, thresholded_coeff;
  double thresholded_norm_squared, uthresh;
  //std::ofstream my_file;
  // bool first;

  Arrows(unsigned long long int steps, float stddev);
void set_uthresh(t_ulong steps, int m_f);
  void restarted_averages_incr(vector<double>& y,\
			       double c_i, t_ulong steps, \
			       double uconst, double rconst,vector<double>& estimates);
  void update_coeff_new_pivot(t_ulong new_pivot, double y);
  void update_coeff_old_pivot(t_ulong t, double y, t_ulong pivot );

  
};
