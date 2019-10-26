#include "algos.h"

#include<omp.h>



vector<double> shoot_arrows(vector<double> y, float sigma, double tv,\
			    double uconst, double rconst);

Arrows::Arrows(unsigned long long int sim_steps, float stddev){
  sigma = stddev;

  haar_coeff.assign(sim_steps, 0);
  thresholded_coeff.assign(sim_steps, 0);

  thresholded_norm_squared = 0;
  //my_file.open ("tnorms.txt");
  // first = true;
}

// sets soft-thresholding factor
void Arrows::set_uthresh(t_ulong sim_steps, int m_f){
  uthresh = sigma*sqrt(m_f*log(sim_steps));
  //my_file<<"setting uthresh to "<<uthresh<<" "<<m_f<<endl;
}


void Arrows::update_coeff_old_pivot(t_ulong t, double y, t_ulong pivot ){
  double scale = sqrt(pivot);
  haar_coeff[0] = haar_coeff[0] + (y/scale);

  t_ulong k;
  for(k=0; k<t_ulong(log2(pivot/2)+1);k++){
    t_ulong coeff_idx = t_ulong(1<<k) + t_ulong(floor(1.0*t*(1<<k)/pivot));
    int wavelet_sign = 1;

    t_ulong idx = t_ulong(floor(1.0*t*(1<<k)/pivot));
    double delta = 1.0*pivot/((1<<k));
    double start = idx * delta;
    double end = (idx+1)*delta;

    if(t >= (start+end)/2)
      wavelet_sign = -1;

    scale = sqrt(delta);

    haar_coeff[coeff_idx] += (wavelet_sign*y/scale);

    // soft-thresholding
    double t_soft;
    if( haar_coeff[coeff_idx] < (-1*uthresh))
      t_soft = haar_coeff[coeff_idx] + uthresh;
    else if( haar_coeff[coeff_idx] > uthresh)
      t_soft = haar_coeff[coeff_idx] - uthresh;
    else
      t_soft = 0;

    thresholded_norm_squared = \
      thresholded_norm_squared + (-abs(thresholded_coeff[coeff_idx]) + abs(t_soft))*pow(2,(floor(log2(idx)) + 1)/2);
    // my_file<<"old "<<thresholded_norm_squared<<" "<<uthresh<<endl;

    thresholded_coeff[coeff_idx] = t_soft;
   
  }
}

void Arrows::update_coeff_new_pivot(t_ulong new_pivot, double y){
  vector<double> new_coeff(new_pivot,0);
  t_ulong old_pivot = new_pivot/2;

  for(t_ulong i =1; i<old_pivot; i++){
    t_ulong new_pos = t_ulong(pow(2,floor(log2(i))+1) + i - pow(2,floor(log2(i))));
    new_coeff[new_pos] = haar_coeff[i];
  }

  double old_scale = sqrt(old_pivot);
  double new_scale = sqrt(new_pivot);

  new_coeff[0] = ((haar_coeff[0]*old_scale) + y)/new_scale;
  new_coeff[1] = ((haar_coeff[0]*old_scale) - y)/new_scale;

  t_ulong pos = 3;
  t_ulong ctr = 0;

  while(pos < new_pivot){
    double scale = sqrt(new_pivot/pow(2,ctr+1));
    new_coeff[pos] = y/scale;
    ctr = ctr + 1;
    pos = pos * 2;
  }

  t_ulong i = 0;
  #pragma omp parallel for
  for(i=0; i<new_pivot; i++){
    haar_coeff[i] = new_coeff[i];
  }

  // if(y == 1.2682295706116542){
  //   for(i=0; i<new_pivot; i++){
  //     my_file<<setprecision(16)<<new_coeff[i]<<endl;
  //   }
  //   first = false;
  // }

  
  i=0;
  #pragma omp parallel for
  for(i=0; i<new_pivot;i++){
    if(new_coeff[i] < (-1*uthresh))
      thresholded_coeff[i] = new_coeff[i] + uthresh;
    else if(new_coeff[i] > uthresh)
      thresholded_coeff[i] = new_coeff[i] - uthresh;
    else
      thresholded_coeff[i] = 0;
  }
  

  
  double sum = 0;
  for(i=1; i<new_pivot; i++){
    sum += (abs(thresholded_coeff[i]) * (pow(2,(floor(log2(i)) + 1)/2)));
  }
  
  thresholded_norm_squared = sum;
}


// main procedure for arrows
void Arrows::restarted_averages_incr(vector<double>& y,\
				     double c_i, t_ulong steps, double uconst, double rconst, vector<double>& estimates){

  
  t_ulong last_bin_head = 0;
  bool new_bin = true;
  double y_prev = 0;
  t_ulong pivot = 0;

  set_uthresh(steps,uconst);
  
  double m_factor = pow(steps,-1.0/3)*pow(sigma,2.0/3)/log(steps);
  double t_factor = m_factor*pow(c_i,1.0/3)*rconst;
  double online_mean = 0;

  for(t_ulong t=0; t<steps;t++){
    if(new_bin){
      estimates.push_back(y_prev);
      online_mean = y_prev;
      new_bin = false;
      
      fill(haar_coeff.begin(), haar_coeff.begin()+pivot, 0);
      fill(thresholded_coeff.begin(), thresholded_coeff.begin()+pivot, 0);

      haar_coeff[0] = y[t];
      thresholded_norm_squared = 0;
      //my_file<<"reset "<<thresholded_norm_squared<<endl;
    }
    else{
      online_mean = ((online_mean * (t-last_bin_head-1)) + y[t-1])/(t-last_bin_head);
      estimates.push_back(online_mean);
      t_ulong shifted_time = t-last_bin_head;

      double l = log2(shifted_time);
      
      if(ceil(l) == floor(l)){
	pivot = 2 * shifted_time;
	update_coeff_new_pivot(pivot,y[t]-online_mean);
      }
      else{
	update_coeff_old_pivot(shifted_time, y[t]-online_mean, pivot);
      }

      // adding in the new adaptive rule below
      c_i = thresholded_norm_squared/sqrt(pivot);
      
      
      //if(thresholded_norm_squared >= sqrt(pivot)*t_factor){
      if(c_i >= sigma/sqrt(pivot)){
	new_bin = true;
	last_bin_head = t+1;
	y_prev = y[t];
	//my_file<<"tnorm: "<<thresholded_norm_squared<<endl;
      }
    }
  }
  //my_file.close();
}


// initialize arrows solver, uconst and rconst are not used anymore. TODO: remove them
vector<double> shoot_arrows(vector<double> y, float sigma, double tv,\
			    double uconst, double rconst){

  vector<double> estimates;
  t_ulong steps = y.size();

  Arrows solver(steps,sigma);
  solver.restarted_averages_incr(y, tv, steps, uconst, rconst, estimates);
  return estimates;
}

PYBIND11_MODULE(arrows, m) {
    m.doc() = "pybind11 arrows plugin"; // optional module docstring

    m.def("shoot_arrows", [](vector<double> y, float sigma,\
			     double tv, double uconst, double rconst) -> py::array {
	    auto v = shoot_arrows(y,sigma,tv,uconst,rconst);
	return py::array(v.size(), v.data());
	  },py::arg("y"), py::arg("sigma"), py::arg("tv"),\
	  py::arg("uconst"), py::arg("rconst"));
}


// the command below creates a python module named arrows. The fucntion arrows.shoot_arrows invokes the solver

/*

c++ -O3 -Wall -shared -std=c++11 -fPIC -fopenmp `python3 -m pybind11 --includes` algos.cpp -o arrows`python3-config --extension-suffix`



 */
