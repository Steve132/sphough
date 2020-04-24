#ifndef SPARSE_HOUGH_2D_HPP
#define SPARSE_HOUGH_2D_HPP

#include<algorithm>
#include<cstdlib>
#include<functional>
#include<array>
#include<cmath>
#include<vector>

class sparse_hough2d_lines
{
protected:
	std::vector<bool> samples;
	size_t theta_n,rho_n;
	std::array<size_t,2> shape;
	std::vector<size_t> hough_out; 
	std::vector<std::array<size_t,2>> windows;
	size_t windowsize;
	
	void load_hough();
	std::vector<std::array<size_t,2>> get_windows(size_t windowsize,size_t overlap_pixels) const;
	void pairwise_hough(const std::vector<std::array<size_t,2>>& vin,std::vector<size_t>& hough_out);
	
	float rho_scale;
	float rho_max;
	float theta_scale;
public:
	sparse_hough2d_lines(const std::array<size_t,2>& tshape,const std::array<size_t,2>& ttheta_rho_shape={},size_t twindowsize=512,size_t toverlap_pixels=64);
	
	template<class Signal>
	void do_frame(const Signal& sig)
	{
		for(size_t y=0;y<shape[1];y++)
		for(size_t x=0;x<shape[0];x++)
		{
			samples[y*shape[1]+x]=sig(x,y);
		}
		load_hough();
	}
	void clear()
	{
		std::fill(hough_out.begin(),hough_out.end(),0);
	}
};

//cache is size*size = shape[0]*shape[1]*shape[0]*shape[1]
#endif
