#include "sparse_hough2d.hpp"

#include<vector>
#include<iostream>

typedef std::array<size_t,2> p2i_t; 

static 
std::vector<p2i_t> find_all_window_points(const std::vector<bool>& f,
								   const std::array<size_t,2>& fshape,
								   const std::array<size_t,2>& shape,
								   const std::array<size_t,2>& offset)
{
	std::vector<p2i_t> points;
	points.reserve(shape[0]*shape[1]/64);
	for(size_t y=0;y<shape[1];y++)
	for(size_t x=0;x<shape[0];x++)
	{
		p2i_t lp{x+offset[0],y+offset[1]};
		if(f[fshape[0]*lp[1]+lp[0]]) 
		{
			points.push_back(p2i_t{lp[0],lp[1]});
		}
	}
	return points;
}
static inline float atan_fast(float x)
{
	return x+0.273f*x*(1.0f-std::abs(x));
}
static inline float acos_fast(float x)
{
	float negate = static_cast<float>(x < 0);
	x = abs(x);
	float ret = -0.0187293;
	ret = ret * x;
	ret = ret + 0.0742610;
	ret = ret * x;
	ret = ret - 0.2121144;
	ret = ret * x;
	ret = ret + 1.5707288;
	ret = ret * sqrt(1.0-x);
	ret = ret - 2 * negate * ret;
	return negate * 3.14159265358979 + ret;	 //this can be vectorized.
}

static inline p2i_t write_to_theta_rho(
	const p2i_t& v1,
	const p2i_t& v2,
	float theta_scale,
	float rho_scale)
{
	float x=(float)v2[0];
	float y=(float)v2[1];
	float dx=x-(float)v1[0];
	float dy=y-(float)v1[1];
	float sf=1.0/sqrt(dx*dx+dy*dy);
	
	float a=-dy;
	float b=dx;
	if(b < 0.0f) {
		b=-b;a=-a;
	}
	float c=abs((a*x+b*y)*sf); //actually we can guarantee that b is positive so we have to negate c it to make sure rho is positive.
	float t=acos_fast(a*sf);
	t*=theta_scale;
	c*=rho_scale;
	
	p2i_t out{static_cast<size_t>(t),static_cast<size_t>(c)};
	return out;
}

void sparse_hough2d_lines::pairwise_hough(const std::vector<std::array<size_t,2>>& vin,std::vector<size_t>& ho)
{
	const size_t N=vin.size();
	for(size_t i=0;i<N;i++)
	{
		for(size_t j=0;j<i;j++)
		{
			p2i_t v1=vin[i];
			p2i_t v2=vin[j];
			p2i_t out=write_to_theta_rho(v1,v2,theta_scale,rho_scale);
			ho[out[1]*theta_n+out[0]]++;
		}
	}
}


std::vector<std::array<size_t,2>> sparse_hough2d_lines::get_windows(size_t windowsize,size_t overlap_pixels) const
{
	std::vector<std::array<size_t,2>> windows_out;
	size_t increment_amount=windowsize-overlap_pixels;
	for(size_t y=0;(y+windowsize) <= shape[1];y+=increment_amount)
	for(size_t x=0;(x+windowsize) <= shape[0];x+=increment_amount)
	{
		windows_out.push_back(p2i_t{x,y});
	}
	return windows_out;
}

void sparse_hough2d_lines::load_hough()
{
	const size_t N=windows.size();
	for(size_t i=0;i<N;i++)
	{
		std::vector<p2i_t> wp=find_all_window_points(samples,shape,{windowsize,windowsize},windows[i]);
		//std::cerr << "Window " << windows[i][0] << "," << windows[i][1] << std::endl;
		pairwise_hough(wp,hough_out);
	}
}

sparse_hough2d_lines::sparse_hough2d_lines(
	const std::array<size_t,2>& tshape,
	const std::array<size_t,2>& ttheta_rho_shape,size_t twindowsize,size_t toverlap_pixels):
	samples(tshape[0]*tshape[1]),
	theta_n(ttheta_rho_shape[0]),
	rho_n(ttheta_rho_shape[1]),
	shape(tshape),
	windowsize(twindowsize)
{
	rho_max=tshape[0]+tshape[1];
	
	if(rho_n==0) 
	{
		rho_n=static_cast<size_t>(rho_max);
	}
	if(theta_n==0)
	{
		theta_n=1800;
	}
	
	rho_scale=(rho_n-1)/rho_max;
	theta_scale=(theta_n-1)/M_PI;
	hough_out.resize(theta_n*rho_n,0);
	windows=get_windows(windowsize,toverlap_pixels);
}
	
