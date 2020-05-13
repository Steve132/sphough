#include "base_hough2d.hpp"

#include<vector>
#include<iostream>
#include<cmath>

base_hough2d_lines::base_hough2d_lines(
	const std::array<size_t,2>& tshape,
	const std::array<size_t,2>& ttheta_rho_shape):
	//samples(tshape[0]*tshape[1]),
	theta_n(ttheta_rho_shape[0]),
	rho_n(ttheta_rho_shape[1]),
	shape(tshape)
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
	
	rho_scale=(rho_n-1)/(2.0f*rho_max);
	rho_offset=(rho_n-1)/2;
	theta_scale=(theta_n-1)/M_PI;
}
	
