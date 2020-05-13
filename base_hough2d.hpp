#ifndef BASE_HOUGH_2D_HPP
#define BASE_HOUGH_2D_HPP

#include<cstdlib>
#include<array>
#include<vector>

class base_hough2d_lines
{
protected:
	//std::vector<uint8_t> samples;
	std::vector<std::array<uint32_t,2>> sparse_samples;
	
	float rho_scale;
	float rho_offset;
	float rho_max;
	float theta_scale;
public:
	std::array<size_t,2> shape;
	size_t theta_n,rho_n;
	
	base_hough2d_lines(const std::array<size_t,2>& tshape,const std::array<size_t,2>& ttheta_rho_shape={});
	
	template<class Signal>
	void load_frame(const Signal& sig)
	{
		sparse_samples.clear();
		sparse_samples.reserve(shape[0]*shape[1]/64);
		for(size_t y=0;y<shape[1];y++)
		for(size_t x=0;x<shape[0];x++)
		{
			if(sig(x,y))
			{
				std::array<uint32_t,2> nxt{static_cast<uint32_t>(x),static_cast<uint32_t>(y)};
				sparse_samples.push_back(nxt);
			}
		}
	}
};

#endif
