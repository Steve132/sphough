#ifndef HOUGH_2D_HPP
#define HOUGH_2D_HPP

#include "base_hough2d.hpp"
#include<atomic>
#include<memory>


//the real header should be a pimpl
class naive_hough2d_lines: public base_hough2d_lines
{
protected:
	std::vector<std::array<float,2>> cs_theta_cache;
	size_t m_size;
	
	mutable std::vector<size_t> iota_cache;
public:
	using base_hough2d_lines::load_frame;
	std::vector<uint32_t> hough_out;
	naive_hough2d_lines(const std::array<size_t,2>& tshape,const std::array<size_t,2>& ttheta_rho_shape={});
	void process_samples(size_t tbegin=0,size_t tend=0);
	size_t size() const { return m_size; }
	
	struct pixel_point
	{
		std::array<uint32_t,2> theta_rho_index;
		std::array<float,2> theta_rho;
		std::array<float,3> line;
		uint32_t count;
		bool operator<(const pixel_point& other) const { return count < other.count; }
	};
	
	void top_k(unsigned k,pixel_point* pout) const;
	unsigned int cluster_top_k(unsigned k,pixel_point* pointsinout,float theta_boundary,float rho_boundary) const;
};


#endif
