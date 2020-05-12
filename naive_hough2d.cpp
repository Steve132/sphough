#include "naive_hough2d.hpp"

#include<vector>
#include<iostream>
#include<cmath>
#include<numeric>
#include<algorithm>

naive_hough2d_lines::naive_hough2d_lines(const std::array<size_t,2>& tshape,const std::array<size_t,2>& ttheta_rho_shape):
	base_hough2d_lines(tshape,ttheta_rho_shape),
	m_size(ttheta_rho_shape[0]*ttheta_rho_shape[1]),
	hough_out(ttheta_rho_shape[0]*ttheta_rho_shape[1],0),
	cs_theta_cache(ttheta_rho_shape[0]),
	iota_cache(ttheta_rho_shape[0]*ttheta_rho_shape[1])
{
	float ft=0.0f;
	float fdt=M_PI/(theta_n);
	for(size_t t=0;t<theta_n;t++)
	{
		ft+=fdt;
		cs_theta_cache[t]=std::array<float,2>{cosf(ft),sinf(ft)};
	}
}
//Premature optimization is the root of all evil, but you could technically do vector gather scatter here to process pixels in sequence.  Assembly language ARMv8 LD1SW and AVX2 both have it.
//hough_out should really be [rho*theta_n+theta] because theta is cache coherent (at least in the current version of it.

//could also do threading and vectorization here
void naive_hough2d_lines::process_samples(size_t tbegin,size_t tend)
{
	if(tend==0) tend=theta_n;
	const size_t N=sparse_samples.size();
	for(size_t i=0;i<N;i++)
	{
		std::array<uint32_t,2> xy=sparse_samples[i];
		float fx=xy[0];
		float fy=xy[1];
		float lrs=rho_scale;
		for(size_t t=tbegin;t<tend;t++)
		{
			std::array<float,2> cs=cs_theta_cache[t];
			float rho=std::fabs(fx*cs[0]+fy*cs[1]);
			size_t rho_out=static_cast<size_t>(rho*rho_scale);
			//hough_out[rho_out*theta_n+t].fetch_add(1,std::memory_order_relaxed);
			hough_out[rho_out*theta_n+t]++;
		}
	}
}

static inline std::array<float,3> to_line(float a,float b,float ac)
{
	return {a,b,-ac};
}

void naive_hough2d_lines::top_k(unsigned int k,naive_hough2d_lines::pixel_point* pout) const
{
	// get max over neighborhood in image, assuming cluster width of n.  Merge first rather than get top-k universal values THEN get top-k maxes.
	std::vector<uint32_t> max_hough_out(hough_out.size(),0);
	std::size_t n = 10; // cluster diameter (must be even)
	// move row center of neighborhood
	// UH: pass in this size stuff/keep somehow but hough_out dims are 1800x1024.
	for(std::size_t i = n/2; i < 1024 - n/2; i+=n)
	{
		// move column center of neighborhood.  Note: moving as block, not sliding window.
		for(std::size_t j = n/2; j < 1800 - n/2; j+=n)
		{
			// get max over neighborhood block.
			uint32_t max_val = 0;
			std::size_t max_loc = 0;
			for(std::size_t ii = i; ii < i + n/2; ii++)
			{
				for(std::size_t jj = j; jj < j + n/2; jj++)
				{
					//max_hough_out[ii*1800+jj] = 255;
					if(hough_out[ii * 1800 + jj] > max_val)
					{
						max_val = hough_out[ii * 1800 + jj];
						max_loc = ii * 1800 + jj;
					}
				}
			}
			if(max_val > 0)
			{
				max_hough_out[max_loc] = max_val;
				std::cout <<"max for neighborhood " << i << ',' << j << " set to:" << max_val << std::endl;
			}
		}
	}
	
	std::iota(iota_cache.begin(),iota_cache.end(),0);
	auto cmpfunc=[this](size_t ai,size_t bi){
		return hough_out[ai] < hough_out[bi];
	};
	auto cmpfunc_max=[this,max_hough_out=max_hough_out](size_t ai,size_t bi){
		return max_hough_out[ai] < max_hough_out[bi];
	};
	auto kiterpoint=iota_cache.begin() + iota_cache.size()-k;
	std::nth_element(iota_cache.begin(),kiterpoint,iota_cache.end(),cmpfunc);//cmpfunc_max); // doesn't this get the linearized coordinate of the highest valued components over the whole hough_out?  Not the local highest cluster seeds?
	kiterpoint=iota_cache.begin() + iota_cache.size()-k;
	
	float itscale=1.0f/theta_scale;float irscale=1.0f/rho_scale;
	
	size_t i=0;
	for(auto ki=kiterpoint;ki<iota_cache.end();++ki)
	{
		size_t hi=*ki;
		size_t count=hough_out[hi];
		
		naive_hough2d_lines::pixel_point pp;
		pp.theta_rho_index=std::array<uint32_t,2>{hi % theta_n,hi/theta_n}; // unravel the linearized iota_cache index into hough_out
		pp.theta_rho=std::array<float,2>{itscale*static_cast<float>(pp.theta_rho_index[0]),irscale*static_cast<float>(pp.theta_rho_index[1])};
		
		std::array<float,2> cs=cs_theta_cache[pp.theta_rho_index[0]];
		pp.line=to_line(cs[0],cs[1],pp.theta_rho[1]);
		pp.count=count;
		pout[i++]=pp;
	}
}

static bool cluster_inside(const naive_hough2d_lines::pixel_point& cluster,const naive_hough2d_lines::pixel_point& point,float tb, float rb)
{
	return (fabs(cluster.theta_rho[0]-point.theta_rho[0]) < tb) && (fabs(cluster.theta_rho[1]-point.theta_rho[1]) < rb);
}


static std::ostream& operator<<(std::ostream& out,const naive_hough2d_lines::pixel_point& pp)
{
	return out << "{theta:" << pp.theta_rho[0] 
				<< " rho:" << pp.theta_rho[1]
				<< " count: " << pp.count 
				<< " line:(" << pp.line[0] << "," << pp.line[1] << "," << pp.line[2] << ")"
				<< "}";
}

static naive_hough2d_lines::pixel_point& operator+=(
	naive_hough2d_lines::pixel_point& clu,
	const naive_hough2d_lines::pixel_point& pp)
{
	float newcount=clu.count+pp.count;
	for(int tri=0;tri<2;tri++)
		clu.theta_rho[tri]=(clu.theta_rho[tri]*clu.count+pp.theta_rho[tri]*pp.count)/newcount;	
	clu.count+=pp.count;
	//restore the line after
}
unsigned int naive_hough2d_lines::cluster_top_k(
	unsigned int K,pixel_point* points,
	float theta_boundary,float rho_boundary) const
{
	
	std::vector<pixel_point> clusters;
	//std::iota(cluster_assignments.begin(),cluster_assignments.end(),0);	
	for(unsigned int i=0;i<K;i++)
	{
		unsigned int proposed_cluster=clusters.size();
		for(unsigned int ci=0;ci<clusters.size();ci++)
		{
			if(cluster_inside(clusters[ci],points[i],theta_boundary,rho_boundary))
			{
				proposed_cluster=ci;
				break;
			}
		}
		if(proposed_cluster==clusters.size())
		{
			std::cout << points[i] << " is a new cluster." << std::endl;
			clusters.push_back(points[i]);
		}
		else
		{
			pixel_point& clu=clusters[proposed_cluster];
			const pixel_point& pp=points[i];
			std::cout << "adding " << pp << " to " << clu << "\n\tResults in";
			clu+=pp;
			clu.count+=pp.count;
			std::cout << clu << std::endl;
		}
	}
	
	for(unsigned int ki=0;ki<clusters.size();ki++)
	{
		pixel_point& clu=clusters[ki];
		clu.theta_rho_index[0]=static_cast<unsigned int>(clu.theta_rho[0]*theta_scale);
		clu.theta_rho_index[1]=static_cast<unsigned int>(clu.theta_rho[1]*rho_scale);
		
		clu.line=to_line(cosf(clu.theta_rho[0]),sinf(clu.theta_rho[0]),clu.theta_rho[1]);
		points[ki]=clu;
		std::cout << "c " << ki << "=" << points[ki] << std::endl;
	}
	return clusters.size();
}

