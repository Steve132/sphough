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
		for(size_t t=tbegin;t<tend;t++)
		{
			std::array<float,2> cs=cs_theta_cache[t];
			float rho=-(fx*cs[0]+fy*cs[1]);  //c=-(ax+by)
			size_t rho_out=static_cast<size_t>(rho*rho_scale+rho_offset);
			//hough_out[rho_out*theta_n+t].fetch_add(1,std::memory_order_relaxed);
			hough_out[rho_out*theta_n+t]++;
		}
	}
}

static inline std::array<float,3> to_line(float a,float b,float ac)
{
	return {a,b,ac};
}

void naive_hough2d_lines::top_k(unsigned int k,naive_hough2d_lines::pixel_point* pout) const
{
	std::iota(iota_cache.begin(),iota_cache.end(),0);
	auto cmpfunc=[this](size_t ai,size_t bi){
		return hough_out[ai] < hough_out[bi];
	};
	auto kiterpoint=iota_cache.begin() + iota_cache.size()-k;
	std::nth_element(iota_cache.begin(),kiterpoint,iota_cache.end(),cmpfunc);
	kiterpoint=iota_cache.begin() + iota_cache.size()-k;
	
	float itscale=1.0f/theta_scale;float irscale=1.0f/rho_scale;
		
	size_t i=0;
	for(auto ki=kiterpoint;ki<iota_cache.end();++ki)
	{
		size_t hi=*ki;
		size_t count=hough_out[hi];
		
		naive_hough2d_lines::pixel_point pp;
		pp.theta_rho_index=std::array<uint32_t,2>{static_cast<uint32_t>(hi % theta_n),
												  static_cast<uint32_t>(hi/theta_n)};
		pp.theta_rho=std::array<float,2>{
			itscale*static_cast<float>(pp.theta_rho_index[0]),
			irscale*(static_cast<float>(pp.theta_rho_index[1])-rho_offset)
		};
		
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


std::ostream& operator<<(std::ostream& out,const naive_hough2d_lines::pixel_point& pp)
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
	return clu;
}
unsigned int naive_hough2d_lines::cluster_top_k(
	unsigned int K,pixel_point* points,
	float theta_boundary,float rho_boundary) const
{
	rho_boundary*=rho_max;
	std::cerr << "RB: " << rho_boundary << std::endl;
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
			std::cerr << points[i] << " is a new cluster." << std::endl;
			clusters.push_back(points[i]);
		}
		else
		{
			pixel_point& clu=clusters[proposed_cluster];
			const pixel_point& pp=points[i];
			std::cerr << "adding " << pp << " to " << clu << "\n\tResults in";
			clu+=pp;
			clu.count+=pp.count;
			std::cerr << clu << std::endl;
		}
	}
	
	for(unsigned int ki=0;ki<clusters.size();ki++)
	{
		pixel_point& clu=clusters[ki];
		clu.theta_rho_index[0]=static_cast<unsigned int>(clu.theta_rho[0]*theta_scale);
		clu.theta_rho_index[1]=static_cast<unsigned int>(clu.theta_rho[1]*rho_scale+rho_offset);
		
		clu.line=to_line(cosf(clu.theta_rho[0]),sinf(clu.theta_rho[0]),clu.theta_rho[1]);
		points[ki]=clu;
		std::cerr<< "c " << ki << "=" << points[ki] << std::endl;
	}
	return clusters.size();
}

