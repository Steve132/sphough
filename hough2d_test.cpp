#include<Eigen/Core>
#include<iostream>
#define cimg_use_jpeg
#include<CImg.h>
#include<chrono>
using namespace std;
using namespace cimg_library;

#include "sparse_hough2d.hpp"
#include "naive_hough2d.hpp"

static inline double timit()
{
	auto t=std::chrono::high_resolution_clock::now();
}


void perftest(const CImg<float>& zout)
{
	for(size_t i=16;i<1024;i*=2)
	{
		sparse_hough2d_lines lines({zout.width(),zout.height()},
								   {1800,1024},i,i/8);
		
		std::cout << "Now testing i==" << i << std::endl;
		auto t=std::chrono::high_resolution_clock::now();
		lines.do_frame([&zout](size_t x,size_t y) { return zout(x,y,0,0) > 0.5f; });
		double elapsed=std::chrono::duration<double>(std::chrono::high_resolution_clock::now()-t).count();
		std::cout << "Elapsed time: " << elapsed << std::endl;
	}//zout.display();	
}
int main()
{  
	CImg<unsigned char> image("../cards.jpg");
	
	CImg<float> gscal=image.get_RGBtoHSL().get_channel(2);
	
	//gscal.display();
	CImgList<float> z=gscal.get_gradient("xy",3);
	CImg<float> zout=z[0].get_mul(z[0])+z[1].get_mul(z[1]);
	//float mx=zout.max();
	//float mn=zout.min();
	zout.normalize(0.0f,1.0f);
	zout.threshold(0.013f);
	
	static constexpr unsigned int K=32;
	naive_hough2d_lines lines({zout.width(),zout.height()},{1800,1024});
	auto t=std::chrono::high_resolution_clock::now();
	lines.load_frame([&zout](size_t x,size_t y) { return zout(x,y,0,0) > 0.5f; });
	lines.process_samples();
	double elapsed=std::chrono::duration<double>(std::chrono::high_resolution_clock::now()-t).count();
	
	std::array<naive_hough2d_lines::pixel_point,K> pointsout;
	lines.top_k(K,&pointsout[0]);
	
	std::cout << "Elapsed time: " << elapsed << std::endl;
	
	
	CImg<uint32_t> htz(lines.hough_out.data(),lines.theta_n,lines.rho_n,1,1,true);
	
	CImg<uint32_t> zimg(lines.theta_n,lines.rho_n,1,3);
	for(unsigned c=0;c<3;c++) zimg.get_shared_channel(c).assign(htz);
	
	for(unsigned k=0;k<K;k++) 
	{
		auto pp=pointsout[k];
		zimg(pp.theta_rho_index[0],pp.theta_rho_index[1],0,1)=0;
		zimg(pp.theta_rho_index[0],pp.theta_rho_index[1],0,2)=0;
	}
	
	zimg.display();

	return 0;
}
