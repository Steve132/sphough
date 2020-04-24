#include<Eigen/Core>
#include<iostream>
#define cimg_use_jpeg
#include<CImg.h>
#include<chrono>
using namespace std;
using namespace cimg_library;

#include "sparse_hough2d.hpp"

static inline double timit()
{
	auto t=std::chrono::high_resolution_clock::now();
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
	
	
	for(size_t i=16;i<1024;i*=2)
	{
		sparse_hough2d_lines lines({zout.width(),zout.height()},
							   {1024,1024},i,i/8);
	
		std::cout << "Now testing i==" << i << std::endl;
		auto t=std::chrono::high_resolution_clock::now();
		lines.do_frame([&zout](size_t x,size_t y) { return zout(x,y,0,0) > 0.5f; });
		double elapsed=std::chrono::duration<double>(std::chrono::high_resolution_clock::now()-t).count();
		std::cout << "Elapsed time: " << elapsed << std::endl;
	}//zout.display();

	return 0;
}
