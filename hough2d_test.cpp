#include<Eigen/Core>
#include<iostream>
#define cimg_use_jpeg
#include<CImg.h>
#include<cmath>
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


void fft_test()
{
	size_t N=1024;
	float z={1.0}; 
	CImg<float> orig_signal(N,N,1,1);
	
	std::array<size_t,2> mid{N/2,N/2};
	float th=(M_PI/180.0f)*45.0f;
	std::array<size_t,2> vec{std::cos(th)*(N/2),std::sin(th)*(N/2)};
	
	
	orig_signal.draw_line(mid[0]-vec[0],mid[1]-vec[1],mid[0]+vec[0],mid[1]+vec[1],&z);
	
	orig_signal.display();
	auto CList=orig_signal.get_FFT();
	auto mag=(CList[0].get_mul(CList[0])+CList[1].get_mul(CList[1])).get_sqrt();
	mag.display();
}



typedef std::array<float,3>  line2d_t;
typedef std::array<float,3>  point2d_t;

static inline point2d_t cross(const line2d_t& a,const line2d_t& b)
{
	return point2d_t{
		a[1]*b[2]-a[2]*b[1],
		a[2]*b[0]-a[0]*b[2],
		a[0]*b[1]-a[1]*b[0]
	};
}

template<class T>
void plot_line(CImg<T>& img,const std::array<float,3>& ln,const T* colors)
{
	float a=ln[0],b=ln[1],c=ln[2];
    std::cout << "a:" << a << " b:" << b << " c:" << c << std::endl;
	std::array<point2d_t,2> corners{};
	int csel=0;
	float w=img.width()-1,h=img.height()-1;
	std::array<line2d_t,4> walls{line2d_t{1.0f,0.0f,-w},line2d_t{1.0,0.0f,0.0f},line2d_t{0.0f,1.0f,-h},line2d_t{0.0,1.0f,0.0f}};
	
	for(unsigned int i=0;i<4;i++)
	{
		point2d_t isect=cross(ln,walls[i]);
		if(fabs(isect[2]) < 0.00000001)
		{
		//	continue;
		}
		isect[0]/=isect[2];isect[1]/=isect[2];
		std::cout << "Wall " << i << std::endl;
		std::cout << "Line: " << ln[0] << "," << ln[1] << "," << ln[2] << std::endl;
		std::cout << "Isect: " << isect[0] << "," << isect[1] << std::endl;
		
		if(isect[0] >= 0.0f && isect[0] <= w && isect[1] >= 0.0f && isect[1] <= h)
		{
			corners[csel++]=isect;
			std::cout << "Found corner" << csel << std::endl;
		}
		if(csel==2) break;
	}
	std::cout << corners[0][0] << "," << corners[0][1] << std::endl;
	std::cout << corners[1][0] << "," << corners[1][1] << std::endl;
	
	img.draw_line((int)corners[0][0],(int)corners[0][1],(int)corners[1][0],(int)corners[1][1],colors);
}

int main()
{
	//fft_test();
	//return 0;
	CImg<unsigned char> image("../diamond.jpg");                 //("../example_wide.jpg");
	const unsigned char red[3]={0xFF,0x00,0x00};

//RGB of blue: 134, 140,  138
	CImg<float> blue=image.get_RGBtoHSL().get_channel(0);     //  get hue channel to match for blue
	CImg<float> gscal = blue.get_threshold(40);
	gscal.display();
	
	//gscal.display();
	CImgList<float> z=gscal.get_gradient("xy",3);
	CImg<float> zout=z[0].get_mul(z[0])+z[1].get_mul(z[1]);
	//float mx=zout.max();
	//float mn=zout.min();
	zout.normalize(0.0f,1.0f);
	zout.threshold(0.013f);
	
	static constexpr unsigned int K=32;
 // Resolution of 1800 degrees for theta and rho 1024 pixels.
	naive_hough2d_lines lines({zout.width(),zout.height()},{1800,1024});
	auto t=std::chrono::high_resolution_clock::now();
	
	lines.load_frame([&zout](size_t x,size_t y) { return zout(x,y,0,0) > 0.5f; });
	lines.process_samples();
	std::array<naive_hough2d_lines::pixel_point,K> pointsout;
	lines.top_k(K,&pointsout[0]);
	unsigned int newk=lines.cluster_top_k(K,&pointsout[0],0.1,40.0f);
	std::cout <<  "newk:" <<  newk <<  std::endl;
	
	double elapsed=std::chrono::duration<double>(std::chrono::high_resolution_clock::now()-t).count();
	std::cout << "Elapsed time: " << elapsed << std::endl;
	
	CImg<uint32_t> htz(lines.hough_out.data(),lines.theta_n,lines.rho_n,1,1,true);
	
	CImg<uint32_t> zimg(lines.theta_n,lines.rho_n,1,3);
	for(unsigned c=0;c<3;c++) zimg.get_shared_channel(c).assign(htz);

	//unsigned int newk=K;
	std::cout << "NEWK: " << newk << std::endl;
	for(unsigned k=0;k<newk;k++) 
	{
		auto pp=pointsout[k];
		zimg(pp.theta_rho_index[0],pp.theta_rho_index[1],0,0)=255;
		zimg(pp.theta_rho_index[0],pp.theta_rho_index[1],0,1)=0;
		zimg(pp.theta_rho_index[0],pp.theta_rho_index[1],0,2)=0;
		plot_line(image,pointsout[k].line,red);
	}
	
	zimg.display();
	image.display();
	
	

	return 0;
}
