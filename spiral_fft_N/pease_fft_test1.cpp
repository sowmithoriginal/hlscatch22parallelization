// SPIRAL License
// 
// Copyright 2017, Carnegie Mellon University
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// 
// The views and conclusions contained in the software and documentation are those
// of the authors and should not be interpreted as representing official policies,
// either expressed or implied, of the SPIRAL project.

#include "header_N.h"
#include <stdio.h>
#ifdef USE_FFTW
#include <fftw3.h>
#endif
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <iostream>
#include <complex.h>

#include <cmath>
using namespace std;

float cal_snr21(float* x, float* y, unsigned int size) {
  float *x_sq = (float *)malloc(sizeof(float)*size);
  float *x_y_sq = (float *)malloc(sizeof(float)*size);
  float sum_x_sq, sum_x_y_sq;
  sum_x_sq = 0;
  sum_x_y_sq = 0;
  int i;
  for (i=0; i<size; i++) {
    x_sq[i] = x[i]*x[i];
    x_y_sq[i] = (x[i]-y[i])*(x[i]-y[i]);
    sum_x_sq += x_sq[i];
    sum_x_y_sq += x_y_sq[i];
  }

  free(x_sq);
  free(x_y_sq);

  float mean_x_sq, mean_x_y_sq;
  mean_x_sq = sum_x_sq / size;
  mean_x_y_sq = sum_x_y_sq / size;
  float rms_x, rms_x_y;
  rms_x = sqrt(mean_x_sq);
  rms_x_y = sqrt(mean_x_y_sq);

  return 20*log10(rms_x / rms_x_y);
}

float cal_snr_complex_float_fixed1(float_cmplx_type x[N1], cmplx_type y[N1]) {

	float* xri = (float *)malloc(sizeof(float_cmplx_type)*2*N1);
	float* yri = (float *)malloc(sizeof(float_cmplx_type)*2*N1);
	int i;
	for (i=0; i<N1; i++) {
		xri[2*i] = x[i].real;
		xri[2*i+1] = x[i].imag;
		yri[2*i] = y[i].real *N1;
		yri[2*i+1] = y[i].imag *N1;
	}
	float snr =  cal_snr21(xri, yri, 2*N1);
	free(xri);
	free(yri);
	return snr;
}

void disp_complex_type1(char* name,  float_cmplx_type x[N1]) {
	std::cout<<name<<" (real):\n";
	int i=0;
	for (i=0; i<N1; i++) {
		std::cout<<x[i].real<<", ";
	}
	std::cout<<"}\n";
	std::cout<<"(img):\n{";

	for (i=0; i<N1; i++) {
		std::cout<<x[i].imag<<", ";
	}
	std::cout<<"}\n";
}

void disp_complex_type1(char* name,  cmplx_type x[N1]) {
	std::cout<<name<<" (real):\n1";
	int i=0;
	for (i=0; i<N1; i++) {
#ifdef FIXED_POINT
		std::cout<<std::setprecision(14)<<x[i].real<<", ";
#else
		std::cout<<x[i].real<<", ";
#endif
	}
	std::cout<<"}\n";
	std::cout<<"(img):\n{";

	for (i=0; i<N1; i++) {
#ifdef FIXED_POINT
		std::cout<<std::setprecision(14)<<x[i].imag<<", ";
#else
		std::cout<<x[i].imag<<", ";
#endif
	}
	std::cout<<"}\n";
}


void convert_fixed_point_to_float1(cmplx_type from[N1], float_cmplx_type to[N1]) {
	int i;
	for (i=0; i<N1; i++) {
		to[i].real = (float)from[i].real;
		to[i].imag = (float)from[i].imag;
	}
}

void convert_float_to_fixed_point1(float_cmplx_type from[N1], cmplx_type to[N1]) {
	int i;
	for (i=0; i<N1; i++) {
		to[i].real = from[i].real;
		to[i].imag = from[i].imag;
	}
}

void init_xilinx_input21(cmplx_type result[N1]) {
#define MAX_SAMPLES N1
#define IP_WIDTH FIXED_POINT_WIDTH
#define DATA_WIDTH 14
#define MATH_PI 3.14159265358979323846

	int i;
	for (i=0; i<MAX_SAMPLES; i++) {
	  double theta, re_real, im_real, theta2;
	  int re_int, im_int;
	  theta = (double)i / (double)MAX_SAMPLES * 2.6 * 2.0 * MATH_PI;
	  re_real = cos(-theta);
	  im_real = sin(-theta);
	  theta2 =  (double)i / (double)MAX_SAMPLES * 23.2 * 2.0 * MATH_PI;
	  re_real = re_real + (cos(-theta2)/4.0);
	  im_real = im_real + (sin(-theta2)/4.0);
	  re_int = round(re_real * (double) pow(2,DATA_WIDTH));
	  im_int = round(im_real * (double) pow(2,DATA_WIDTH));

	  while (re_real > 1) {
		  re_real = re_real - 1;
	  }
	  while (re_real < -1) {
		  re_real = re_real + 1;
	  }
	  while (im_real > 1) {
		  im_real = im_real - 1;
	  }
	  while (im_real < -1) {
		  im_real = im_real + 1;
	  }
	  result[i].real = re_real;
	  result[i].imag = im_real;
	}
  std::cout<<"\n";
}

#ifdef USE_FFTW

#ifdef DOUBLE
void fftw_exe1(cmplx_type ref_in[N1], cmplx_type fftw_out[N1]) {
	fftw_complex* in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)* N1);
	fftw_complex* out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)* N1);
	int i;
	for (i=0; i<N1; i++) {
		(in[i])[0] = ref_in[i].real;
		(in[i])[1] = ref_in[i].imag;
	}

	fftw_plan p;
	int n[LOGrN];
	for (i=0; i<LOGrN; i++) {
		n[i] = N1;
	}
	p = fftw_plan_dft(1, n, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
	fftw_execute(p);
	fftw_destroy_plan(p);

	for (i=0; i<N1; i++) {
		fftw_out[i].real = out[i][0];
		fftw_out[i].imag = out[i][1];
	}
//	disp_fftw_complex("fftw output", out);

	fftw_free(out);
	fftw_free(in);
}
#else
void fftw_exe1(float_cmplx_type ref_in[N1], float_cmplx_type fftw_out[N1]) {
	fftwf_complex* in = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex)* N1);
	fftwf_complex* out = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex)* N1);
	int i;
	for (i=0; i<N1; i++) {
		(in[i])[0] = ref_in[i].real;
		(in[i])[1] = ref_in[i].imag;
	}

	fftwf_plan p;
	int n[LOGrN];
	for (i=0; i<LOGrN; i++) {
		n[i] = N1;
	}
	p = fftwf_plan_dft(1, n, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
	fftwf_execute(p);
	fftwf_destroy_plan(p);

	for (i=0; i<N1; i++) {
		fftw_out[i].real = out[i][0];
		fftw_out[i].imag = out[i][1];
	}
//	disp_fftw_complex("fftw output", out);

	fftwf_free(out);
	fftwf_free(in);
}

#endif

#endif


void derive_parameters1() {
	// LOG2N1
	printf("#define LOG2N1 %.0f\n", log2(N1));
	printf("#define LOG2R1 %.0f\n", log2(R1));
	printf("#define LOGrN %.0f\n", log2(N1)/log2(R1));
	printf("#define LOG2SW1 %.0f\n", log2(SW1));
	printf("#define N_O_SW %d\n", N1/(SW1));
	printf("#define MIN_SW_N_O_R %d\n", (SW1<=N1/R1)?SW1:N1/R1);
	printf("#define LOOP_II %d\n", (SW1<=N1/R1)?R1:N1/SW1);
}

int validate_original_params1() {
	// R1, SW1, N1
	long int i;
	for (i=R1; i<N1; ) {
		i = i * R1;
	}
	if (i!= N1) {
		printf("wrong R1, N1\n");
		return 1;
	}

	for (i=SW1; i<N1; ) {
		i = i + SW1;
	}
	if (i!= N1) {
		printf("wrong SW1, N1\n");
		return 1;
	}

	derive_parameters1();
	return 0;
}

int validate_parameters1() {
	int i, tmp;



	// R1, LOGrN, N1
	tmp = 1;
	for (i=0; i<LOGrN; i++) {
		tmp = tmp * R1;
	}
	if (tmp != N1) {
		printf("wrong R1, LOGrN, N1\n");
		return 1;
	}
//	// R1, LOGrSW, SW1
//	tmp = 0;
//	for (i=0; i<LOGrSW; i++) {
//		tmp = tmp + R1;
//	}
//	if (tmp != SW1) {
//		printf("wrong R1, LOGrSW, SW1\n");
//		return 1;
//	}

	// LOG2SW1
	tmp = 1;
	for (i=0; i<LOG2SW1; i++) {
		tmp = tmp * 2;
	}
	if (tmp != SW1) {
		printf("wrong LOG2SW1\n");
		return 1;
	}

	// LOG2R1
	tmp = 1;
	for (i=0; i<LOG2R1; i++) {
		tmp = tmp * 2;
	}
	if (tmp != R1) {
		printf("wrong LOG2R1\n");
		return 1;
	}

//	// N_O_SW
//	if (N1/SW1 != N_O_SW) {
//		printf("wrong N_O_SW\n");
//		return 1;
//	}

	return 0;
}

int main() {
	if (1 == validate_original_params1()) {
		return 1;
	}
	if (1 == validate_parameters1()) {
		return 1;
	}

	cmplx_type xilinx_input1[N1];
	float_cmplx_type x_ref1[N1];
	float_cmplx_type y_ref1[N1];
	cmplx_type y_test1[N1];
	float_cmplx_type y_test_float1[N1];

	init_xilinx_input21(xilinx_input1);
	convert_fixed_point_to_float1(xilinx_input1, x_ref1);

    if (N1<1024) {
    	disp_complex_type1("input x to fftw ",x_ref1);
    	disp_complex_type1("input x to IP ", xilinx_input1);
    }

	fftw_exe1(x_ref1, y_ref1);

    if (N1<1024) {
    	disp_complex_type1("reference y ",y_ref1);
    }

	pease_fft_N(xilinx_input1, y_test1);

	// Scale the fixed point results
	// because the IP right shift the bits internally to mitigate overflow.
    int i;
	for (i=0; i<N1; i++) {
		y_test_float1[i].real = y_test1[i].real *N1;
		y_test_float1[i].imag = y_test1[i].imag *N1;
	}

    if (N1<1024) {
    	disp_complex_type1("actual y ",y_test_float1);
    }

 	float snr;
 	snr = cal_snr_complex_float_fixed1(y_ref1, y_test1);
 	std::cout<<"SNR = "<<snr<<" db \n";

 	if ( snr > 100)
 		return 0;
 	else
 		return 1;
}
