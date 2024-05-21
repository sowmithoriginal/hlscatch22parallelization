#include "constants.h"
#include <float.h>
#include <iostream>
#include <math.h>
#include "header.h"
#include "header_N.h"
#include <float.h>
#include <string.h>
#include <time.h>
#include <cmath>
#include <stdlib.h>
#include <cstdlib>
#include <complex>
#include <algorithm>
#include <complex.h>
#include <hls_math.h> 


#define nCoeffs 3
#define nPoints 4
#define pieces 2
#define nBreaks 3
#define deg 3
#define nSpline 4
#define piecesExt 8 //3 * deg - 1

int nextpow2(int n);
void co_autocorrs(double y[DATA_SIZE], double z[DATA_SIZE]);
cmplx_type cmpxdiv(cmplx_type a, cmplx_type b);
void dot_multiply(cmplx_type a[DATA_SIZE], cmplx_type b[DATA_SIZE]);
double mean1(double a[]);
int co_firstzero( double y[], int size, int maxtau);
double FC_LocalSimple_mean_tauresrat(double y[], int size, int train_length);
double round(double var, int precision);
double mean(const double a[], const int size);
int num_bins_auto(const double y[], const int size);
double stddev(const double a[], const int size);
int histcounts_preallocated(const double y[], const int size, int nBins, int binCounts[], double binEdges[]);
double max_(const double a[], const int size);
double min_(const double a[], const int size);
void histbinassign(const double y[], const int size, const double binEdges[], const int nEdges, int output[]);
void histcount_edges(const double y[], const int size, const double binEdges[], const int nEdges, int output[]);
void sb_coarsegrain(const double y[], const int size, const char how[], const int num_groups, int labels[]);
double f_entropy(const double a[], const int size);
void subset(const int a[], int b[], const int start, const int end);
void linspace(double start, double end, int num_groups, double out[]);
double quantile(const double y[], const int size, const double quant);
void sort(double y[], int size);
int simple_strcmp(const char *str1, const char *str2);
double cov(const double x[], const double y[], const int size);
int histcounts(const double y[DATA_SIZE], const int size, int nBins, int binCounts[], double binEdges[]);
int linreg(const int n, const double x[], const double y[], double* m, double* b);
double norm_(const double a[], const int size);
double SC_FluctAnal_2_50_1_logi_prop_r1(const double y[DATA_SIZE], const int size, const int lag, const char how[]);
double autocov_lag(const double x[DATA_SIZE], const int size, const int lag);
double cov_mean(double x[DATA_SIZE], double y[DATA_SIZE], const int size);
int splinefit(const double y[DATA_SIZE], const int size, double yOut[]);
void gauss_elimination(int size, double A[DATA_SIZE][DATA_SIZE], double b[DATA_SIZE], double x[DATA_SIZE]);
void matrix_times_vector(const int sizeA1, const int sizeA2, const double A[DATA_SIZE][DATA_SIZE], const int sizeb, const double b[DATA_SIZE], double c[DATA_SIZE]);
void matrix_multiply(const int sizeA1, const int sizeA2, const double A[DATA_SIZE][DATA_SIZE], const int sizeB1, const int sizeB2, const double B[DATA_SIZE][DATA_SIZE], double C[DATA_SIZE][DATA_SIZE]);
void lsqsolve_sub(const int sizeA1, const int sizeA2, const double A[DATA_SIZE*DATA_SIZE], const int sizeb, const double b[DATA_SIZE], double x[DATA_SIZE]);
void icumsum(const int a[DATA_SIZE], const int size, int b[DATA_SIZE]);
int iLimit(int x, int lim);
double FC_LocalSimple_mean_stderr(const double y[DATA_SIZE], const int size, const int train_length);
void cumsum(const double a[], const int size, double b[]);
double SP_Summaries_welch_rect(const double y[], const int size, const char what[]);
int welch(const double y[], const int size, const int NFFT, const double Fs, const double window[], const int windowWidth, double *Pxx, double *f);
double complex_abs(const cmplx_type c);
bool is_infinite(float x);
double median(const double a[], const int size);
void DN_OutlierInclude_np_001_mdrmd(double y[], const int size, const int sign, hls::stream<data_t> &output);
double corr(const double x[], const double y[], const int size);
double autocorr_lag(double x[], const int size, const int lag);
void diff(double a[], const int size, double b[]);
void generate(hls::stream<data_t> &input, double y[DATA_SIZE]);
void replicate_stream(hls::stream<data_t> &input, hls::stream<data_t> &input1, hls::stream<data_t> &input2, hls::stream<data_t> &input3);

void MD_hrv_classic_pnn40(hls::stream<data_t> &input, const int size, hls::stream<data_t> &output){
    
    static double y[DATA_SIZE];
    generate(input,y);
    // NaN check
    // for(int i = 0; i < size; i++)
    // {
    //     if(isnan(y[i]))
    //     {
    //         return NAN;
    //     }
    // }
    
    const int pNNx = 40;
    
    // compute diff
    double Dy[DATA_SIZE];
    diff(y, size, Dy);
    
    double pnn40 = 0;
    for(int i = 0; i < size-1; i++){
        if(fabs(Dy[i])*1000 > pNNx){
            pnn40 += 1;
        }
    }
    
    double l2;
    l2 = pnn40/(size-1);
    output << l2;
}

void diff(double a[], const int size, double b[])
{
    for (int i = 1; i < size; i++) {
        b[i - 1] = a[i] - a[i - 1];
    }
}

double CO_trev_1_num(const double y[], const int size)
{
    
    // NaN check
    // for(int i = 0; i < size; i++)
    // {
    //     if(isnan(y[i]))
    //     {
    //         return NAN;
    //     }
    // }
    
    int tau = 1;
    
    double diffTemp[DATA_SIZE];
    
    for(int i = 0; i < size-tau; i++)
    {
        diffTemp[i] = pow(y[i+1] - y[i],3);
    }
    
    double out;
    
    out = mean(diffTemp, size-tau);
    
    // free(diffTemp);
    
    return out;
}

void IN_AutoMutualInfoStats_40_gaussian_fmmi(hls::stream<data_t> &input, const int size, hls::stream<data_t> &output)
{
    static double y[DATA_SIZE];
    generate(input,y);
    // NaN check
    // for(int i = 0; i < size; i++)
    // {
    //     if(isnan(y[i]))
    //     {
    //         return NAN;
    //     }
    // }
    
    // maximum time delay
    int tau = 40;
    
    // don't go above half the signal length
    if(tau > ceil((double)size/2)){
        tau = ceil((double)size/2);
    }
    
    // compute autocorrelations and compute automutual information
    double ami[DATA_SIZE];
    for(int i = 0; i < tau; i++){
        double ac = autocorr_lag(y,size, i+1);
        ami[i] = -0.5 * log(1 - ac*ac);
        // printf("ami[%i]=%1.7f\n", i, ami[i]);
    }
    
    // find first minimum of automutual information
    double fmmi = tau;
    for(int i = 1; i < tau-1; i++){
        if(ami[i] < ami[i-1] & ami[i] < ami[i+1]){
            fmmi = i;
            // printf("found minimum at %i\n", i);
            break;
        }
    }
    
    // free(ami);
    
    output << fmmi;
}

double corr(const double x[], const double y[], const int size){
    
    double nom = 0;
    double denomX = 0;
    double denomY = 0;
    
    double meanX = mean(x, size);
    double meanY = mean(y, size);
    
    for(int i = 0; i < size; i++){
        nom += (x[i] - meanX) * (y[i] - meanY);
        denomX += (x[i] - meanX) * (x[i] - meanX);
        denomY += (y[i] - meanY) * (y[i] - meanY);
        
        //printf("x[%i]=%1.3f, y[%i]=%1.3f, nom[%i]=%1.3f, denomX[%i]=%1.3f, denomY[%i]=%1.3f\n", i, x[i], i, y[i], i, nom, i, denomX, i, denomY);
    }
    
    return nom/sqrt(denomX * denomY);
    
}

double autocorr_lag(double x[], const int size, const int lag){
    
    return corr(x, &(x[lag]), size-lag);
    
}

double SB_BinaryStats_diff_longstretch0(const double y[], const int size){
    
    // NaN check
    // for(int i = 0; i < size; i++)
    // {
    //     if(isnan(y[i]))
    //     {
    //         return NAN;
    //     }
    // }
    
    // binarize
    int yBin[DATA_SIZE];
    for(int i = 0; i < size-1; i++){
        
        double diffTemp = y[i+1] - y[i];
        yBin[i] = diffTemp < 0 ? 0 : 1;
        
        /*
        if( i < 300)
            printf("%i, y[i+1]=%1.3f, y[i]=%1.3f, yBin[i]=%i\n", i, y[i+1], y[i], yBin[i]);
         */
        
    }
    
    int maxstretch0 = 0;
    int last1 = 0;
    for(int i = 0; i < size-1; i++){
        if(yBin[i] == 1 | i == size-2){
            double stretch0 = i - last1;
            if(stretch0 > maxstretch0){
                maxstretch0 = stretch0;
            }
            last1 = i;
        }
    }
    
    // free(yBin);
    
    return maxstretch0;
}

double SB_BinaryStats_mean_longstretch1(const double y[], const int size){
    
    // NaN check
    // for(int i = 0; i < size; i++)
    // {
    //     if(isnan(y[i]))
    //     {
    //         return NAN;
    //     }
    // }
    
    // binarize
    int yBin[DATA_SIZE];
    double yMean = mean(y, size);
    for(int i = 0; i < size-1; i++){
        
        yBin[i] = (y[i] - yMean <= 0) ? 0 : 1;
        //printf("yBin[%i]=%i\n", i, yBin[i]);
        
    }
    
    int maxstretch1 = 0;
    int last1 = 0;
    for(int i = 0; i < size-1; i++){
        if(yBin[i] == 0 | i == size-2){
            double stretch1 = i - last1;
            if(stretch1 > maxstretch1){
                maxstretch1 = stretch1;
            }
            last1 = i;
        }
        
    }
    
    // free(yBin);
    
    return maxstretch1;
}

void DN_OutlierInclude_p_001_mdrmd(hls::stream<data_t> &input, const int size, hls::stream<data_t> &output)
{   
    static double y[DATA_SIZE];
    generate(input,y);
    DN_OutlierInclude_np_001_mdrmd(y, size, 1.0, output);

}

void DN_OutlierInclude_n_001_mdrmd(hls::stream<data_t> &input, const int size, hls::stream<data_t> &output)
{
    static double y[DATA_SIZE];
    generate(input,y);
    DN_OutlierInclude_np_001_mdrmd(y, size, -1.0, output);
}

void DN_OutlierInclude_np_001_mdrmd(double y[], const int size, const int sign, hls::stream<data_t> &output)
{
    
    // // NaN check
    // for(int i = 0; i < size; i++)
    // {
    //     if(isnan(y[i]))
    //     {
    //         return NAN;
    //     }
    // }
    
    double inc = 0.01;
    int tot = 0;
    double yWork[DATA_SIZE];
    
    // apply sign and check constant time series
    int constantFlag = 1;
    for(int i = 0; i < size; i++)
    {
        if(y[i] != y[0])
        {
            constantFlag = 0;
        }
        
        // apply sign, save in new variable
        yWork[i] = sign*y[i];
        
        // count pos/ negs
        if(yWork[i] >= 0){
            tot += 1;
        }
        
    }
    if(constantFlag){
        // free(yWork);
        output << 0;; // if constant, return 0
    }
    
    // find maximum (or minimum, depending on sign)
    double maxVal = max_(yWork, size);
    
    // maximum value too small? return 0
    if(maxVal < inc){
        // free(yWork);
        output << 0;
    }
    
    int nThresh = maxVal/inc + 1;
    
    // save the indices where y > threshold
    double r[DATA_SIZE];
    
    // save the median over indices with absolute value > threshold
    double msDti1[2*DATA_SIZE];
    double msDti3[2*DATA_SIZE];
    double msDti4[2*DATA_SIZE];
    
    for(int j = 0; j < nThresh; j++)
    {
        //printf("j=%i, thr=%1.3f\n", j, j*inc);
        
        int highSize = 0;
        
        for(int i = 0; i < size; i++)
        {
            if(yWork[i] >= j*inc)
            {
                r[highSize] = i+1;
                //printf("r[%i]=%1.f \n", highSize, r[highSize]);
                highSize += 1;
            }
        }
        
        // intervals between high-values
        double Dt_exc[2*DATA_SIZE];
        
        for(int i = 0; i < highSize-1; i++)
        {
            //printf("i=%i, r[i+1]=%1.f, r[i]=%1.f \n", i, r[i+1], r[i]);
            Dt_exc[i] = r[i+1] - r[i];
        }

        /*
        // median
        double medianOut;
        medianOut = median(r, highSize);
        */
         
        msDti1[j] = mean(Dt_exc, highSize-1);
        msDti3[j] = (highSize-1)*100.0/tot;
        msDti4[j] = median(r, highSize) / ((double)size/2) - 1;
        
        //printf("msDti1[%i] = %1.3f, msDti13[%i] = %1.3f, msDti4[%i] = %1.3f\n",
        //       j, msDti1[j], j, msDti3[j], j, msDti4[j]);
        
        // free(Dt_exc);
        
    }
    
    int trimthr = 2;
    int mj = 0;
    int fbi = nThresh-1;
    for(int i = 0; i < nThresh; i ++)
    {
        if (msDti3[i] > trimthr)
        {
            mj = i;
        }
        // if (isnan(msDti1[nThresh-1-i]))
        // {
        //     fbi = nThresh-1-i;
        // }
    }
    
    double outputScalar;
    int trimLimit = mj < fbi ? mj : fbi;
    outputScalar = median(msDti4, trimLimit+1);
    
    // free(r);
    // free(yWork);
    // free(msDti1);
    // free(msDti3);
    // free(msDti4);
    
    output << outputScalar;
}

double median(const double a[], const int size)
{
    double m;
    double b[DATA_SIZE];
    // memcpy(b, a, size * sizeof *b);
    for(int i = 0; i< DATA_SIZE; ++i){
        b[i] = a[i];
    }
    sort(b, size);
    if (size % 2 == 1) {
        m = b[size / 2];
    } else {
        int m1 = size / 2;
        int m2 = m1 - 1;
        m = (b[m1] + b[m2]) / (double)2.0;
    }
    // free(b);
    return m;
}



// Function to calculate the magnitude of a custom complex type
double complex_abs(const cmplx_type c) {
    // Direct multiplication instead of pow
    double real_sq = c.real * c.real;
    double imag_sq = c.imag * c.imag;
    return std::sqrt(real_sq + imag_sq);
}

bool is_infinite(float x) {
    // Use the IEEE 754 single-precision floating-point format
    // Interpret the floating-point number as an integer
    union {
        float f;
        unsigned int u;
    } conv;
    conv.f = x;
    
    // Extract the exponent bits (8 bits starting from bit 23)
    unsigned int exponent = (conv.u >> 23) & 0xFF;
    unsigned int mantissa = conv.u & 0x7FFFFF;  // 23-bit mantissa
    
    // Check if the exponent is all 1s and the mantissa is zero
    if (exponent == 0xFF && mantissa == 0) {
        return true;
    }
    return false;
}



double SP_Summaries_welch_rect_area_5_1(const double y[], const int size)
{
    return SP_Summaries_welch_rect(y, size, "area_5_1");
}
double SP_Summaries_welch_rect_centroid(const double y[], const int size)
{
    return SP_Summaries_welch_rect(y, size, "centroid");
    
}

int welch(const double y[], const int size, const int NFFT, const double Fs, const double window[], const int windowWidth, double *Pxx, double* f){
    
    // #undef N
    // #define N 128
    // #undef R
    // #define R 2
    // #undef SW
    // #define SW 4
    // #undef LOGrN
    // #define LOGrN 7
    // #undef LOG2N
    // #define LOG2N 7
    // #undef LOG2R 
    // #define LOG2R 1
    // #undef LOG2SW 
    // #define LOG2SW 2
    // #undef LOG2_LOGrN 
    // #define LOG2_LOGrN 3
    double dt = 1.0/Fs;
    double df = 1.0/(nextpow2(windowWidth))/dt;
    double m = mean(y, size);
    
    // number of windows, should be 1
    int k = floor((double)size/((double)windowWidth/2.0))-1;
    
    // normalising scale factor
    double KMU = k * pow(norm_(window, windowWidth),2);
    
    double P[DATA_SIZE];
    for(int i = 0; i < NFFT; i++){
        P[i] = 0;
    }
    
    // fft variables
    cmplx_type F[DATA_SIZE];
    cmplx_type tw[DATA_SIZE];
    // twiddles(tw, NFFT);
    
    double xw[DATA_SIZE];
    for(int i = 0; i<k; i++){
        
        // apply window
        for(int j = 0; j<windowWidth; j++){
            xw[j] = window[j]*y[j + (int)(i*(double)windowWidth/2.0)];
        }
        
        // initialise F (
        for (int i = 0; i < windowWidth; i++) {
            F[i].real = y[i] - m;
            F[i].imag = 0.0; 
            
	    // // #if defined(__GNUC__) || defined(__GNUG__)
		// // cplx tmp = xw[i] - m + 0.0 * I;
	    // // #elif defined(_MSC_VER)
		//     cplx tmp = { xw[i] - m, 0.0 };
	    // // #endif
            
            
        //     F[i] = tmp; // CMPLX(xw[i] - m, 0.0);
    
        }
        for (int i = windowWidth; i < NFFT; i++) {
            // F[i] = CMPLX(0.0, 0.0);
            //cplx tmp = { 0.0, 0.0 };
        //     #if defined(__GNUC__) || defined(__GNUG__)
		// cplx tmp = 0.0 + 0.0 * I;
	    // #elif defined(_MSC_VER)
		    F[i].real = 0.0;
            F[i].imag = 0.0; 
        }
        
        pease_fft_N(F, tw);
        
        /*
        for(int i = 0; i < NFFT; i++){
            printf("F1[%i] real: %1.3f, imag: %1.3f\n", i, creal(F[i]), cimag(F[i]));
        }
         */
        // std::cout << "kernelF0" << F[0];
        // std::cout << "kernelF1" << F[1];
        
        for(int l = 0; l < NFFT; l++){
            double magnitude_sq = std::pow(complex_abs(tw[l]), 2);
            P[l] += round(magnitude_sq,5);
        }
        std::cout << "kernelp0" << P[0];
        std::cout << "kernelp1" << P[1];
        /*
        for(int i = 0; i < NFFT; i++){
            printf("P[%i]: %1.3f\n", i, P[i]);
        }
         */
        
    }
    
    int Nout = (NFFT/2+1);
    for(int i = 0; i < Nout; i++){
        Pxx[i] = round(P[i]/KMU*dt,5);
        if(i>0 & i < Nout-1){
            Pxx[i] *= 2;
        }
    }


    /*
    for(int i = 0; i < Nout; i++){
        printf("Pxx[%i]: %1.3f\n", i, Pxx[i]);
    }
     */
    
    for(int i = 0; i < Nout; i++){
        f[i] = round(((double)i)*df,5);
    }
    std::cout << "kernelf0" << f[0];
    std::cout << "kernelf1" << f[1];
    /*
    for(int i = 0; i < Nout; i++){
        printf("f[%i]: %1.3f\n", i, (*f)[i]);
    }
     */
    
    // free(P);
    // free(F);
    // free(tw);
    // free(xw);
    
    return Nout;
}
//int 0 1 
double SP_Summaries_welch_rect(const double y[], const int size, const char what[])
{
    
    // NaN check
    // for(int i = 0; i < size; i++)
    // {
    //     if(isnan(y[i]))
    //     {
    //         return NAN;
    //     }
    // }
    
    // rectangular window for Welch-spectrum
    double window[DATA_SIZE];
    for(int i = 0; i < size; i++){
        window[i] = 1;
    }
    
    double Fs = 1.0; // sampling frequency
    const int N2 = DATA_SIZE;
    
    double S[DATA_SIZE];
    double f[DATA_SIZE];
    
    // compute Welch-power
    int nWelch = welch(y, size, N2, Fs, window, size, S, f);
    // free(window);
    
    // angualr frequency and spectrum on that
    double w[DATA_SIZE];
    double Sw[DATA_SIZE];
    
    double PI = 3.14;
    for(int i = 0; i < nWelch; i++){
        w[i] = round(2*PI*f[i],1);
        Sw[i] = round(S[i]/(2*PI),1);
        //printf("w[%i]=%1.3f, Sw[%i]=%1.3f\n", i, w[i], i, Sw[i]);
        // if(is_infinite(Sw[i]) | is_infinite(-Sw[i])){
        //     return 0;
        // }
    }
    
    double dw = w[1] - w[0];
    
    double csS[DATA_SIZE];
    cumsum(Sw, nWelch, csS);
    /*
    for(int i=0; i<nWelch; i++)
    {
        printf("csS[%i]=%1.3f\n", i, csS[i]);
    }
     */
    
    double output = 0;
    
    if(simple_strcmp(what, "centroid") == 0){
        
        double csSThres = csS[nWelch-1]*0.5;
        double centroid = 0;
        for(int i = 0; i < nWelch; i ++){
            if(csS[i] > csSThres){
                centroid = round(w[i],5);
                break;
            }
        }
        
        output = centroid;
        
    }
    else if(simple_strcmp(what, "area_5_1") == 0){
        double area_5_1 = 0;;
        for(int i=0; i<nWelch/5; i++){
            area_5_1 += Sw[i];
        }
        area_5_1 *= dw;
        
        output = round(area_5_1,5);
    }
    
    // free(w);
    // free(Sw);
    // free(csS);
    // free(f);
    // free(S);
    
    return round(output,5);
    
    
}

void cumsum(const double a[], const int size, double b[])
{
    b[0] = a[0];
    for (int i = 1; i < size; i++) {
        b[i] = a[i] + b[i-1];
        //printf("b[%i]%1.3f = a[%i]%1.3f + b[%i-1]%1.3f\n", i, b[i], i, a[i], i, a[i-1]);
    }
    
}

int nextpow2(int n)
{
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}


double FC_LocalSimple_mean3_stderr(const double y[DATA_SIZE], const int size)
{
    return FC_LocalSimple_mean_stderr(y, size, 3);
}

double FC_LocalSimple_mean_stderr(const double y[DATA_SIZE], const int size, const int train_length)
{
    // // NaN check
    // for(int i = 0; i < size; i++)
    // {
    //     if(isnan(y[i]))
    //     {
    //         return NAN;
    //     }
    // }
    
    double res[DATA_SIZE];
    
    for (int i = 0; i < size - train_length; i++)
    {
        double yest = 0;
        for (int j = 0; j < train_length; j++)
        {
            yest += y[i+j];
            
        }
        yest /= train_length;
        
        res[i] = y[i+train_length] - yest;
    }
    
    double output = stddev(res, size - train_length);
    
    return output;
    
}


int PD_PeriodicityWang_th0_01(const double y[], const int size){
    
    
    const double th = 0.01;
    
    double ySpline[DATA_SIZE];
    
    // fit a spline with 3 nodes to the data
    splinefit(y, size, ySpline);
    
    //printf("spline fit complete.\n");
    
    // subtract spline from data to remove trend
    double ySub[DATA_SIZE];
    for(int i = 0; i < size; i++){
        ySub[i] = y[i] - ySpline[i];
        //printf("ySub[%i] = %1.5f\n", i, ySub[i]);
    }
    
    // compute autocorrelations up to 1/3 of the length of the time series
    int acmax = (int)ceil((double)size/3);
    double acf[int(DATA_SIZE/3)+1];
    for(int tau = 1; tau <= acmax; tau++){
        // correlation/ covariance the same, don't care for scaling (cov would be more efficient)
        acf[tau-1] = autocov_lag(ySub, size, tau);
        //printf("acf[%i] = %1.9f\n", tau-1, acf[tau-1]);
    }
    
    //printf("ACF computed.\n");
    
    // find troughts and peaks
    double troughs[int(DATA_SIZE/3)+1];
    double peaks[int(DATA_SIZE/3)+1];
    int nTroughs = 0;
    int nPeaks = 0;
    double slopeIn = 0;
    double slopeOut = 0;
    for(int i = 1; i < acmax-1; i ++){
        slopeIn = acf[i] - acf[i-1];
        slopeOut = acf[i+1] - acf[i];
        
        if(slopeIn < 0 & slopeOut > 0)
        {
            // printf("trough at %i\n", i);
            troughs[nTroughs] = i;
            nTroughs += 1;
        }
        else if(slopeIn > 0 & slopeOut < 0)
        {
            // printf("peak at %i\n", i);
            peaks[nPeaks] = i;
            nPeaks += 1;
        }
    }
    
    //printf("%i troughs and %i peaks found.\n", nTroughs, nPeaks);
    
    
    // search through all peaks for one that meets the conditions:
    // (a) a trough before it
    // (b) difference between peak and trough is at least 0.01
    // (c) peak corresponds to positive correlation
    int iPeak = 0;
    double thePeak = 0;
    int iTrough = 0;
    double theTrough = 0;
    
    int out = 0;
    
    for(int i = 0; i < nPeaks; i++){
        iPeak = peaks[i];
        thePeak = acf[iPeak];
        
        //printf("i=%i/%i, iPeak=%i, thePeak=%1.3f\n", i, nPeaks-1, iPeak, thePeak);
        
        // find trough before this peak
        int j = -1;
        while(troughs[j+1] < iPeak && j+1 < nTroughs){
            // printf("j=%i/%i, iTrough=%i, theTrough=%1.3f\n", j+1, nTroughs-1, (int)troughs[j+1], acf[(int)troughs[j+1]]);
            j++;
        }
        if(j == -1)
            continue;
        
        iTrough = troughs[j];
        theTrough = acf[iTrough];
        
        // (a) should be implicit
        
        // (b) different between peak and trough it as least 0.01
        if(thePeak - theTrough < th)
            continue;
        
        // (c) peak corresponds to positive correlation
        if(thePeak < 0)
            continue;
        
        // use this frequency that first fulfils all conditions.
        out = iPeak;
        break;
    }
    
    //printf("Before freeing stuff.\n");
    

    
    return out;
    
}

int iLimit(int x, int lim){
    return x<lim ? x : lim;
}

void icumsum(const int a[DATA_SIZE], const int size, int b[DATA_SIZE])
{
    b[0] = a[0];
    for (int i = 1; i < size; i++) {
        b[i] = a[i] + b[i-1];
        //printf("b[%i]%1.3f = a[%i]%1.3f + b[%i-1]%1.3f\n", i, b[i], i, a[i], i, a[i-1]);
    }
    
}

void lsqsolve_sub(const int sizeA1, const int sizeA2, const double A[DATA_SIZE*DATA_SIZE], const int sizeb, const double b[DATA_SIZE], double x[DATA_SIZE])
//void lsqsolve_sub(int sizeA1, int sizeA2, double A[sizeA1][sizeA2], int sizeb, double b[sizeb], double x[sizeA1])
{
    // create temp matrix and vector
    /*
    double *AT[sizeA1*sizeA2];
    for (int i = 0; i < sizeA2; i++)
        AT[i] = (double *)malloc(sizeA1 * sizeof(double));
    double *ATA[sizeA2];
    for (int i = 0; i < sizeA2; i++)
        ATA[i] = (double *)malloc(sizeA2 * sizeof(double));
    double * ATb = malloc(sizeA1 * sizeof(double));
     */
    
    double AT[DATA_SIZE][DATA_SIZE];
    double ATA[DATA_SIZE][DATA_SIZE];
    double ATb[5];
   
    double k[DATA_SIZE][DATA_SIZE];

     for(int i = 0; i < sizeA1; i++){
        for(int j = 0; j < sizeA2; j++){
            k[i][j] = A[i*DATA_SIZE+j];
        }
    }

    
    for(int i = 0; i < sizeA1; i++){
        for(int j = 0; j < sizeA2; j++){
            //AT[i,j] = A[j,i]
            AT[i][j] = k[j][i];
        }
    }
    
    /*
    printf("\n b \n");
    for(int i = 0; i < sizeA1; i++){
        printf("%i, %1.3f\n", i, b[i]);
    }
     */
    
    /*
    printf("\nA\n");
     for(int i = 0; i < sizeA2; i++){
         for(int j = 0; j < sizeA1; j++){
             printf("%1.3f, ", AT[i * sizeA1 + j]);
         }
         printf("\n");
     }
     */
     
    
    matrix_multiply(sizeA2, sizeA1, AT, sizeA1, sizeA2, k, ATA);
    
    /*
    printf("ATA\n");
    for(int i = 0; i < sizeA2; i++){
        for(int j = 0; j < sizeA2; j++){
            printf("%1.3f, ", ATA[i * sizeA2 + j]);
        }
        printf("\n");
    }
     */
    
    
    
    matrix_times_vector(sizeA2, sizeA1, AT, sizeA1, b, ATb);
    
    /*
    for(int i = 0; i < sizeA2; i++){
        ATb[i] = 0;
        for(int j = 0; j < sizeA1; j++){
            ATb[i] += AT[i*sizeA1 + j]*b[j];
            //printf("%i, ATb[%i]=%1.3f, AT[i*sizeA1 + j]=%1.3f, b[j]=%1.3f\n", i, i, ATb[i], AT[i*sizeA1 + j],b[j]);
        }
    }
     */
    
    /*
     for(int i = 0; i < nCoeffs; i++){
     printf("b[%i] = %1.3f\n", i, b[i]);
     }
     */
    
    /*
    for(int i = 0; i < sizeA2; i++){
        printf("ATb[%i] = %1.3f\n", i, ATb[i]);
    }
     */
    
    
    gauss_elimination(sizeA2, ATA, ATb, x);
    
   
    
}

void matrix_multiply(const int sizeA1, const int sizeA2, const double A[DATA_SIZE][DATA_SIZE], const int sizeB1, const int sizeB2, const double B[DATA_SIZE][DATA_SIZE], double C[DATA_SIZE][DATA_SIZE]){
//void matrix_multiply(int sizeA1, int sizeA2, double **A, int sizeB1, int sizeB2, double **B, double C[sizeA1][sizeB2]){
    
    if(sizeA2 != sizeB1){
        return;
    }
    
    /*
    // show input
    for(int i = 0; i < sizeA1; i++){
        for(int j = 0; j < sizeA2; j++){
            printf("A[%i][%i] = %1.3f\n", i, j, A[i*sizeA2 + j]);
        }
    }
     */
    
    for(int i = 0; i < sizeA1; i++){
        for(int j = 0; j < sizeB2; j++){
            
            C[i][j] = 0;
            for(int k = 0; k < sizeB1; k++){
                C[i][j] += A[i][k]*B[k][j];
                
            }
            
        }
    }
    
}

void matrix_times_vector(const int sizeA1, const int sizeA2, const double A[DATA_SIZE][DATA_SIZE], const int sizeb, const double b[DATA_SIZE], double c[DATA_SIZE]){ //c[sizeb]
    
    if(sizeA2 != sizeb){
        return;
    }
    
    // row
    for(int i = 0; i < sizeA1; i++){
        
        // column
        c[i] = 0;
        for(int k = 0; k < sizeb; k++){
            c[i] += A[i][k]*b[k];
        }
        
    }
    
}

void gauss_elimination(int size, double A[DATA_SIZE][DATA_SIZE], double b[DATA_SIZE], double x[DATA_SIZE]){
// void gauss_elimination(int size, double A[size][size], double b[size], double x[size]){
    
    double factor;
    
    // create temp matrix and vector
    // double *AElim[size];
    double AElim[DATA_SIZE][DATA_SIZE];
    
    double bElim[DATA_SIZE];
    
    // -- create triangular matrix
    
    // initialise to A and b
    for(int i = 0; i < size; i++){
        for(int j = 0; j < size; j++){
            AElim[i][j] = A[i][j];
        }
        bElim[i] = b[i];
    }
    
    /*
    printf("AElim\n");
    for(int i = 0; i < size; i++){
        for(int j = 0; j < size; j++){
            printf("%1.3f, ", AElim[i][j]);
        }
        printf("\n");
    }
     */
    
    // go through columns in outer loop
    for(int i = 0; i < size; i++){
    
        // go through rows to eliminate
        for(int j = i+1; j < size; j++){
            
            factor = AElim[j][i]/AElim[i][i];
            
            // subtract in vector
            bElim[j] = bElim[j] - factor*bElim[i];
            
            // go through entries of this row
            for(int k = i; k < size; k++){
                AElim[j][k] = AElim[j][k] - factor*AElim[i][k];
            }
            
            /*
            printf("AElim i=%i, j=%i\n", i, j);
            for(int i = 0; i < size; i++){
                for(int j = 0; j < size; j++){
                    printf("%1.3f, ", AElim[i][j]);
                }
                printf("\n");
            }
             */
            
        }
        
    }
    
    /*
    for(int i = 0; i < size; i++){
        for(int j = 0; j < size; j++){
            printf("AElim[%i][%i] = %1.3f\n", i, j, AElim[i][j]);
        }
    }
    for(int i = 0; i < size; i++){
        printf("bElim[%i] = %1.3f\n", i, bElim[i]);
    }
     */
    
    
    // -- go backwards through triangular matrix and solve for x
    
    // row
    double bMinusATemp;
    for(int i = size-1; i >= 0; i--){
        
        bMinusATemp = bElim[i];
        for(int j = i+1; j < size; j++){
            bMinusATemp -= x[j]*AElim[i][j];
        }
        
        x[i] = bMinusATemp/AElim[i][i];
    }
    /*
    for(int j = 0; j < size; j++){
        printf("x[%i] = %1.3f\n", j, x[j]);
    }
     */
    
}





int splinefit(const double y[DATA_SIZE], const int size, double yOut[])
{
    // degree of spline
    //const int nSpline = 4;
    //const int deg = 3;
    
    // x-positions of spline-nodes
    //const int nBreaks = 3;
    int breaks[nBreaks];
    breaks[0] = 0;
    breaks[1] = (int)floor((double)size/2.0)-1;
    breaks[2] = size-1;
    
    // -- splinebase
    
    // spacing
    int h0[2];
    h0[0] = breaks[1] - breaks[0];
    h0[1] = breaks[2] - breaks[1];
    
    //const int pieces = 2;
    
    // repeat spacing
    int hCopy[4];
    hCopy[0] = h0[0], hCopy[1] = h0[1], hCopy[2] = h0[0], hCopy[3] = h0[1];
    
    // to the left
    int hl[deg];
    hl[0] = hCopy[deg-0];
    hl[1] = hCopy[deg-1];
    hl[2] = hCopy[deg-2];
    
    int hlCS[deg]; // cumulative sum
    icumsum(hl, deg, hlCS);
    
    int bl[deg];
    for(int i = 0; i < deg; i++){
        bl[i] = breaks[0] - hlCS[i];
    }
    
    // to the left
    int hr[deg];
    hr[0] = hCopy[0];
    hr[1] = hCopy[1];
    hr[2] = hCopy[2];
    
    int hrCS[deg]; // cumulative sum
    icumsum(hr, deg, hrCS);
    
    int br[deg];
    for(int i = 0; i < deg; i++){
        br[i] = breaks[2] + hrCS[i];
    }
    
    // add breaks
    int breaksExt[3*deg];
    for(int i = 0; i < deg; i++){
        breaksExt[i] = bl[deg-1-i];
        breaksExt[i + 3] = breaks[i];
        breaksExt[i + 6] = br[i];
    }
    int hExt[3*deg-1];
    for(int i = 0; i < deg*3-1; i++){
        hExt[i] = breaksExt[i+1] - breaksExt[i];
    }
    //const int piecesExt = 3*deg-1;
    
    // initialise polynomial coefficients
    double coefs[nSpline*piecesExt][nSpline+1];
    for(int i = 0; i < nSpline*piecesExt; i++){
        for(int j = 0; j < nSpline; j++){
        coefs[i][j] = 0;
        }
    }
    for(int i = 0; i < nSpline*piecesExt; i=i+nSpline){
        coefs[i][0] = 1;
    }
    
    // expand h using the index matrix ii
    int ii[deg+1][piecesExt];
    for(int i = 0; i < piecesExt; i++){
        ii[0][i] = iLimit(0+i, piecesExt-1);
        ii[1][i] = iLimit(1+i, piecesExt-1);
        ii[2][i] = iLimit(2+i, piecesExt-1);
        ii[3][i] = iLimit(3+i, piecesExt-1);
    }
    
    // expanded h
    double H[(deg+1)*piecesExt];
    int iiFlat;
    for(int i = 0; i < nSpline*piecesExt; i++){
        iiFlat = ii[i%nSpline][i/nSpline];
        H[i] = hExt[iiFlat];
    }
    
    //recursive generation of B-splines
    double Q[nSpline][piecesExt];
    for(int k = 1; k < nSpline; k++){
        
        //antiderivatives of splines
        for(int j = 0; j<k; j++){
            for(int l = 0; l<(nSpline*piecesExt); l++){
                coefs[l][j] *= H[l]/(k-j);
                //printf("coefs[%i][%i]=%1.3f\n", l, j, coefs[l][j]);
            }
        }
        
        for(int l = 0; l<(nSpline*piecesExt); l++){
            Q[l%nSpline][l/nSpline] = 0;
            for(int m = 0; m < nSpline; m++){
                Q[l%nSpline][l/nSpline] += coefs[l][m];
            }
        }
        
        /*
        printf("\nQ:\n");
        for(int i = 0; i < n; i++){
            for(int j = 0; j < piecesExt; j ++){
                printf("%1.3f, ", Q[i][j]);
            }
            printf("\n");
        }
        */
        
        //cumsum
        for(int l = 0; l<piecesExt; l++){
            for(int m = 1; m < nSpline; m++){
                Q[m][l] += Q[m-1][l];
            }
        }
        
        /*
        printf("\nQ cumsum:\n");
        for(int i = 0; i < n; i++){
            for(int j = 0; j < piecesExt; j ++){
                printf("%1.3f, ", Q[i][j]);
            }
            printf("\n");
        }
        */
        
        for(int l = 0; l<nSpline*piecesExt; l++){
            if(l%nSpline == 0)
                coefs[l][k] = 0;
            else{
                coefs[l][k] = Q[l%nSpline-1][l/nSpline]; // questionable
            }
            // printf("coefs[%i][%i]=%1.3f\n", l, k, coefs[l][k]);
        }
        
        // normalise antiderivatives by max value
        double fmax[piecesExt*nSpline];
        for(int i = 0; i < piecesExt; i++){
            for(int j = 0; j < nSpline; j++){
                
                fmax[i*nSpline+j] = Q[nSpline-1][i];
                
            }
        }
        
        /*
        printf("\n fmax:\n");
        for(int i = 0; i < piecesExt*n; i++){
            printf("%1.3f, \n", fmax[i]);
        }
        */
        
        for(int j = 0; j < k+1; j++){
            for(int l = 0; l < nSpline*piecesExt; l++){
                coefs[l][j] /= fmax[l];
                // printf("coefs[%i][%i]=%1.3f\n", l, j, coefs[l][j]);
            }
        }

        // diff to adjacent antiderivatives
        for(int i = 0; i < (nSpline*piecesExt)-deg; i++){
            for(int j = 0; j < k+1; j ++){
                coefs[i][j] -= coefs[deg+i][j];
                //printf("coefs[%i][%i]=%1.3f\n", i, j, coefs[i][j]);
            }
        }
        for(int i = 0; i < nSpline*piecesExt; i += nSpline){
            coefs[i][k] = 0;
        }
        
        /*
        printf("\ncoefs:\n");
        for(int i = 0; i < (n*piecesExt); i++){
            for(int j = 0; j < n; j ++){
                printf("%1.3f, ", coefs[i][j]);
            }
            printf("\n");
        }
        */
        
    }
    
    // scale coefficients
    double scale[nSpline*piecesExt];
    for(int i = 0; i < nSpline*piecesExt; i++)
    {
        scale[i] = 1;
    }
    for(int k = 0; k < nSpline-1; k++){
        for(int i = 0; i < (nSpline*piecesExt); i++){
            scale[i] /= H[i];
        }
        for(int i = 0; i < (nSpline*piecesExt); i++){
            coefs[i][(nSpline-1)-(k+1)] *= scale[i];
        }
    }
    
    /*
    printf("\ncoefs scaled:\n");
    for(int i = 0; i < (n*piecesExt); i++){
        for(int j = 0; j < n; j ++){
            printf("%1.4f, ", coefs[i][j]);
        }
        printf("\n");
    }
    */
    
    // reduce pieces and sort coefficients by interval number
    int jj[nSpline][pieces];
    for(int i = 0; i < nSpline; i++){
        for(int j = 0; j < pieces; j++){
            if(i == 0)
                jj[i][j] = nSpline*(1+j);
            else
                jj[i][j] = deg;
        }
    }
    
    /*
    printf("\n jj\n");
    for(int i = 0; i < n; i++){
        for(int j = 0; j < pieces; j++){
            printf("%i, ", jj[i][j]);
        }
        printf("\n");
    }
    */
        
    
    for(int i = 1; i < nSpline; i++){
        for(int j = 0; j < pieces; j++){
            jj[i][j] += jj[i-1][j];
        }
    }
    
    /*
    printf("\n jj cumsum\n");
    for(int i = 0; i < n; i++){
        for(int j = 0; j < pieces; j++){
            printf("%i, ", jj[i][j]);
        }
        printf("\n");
    }
    */
    
    double coefsOut[nSpline*pieces][nSpline];
    int jj_flat;
    for(int i = 0; i < nSpline*pieces; i++){
        jj_flat = jj[i%nSpline][i/nSpline]-1;
        //printf("jj_flat(%i) = %i\n", i, jj_flat);
        for(int j = 0; j < nSpline; j++){
            coefsOut[i][j] = coefs[jj_flat][j];
            //printf("coefsOut[%i][%i]=%1.3f\n", i, j, coefsOut[i][j]);
        }
        
    }
    
    /*
    printf("\n coefsOut * 1000\n");
    for(int i = 0; i < n*pieces; i++){
        for(int j = 0; j < n; j++){
            printf("%1.3f, ", coefsOut[i][j]*1000);
        }
        printf("\n");
    }
     */
    
    
    // -- create first B-splines to feed into optimization
    
    // x-values for B-splines
    int xsB[DATA_SIZE*nSpline];
    int indexB[DATA_SIZE*nSpline];
   
    
    int breakInd = 1;
    for(int i = 0; i < size; i++){
        if(i >= breaks[breakInd] & breakInd<nBreaks-1)
            breakInd += 1;
        for(int j = 0; j < nSpline; j++){
            xsB[i*nSpline+j] = i - breaks[breakInd-1];
            indexB[i*nSpline+j] = j + (breakInd-1)*nSpline;
        }
    }
    
    /*
    printf("\nxsB\n");
    for(int i = 0; i < size*n; i++){
        printf("%i, %i\n", i, xsB[i]);
    }
    printf("\nindexB\n");
    for(int i = 0; i < size*n; i++){
        printf("%i, %i\n", i, indexB[i]);
    }
     */
    
    double vB[DATA_SIZE*nSpline];
    for(int i = 0; i < size*nSpline; i++){
        vB[i] = coefsOut[indexB[i]][0];
    }
    
    /*
    printf("\nvB first iteration\n");
    for(int i = 0; i < size*n; i++){
        printf("%i, %1.4f\n", i, vB[i]);
    }
     */
    
    for(int i = 1; i < nSpline; i ++){
        for(int j = 0; j < size*nSpline; j++){
            vB[j] = vB[j]*xsB[j] + coefsOut[indexB[j]][i];
        }
        /*
        printf("\nvB k=%i\n", i+1);
        for(int i = 0; i < size*n; i++){
            printf("%i, %1.4f\n", i, vB[i]);
        }
         */
    }
    
    /*
    printf("\nvB final\n");
    for(int i = 0; i < size*n; i++){
        printf("%i, %1.4f\n", i, vB[i]);
    }
     */
    
    
    double A[DATA_SIZE*(nSpline+1)];
    
    for(int i = 0; i < (nSpline+1)*size; i++){
        A[i] = 0;
    }
    breakInd = 0;
    for(int i = 0; i < nSpline*size; i++){
        if(i/nSpline >= breaks[1])
            breakInd = 1;
        A[(i%nSpline)+breakInd + (i/nSpline)*(nSpline+1)] = vB[i];
    }
    
    /*
    printf("\nA:\n");
    for(int i = 0; i < size; i++){
        for(int j = 0; j < n+1; j++){
            printf("%1.5f, ", A[i * (n+1) + j]);
        }
        printf("\n");
    }
     */
    
    
    
    double x[nSpline+1];
    // lsqsolve_sub(int sizeA1, int sizeA2, double *A, int sizeb, double *b, double *x)
    lsqsolve_sub(size, nSpline+1, A, size, y, x);
    
    /*
    printf("\nsolved x\n");
    for(int i = 0; i < n+1; i++){
        printf("%i, %1.4f\n", i, x[i]);
    }
    */
    
    // coeffs of B-splines to combine by optimised weighting in x
    double C[pieces+nSpline-1][nSpline*pieces];
    // initialise to 0
    for(int i = 0; i < nSpline+1; i++){
        for(int j = 0; j < nSpline*pieces; j++){
            C[i][j] = 0;
        }
    }
    
    int CRow, CCol, coefRow, coefCol;
    for(int i = 0; i < nSpline*nSpline*pieces; i++){
        
        CRow = i%nSpline + (i/nSpline)%2;
        CCol = i/nSpline;
        
        coefRow = i%(nSpline*2);
        coefCol =i/(nSpline*2);
        
        C[CRow][CCol] = coefsOut[coefRow][coefCol];
        
    }
    
    /*
    printf("\nC:\n");
    for(int i = 0; i < n+1; i++){
        for(int j = 0; j < n*pieces; j++){
            printf("%1.5f, ", C[i][j]);
        }
        printf("\n");
    }
    */
    
    // final coefficients
    double coefsSpline[pieces][nSpline];
    for(int i = 0; i < pieces; i++){
        for(int j = 0; j < nSpline; j++){
            coefsSpline[i][j] = 0;
        }
    }
    
    //multiply with x
    for(int j = 0; j < nSpline*pieces; j++){
        coefCol = j/pieces;
        coefRow = j%pieces;
        
        for(int i = 0; i < nSpline+1; i++){
            
            coefsSpline[coefRow][coefCol] += C[i][j]*x[i];
            
        }
    }
    
    /*
    printf("\ncoefsSpline:\n");
    for(int i = 0; i < pieces; i++){
        for(int j = 0; j < n; j++){
            printf("%1.5f, ", coefsSpline[i][j]);
        }
        printf("\n");
    }
     */
     
    
    // compute piecewise polynomial
    
    int secondHalf = 0;
    for(int i = 0; i < size; i++){
        secondHalf = i < breaks[1] ? 0 : 1;
        yOut[i] = coefsSpline[secondHalf][0];
    }
    
    /*
    printf("\nvSpline first iter\n");
    for(int i = 0; i < size; i++){
        printf("%i, %1.5f\n", i, vSpline[i]);
    }
    */
    
    for(int i = 1; i < nSpline; i ++){
        for(int j = 0; j < size; j++){
            secondHalf = j < breaks[1] ? 0 : 1;
            yOut[j] = yOut[j]*(j - breaks[1]*secondHalf) + coefsSpline[secondHalf][i];
        }
        
        /*
        printf("\nvSpline %i th iter\n", i);
        for(int i = 0; i < size; i++){
            printf("%i, %1.4f\n", i, vSpline[i]);
        }
         */
    }
    
    /*
    printf("\nvSpline\n");
    for(int i = 0; i < size; i++){
        printf("%i, %1.4f\n", i, yOut[i]);
    }
     */
    
    return 0;
    
}

double cov_mean(const double x[DATA_SIZE], const double y[DATA_SIZE], const int size){
    
    double covariance = 0;
    
    for(int i = 0; i < size; i++){
        // double xi =x[i];
        // double yi =y[i];
        covariance += x[i] * y[i];
        
    }
    
    return covariance/size;
    
}

double autocov_lag(const double x[DATA_SIZE], const int size, const int lag){
    
    // Ensure 'lag' does not exceed the bounds of 'x'
    if (lag < 0 || lag >= size) return 0;

    double tempX[DATA_SIZE] = {0}; // Temporary array to hold shifted values
    for (int i = 0; i < size - lag; ++i) {
        tempX[i] = x[i + lag];
    }
    return cov_mean(x, tempX, size - lag);
}






double SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1(const double y[DATA_SIZE], const int size){
    return SC_FluctAnal_2_50_1_logi_prop_r1(y, size, 2, "dfa");
}

double SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1(const double y[DATA_SIZE], const int size){
    return SC_FluctAnal_2_50_1_logi_prop_r1(y, size, 1, "rsrangefit");
}



double SC_FluctAnal_2_50_1_logi_prop_r1(const double y[DATA_SIZE], const int size, const int lag, const char how[])
{
    // // NaN check
    // for(int i = 0; i < size; i++)
    // {
    //     if(isnan(y[i]))
    //     {
    //         return NAN;
    //     }
    // }
    
    // generate log spaced tau vector
    double linLow = log(5);
    double linHigh = log(size/2);
    
    int nTauSteps = 50;
    double tauStep = (linHigh - linLow) / (nTauSteps-1);
    
    int tau[50];
    for(int i = 0; i < nTauSteps; i++)
    {
        tau[i] = round(exp(linLow + i*tauStep));
    }
    
    // check for uniqueness, use ascending order
    int nTau = nTauSteps;
    for(int i = 0; i < nTauSteps-1; i++)
    {
        
        while (tau[i] == tau[i+1] && i < nTau-1)
        {
            for(int j = i+1; j < nTauSteps-1; j++)
            {
                tau[j] = tau[j+1];
            }
            // lost one
            nTau -= 1;
        }
    }
    
    // fewer than 12 points -> leave.
    if(nTau < 12){
        return 0;
    }
    
    int sizeCS = size/lag;
    double yCS[DATA_SIZE];
    
    /*
    for(int i = 0; i < 50; i++)
    {
        printf("y[%i]=%1.3f\n", i, y[i]);
    }
     */
    
    // transform input vector to cumsum
    yCS[0] = y[0];
    for(int i = 0; i < sizeCS-1; i++)
    {
        yCS[i+1] = yCS[i] + y[(i+1)*lag];
        
        /*
        if(i<300)
            printf("yCS[%i]=%1.3f\n", i, yCS[i]);
         */
    }
    
    //for each value of tau, cut signal into snippets of length tau, detrend and
    
    // first generate a support for regression (detrending)
    double xReg[DATA_SIZE];
    for(int i = 0; i < tau[nTau-1]; i++)
    {
        xReg[i] = i+1;
    }
    //first
    // iterate over taus, cut signal, detrend and save amplitude of remaining signal
    double F[DATA_SIZE];
    for(int i = 0; i < nTau; i++)
    {
        int nBuffer = sizeCS/tau[i];
        double buffer[DATA_SIZE];
        double m = 0.0, b = 0.0;
        
        //printf("tau[%i]=%i\n", i, tau[i]);
        
        F[i] = 0;
        for(int j = 0; j < nBuffer; j++)
        {
            
            //printf("%i th buffer\n", j);
            
            linreg(tau[i], xReg, yCS+j*tau[i], &m, &b);
            
            
            for(int k = 0; k < tau[i]; k++)
            {
                buffer[k] = yCS[j*tau[i]+k] - (m * (k+1) + b);
                //printf("buffer[%i]=%1.3f\n", k, buffer[k]);
            }
            
            if (simple_strcmp(how, "rsrangefit") == 0) {
                F[i] += pow(max_(buffer, tau[i]) - min_(buffer, tau[i]), 2);
            }
            else if (simple_strcmp(how, "dfa") == 0) {
                for(int k = 0; k<tau[i]; k++){
                    F[i] += buffer[k]*buffer[k];
                }
            }
            else{
                return 0.0;
            }
        }
        
        if (simple_strcmp(how, "rsrangefit") == 0) {
            F[i] = sqrt(F[i]/nBuffer);
        }
        else if (simple_strcmp(how, "dfa") == 0) {
            F[i] = sqrt(F[i]/(nBuffer*tau[i]));
        }
        //printf("F[%i]=%1.3f\n", i, F[i]);
        
    
        
    }
    
    double logtt[DATA_SIZE];
    double logFF[DATA_SIZE];
    int ntt = nTau;
    
    for (int i = 0; i < nTau; i++)
    {
        logtt[i] = log(tau[i]);
        logFF[i] = log(F[i]);
    }
    
    int minPoints = 6;
    int nsserr = ntt - 2*minPoints + 1;
    double sserr[DATA_SIZE];
    double buffer[DATA_SIZE];
    for (int i = minPoints; i < ntt - minPoints + 1; i++)
    {
        // this could be done with less variables of course
        double m1 = 0.0, b1 = 0.0;
        double m2 = 0.0, b2 = 0.0;
        
        sserr[i - minPoints] = 0.0;
        
        linreg(i, logtt, logFF, &m1, &b1);
        linreg(ntt-i+1, logtt+i-1, logFF+i-1, &m2, &b2);
        
        for(int j = 0; j < i; j ++)
        {
            buffer[j] = logtt[j] * m1 + b1 - logFF[j];
        }
        
        sserr[i - minPoints] += norm_(buffer, i);
        
        for(int j = 0; j < ntt-i+1; j++)
        {
            buffer[j] = logtt[j+i-1] * m2 + b2 - logFF[j+i-1];
        }
        
        sserr[i - minPoints] += norm_(buffer, ntt-i+1);
        
    }
    
    double firstMinInd = 0.0;
    double minimum = min_(sserr, nsserr);
    for(int i = 0; i < nsserr; i++)
    {
        if(sserr[i] == minimum)
        {
            firstMinInd = i + minPoints - 1;
            break;
        }
    }
    
   
    return (firstMinInd+1)/ntt;
    
}



double norm_(const double a[], const int size)
{
    
    double out = 0.0;
    
    for (int i = 0; i < size; i++)
    {
        out += a[i]*a[i];
    }
    
    out = sqrt(out);
    
    return out;
}





int linreg(const int n, const double x[], const double y[], double* m, double* b) //, double* r)
{
    double   sumx = 0.0;                      /* sum of x     */
    double   sumx2 = 0.0;                     /* sum of x**2  */
    double   sumxy = 0.0;                     /* sum of x * y */
    double   sumy = 0.0;                      /* sum of y     */
    double   sumy2 = 0.0;                     /* sum of y**2  */
    
    /*
    for (int i = 0; i < n; i++)
    {
        fprintf(stdout, "x[%i] = %f, y[%i] = %f\n", i, x[i], i, y[i]);
    }
    */
    
    for (int i=0;i<n;i++){
        sumx  += x[i];
        sumx2 += x[i] * x[i];
        sumxy += x[i] * y[i];
        sumy  += y[i];
        sumy2 += y[i] * y[i];
    }
    
    double denom = (n * sumx2 - sumx * sumx);
    if (denom == 0) {
        // singular matrix. can't solve the problem.
        *m = 0;
        *b = 0;
        //if (r) *r = 0;
        return 1;
    }
    
    *m = (n * sumxy  -  sumx * sumy) / denom;
    *b = (sumy * sumx2  -  sumx * sumxy) / denom;
    
    /*if (r!=NULL) {
        *r = (sumxy - sumx * sumy / n) /    // compute correlation coeff
        sqrt((sumx2 - sumx * sumx/n) *
             (sumy2 - sumy * sumy/n));
    }
    */
    
    return 0;
}




double DN_HistogramMode_5(const double y[DATA_SIZE], const int size) {
    double maxCount = 0;
    int numMaxs = 1;
    double out = 0;
    
    const int MAX_BINS = 5;
    int histCounts[MAX_BINS] = {0};  // Histogram counts array
    double binEdges[MAX_BINS + 1] = {0};  // Bin edges array

    // Calculate histogram
    histcounts(y, size, 5, histCounts, binEdges);  // 'size' parameter used to process the actual number of elements

    // Find the histogram bin(s) with the maximum count
    for(int i = 0; i < 5; i++) {  // Iterate through histogram bins
        if (histCounts[i] > maxCount) {
            maxCount = histCounts[i];
            numMaxs = 1;
            out = (binEdges[i] + binEdges[i+1]) * 0.5;  // Calculate the mid-point of the bin
        } else if (histCounts[i] == maxCount) {
            numMaxs += 1;
            out += (binEdges[i] + binEdges[i+1]) * 0.5;  // Sum mid-points for averaging
        }
    }
    out = out / numMaxs;  // Average mid-point of the mode bin(s)

    return out;
}

double DN_HistogramMode_10(const double y[DATA_SIZE], const int size) {
    double maxCount = 0;
    int numMaxs = 1;
    double out = 0;
    const int MAX_BINS = 10;
    int histCounts[MAX_BINS] = {0};  // Histogram counts array
    double binEdges[MAX_BINS + 1] = {0};  // Bin edges array

    // Calculate histogram
    histcounts(y, size, 10, histCounts, binEdges);  // 'size' parameter used to process the actual number of elements

    // Find the histogram bin(s) with the maximum count
    for(int i = 0; i < 10; i++) {  // Iterate through histogram bins
        if (histCounts[i] > maxCount) {
            maxCount = histCounts[i];
            numMaxs = 1;
            out = (binEdges[i] + binEdges[i+1]) * 0.5;  // Calculate the mid-point of the bin
        } else if (histCounts[i] == maxCount) {
            numMaxs += 1;
            out += (binEdges[i] + binEdges[i+1]) * 0.5;  // Sum mid-points for averaging
        }
    }
    out = out / numMaxs;  // Average mid-point of the mode bin(s)

    return out;
}


int histcounts(const double y[DATA_SIZE], const int size, int nBins, int binCounts[], double binEdges[]) {
    double minVal = DBL_MAX, maxVal = -DBL_MAX;
    for(int i = 0; i < size; i++) {  // Use 'size' to process only the relevant part of 'y[]'
        if (y[i] < minVal) minVal = y[i];
        if (y[i] > maxVal) maxVal = y[i];
    }

    double binStep = (maxVal - minVal) / nBins;
    for(int i = 0; i < nBins; i++) binCounts[i] = 0;

    for(int i = 0; i < size; i++) {  // Again, use 'size' to consider only valid elements
        int binInd = (int)((y[i] - minVal) / binStep);
        binInd = binInd < 0 ? 0 : (binInd >= nBins ? nBins - 1 : binInd);
        binCounts[binInd] += 1;
    }

    for(int i = 0; i < nBins + 1; i++) {
        binEdges[i] = i * binStep + minVal;
    }

    return nBins;
}




double SB_TransitionMatrix_3ac_sumdiagcov(double y[], int size)
{
    
    // NaN and const check
    int constant = 1;
    for(int i = 0; i < size; i++)
    {
        // if(isnan(y[i]))
        // {
        //     return NAN;
        // }
        if(y[i] != y[0]){
            constant = 0;
        }
    }
    // if (constant){
    //     return NAN;
    // }
    
    const int numGroups = 3;
    
    int tau = co_firstzero(y, size, size);
    
    double yFilt[DATA_SIZE];
    
    // sometimes causes problems in filt!!! needs fixing.
    /*
    if(tau > 1){
        butterworthFilter(y, size, 4, 0.8/tau, yFilt);
    }
    */
    
    for(int i = 0; i < size; i++){
        yFilt[i] = y[i];
    }
    
    /*
    for(int i = 0; i < size; i++){
        printf("yFilt[%i]=%1.4f\n", i, yFilt[i]);
    }
     */
    
    int nDown = (size-1)/tau+1;
    double yDown[DATA_SIZE];
    
    for(int i = 0; i < nDown; i++){
        yDown[i] = yFilt[i*tau];
    }
    
    /*
    for(int i = 0; i < nDown; i++){
        printf("yDown[%i]=%1.4f\n", i, yDown[i]);
    }
     */
    
    
    // transfer to alphabet
    int yCG[DATA_SIZE];
    sb_coarsegrain(yDown, nDown, "quantile", numGroups, yCG);
    
    /*
    for(int i = 0; i < nDown; i++){
        printf("yCG[%i]=%i\n", i, yCG[i]);
    }
     */
    
    
    double T[3][3];
    for(int i = 0; i < numGroups; i++){
        for(int j = 0; j < numGroups; j++){
            T[i][j] = 0;
        }
    }
    
    // more efficient way of doing the below 
    for(int j = 0; j < nDown-1; j++){
        T[yCG[j]-1][yCG[j+1]-1] += 1;
    }
    
    /*
    for(int i = 0; i < numGroups; i++){
        for(int j = 0; j < numGroups; j++){
            printf("%1.f, ", T[i][j]);
        }
        printf("\n");
    }
     */
    
    /*
    for(int i = 0; i < numGroups; i++){
        for(int j = 0; j < nDown-1; j++){
            if(yCG[j] == i+1){
                T[i][yCG[j+1]-1] += 1;
            }
        }
    }
     */
    
    for(int i = 0; i < numGroups; i++){
        for(int j = 0; j < numGroups; j++){
            T[i][j] /= (nDown-1);
            // printf("T(%i, %i) = %1.3f\n", i, j, T[i][j]);
            
        }
    }
    
    // double column1[3] = {0};
    // double column2[3] = {0};
    // double column3[3] = {0};
    
    // for(int i = 0; i < numGroups; i++){
    //     column1[i] = T[i][0];
    //     column2[i] = T[i][1];
    //     column3[i] = T[i][2];
    //     // printf("column3(%i) = %1.3f\n", i, column3[i]);
    // }
    
    // double *columns[3];
    // columns[0] = &(column1[0]);
    // columns[1] = &(column2[0]);
    // columns[2] = &(column3[0]);

    double columns[3][3];
    for (int i = 0; i < numGroups; i++) {
        columns[0][i] = T[i][0];
        columns[1][i] = T[i][1];
        columns[2][i] = T[i][2];
        // Print statement could be replaced with non-std I/O if needed
        // std::cout << "column3(" << i << ") = " << std::setprecision(3) << columns[2][i] << std::endl;
    }
    
    
    double COV[3][3];
    double covTemp = 0;
    for(int i = 0; i < numGroups; i++){
        for(int j = i; j < numGroups; j++){
            
            covTemp = cov(columns[i], columns[j], 3);
            
            COV[i][j] = covTemp;
            COV[j][i] = covTemp;
            
            // printf("COV(%i , %i) = %1.3f\n", i, j, COV[i][j]);
        }
    }
    
    double sumdiagcov = 0;
    for(int i = 0; i < numGroups; i++){
        sumdiagcov += COV[i][i];
    }
    
    
    return sumdiagcov;
    
    
}

double cov(const double x[], const double y[], const int size){
    
    double covariance = 0;
    
    double meanX = mean(x, size);
    double meanY = mean(y, size);
    
    for(int i = 0; i < size; i++){
        // double xi =x[i];
        // double yi =y[i];
        covariance += (x[i] - meanX) * (y[i] - meanY);
        
    }
    
    return covariance/(size-1);
    
}



double SB_MotifThree_quantile_hh(const double y[], const int size)
{
    // NaN check
    // for(int i = 0; i < size; i++)
    // {
    //     if(isnan(y[i]))
    //     {
    //         return NAN;
    //     }
    // }
    
    int tmp_idx, r_idx;
    int dynamic_idx;
    int alphabet_size = 3;
    int array_size;
    int yt[DATA_SIZE]; // alphabetized array
    double hh; // output
    double out[124]; // output array
    
    // transfer to alphabet
    sb_coarsegrain(y, size, "quantile", 3, yt);
    
    // words of length 1
    array_size = alphabet_size;
    int r1[3][DATA_SIZE];
    int sizes_r1[3];
    double out1[3];
    for (int i = 0; i < alphabet_size; i++) {
        // r1[i] = malloc(size * sizeof(r1[i])); // probably can be rewritten
        // using selfresizing array for memory efficiency. Time complexity
        // should be comparable due to ammotization.
        r_idx = 0;
        sizes_r1[i] = 0;
        for (int j = 0; j < size; j++) {
            if (yt[j] == i + 1) {
                r1[i][r_idx++] = j;
                sizes_r1[i]++;
            }
        }
    }
    
    // words of length 2
    array_size *= alphabet_size;
    // removing last item if it is == max possible idx since later we are taking idx + 1
    // from yt
    for (int i = 0; i < alphabet_size; i++) {
        if (sizes_r1[i] != 0 && r1[i][sizes_r1[i] - 1] == size - 1) {
            //int * tmp_ar = malloc((sizes_r1[i] - 1) * sizeof(tmp_ar));
            int tmp_ar[DATA_SIZE];
            subset(r1[i], tmp_ar, 0, sizes_r1[i]);
            // memcpy(r1[i], tmp_ar, (sizes_r1[i] - 1) * sizeof(tmp_ar));
            for (int l = 0; l < sizes_r1[i] - 1; l++) {
                r1[i][l] = tmp_ar[l];
            }
            sizes_r1[i]--;
        }
    }
    
    /*
    int *** r2 = malloc(array_size * sizeof(**r2));
    int ** sizes_r2 = malloc(array_size * sizeof(*sizes_r2));
    double ** out2 = malloc(array_size * sizeof(*out2));
    */
    int r2[3][3][DATA_SIZE];
    int sizes_r2[3][3];
    double out2[3][3];
    

    // allocate separately
    // for (int i = 0; i < alphabet_size; i++) {
    //     r2[i] = malloc(alphabet_size * sizeof(*r2[i]));
    //     sizes_r2[i] = malloc(alphabet_size * sizeof(*sizes_r2[i]));
    //     //out2[i] = malloc(alphabet_size * sizeof(out2[i]));
    //     out2[i] = malloc(alphabet_size * sizeof(**out2));
    //     for (int j = 0; j < alphabet_size; j++) {
    //         r2[i][j] = malloc(size * sizeof(*r2[i][j]));
    //     }
    // }

    // fill separately
    for (int i = 0; i < alphabet_size; i++) {
    // for (int i = 0; i < array_size; i++) {
        //r2[i] = malloc(alphabet_size * sizeof(r2[i]));
        //sizes_r2[i] = malloc(alphabet_size * sizeof(sizes_r2[i]));
        //out2[i] = malloc(alphabet_size * sizeof(out2[i]));
        for (int j = 0; j < alphabet_size; j++) {
            //r2[i][j] = malloc(size * sizeof(r2[i][j]));
            sizes_r2[i][j] = 0;
            dynamic_idx = 0; //workaround as you can't just add elements to array
            // like in python (list.append()) for example, so since for some k there will be no adding,
            // you need to keep track of the idx at which elements will be inserted
            for (int k = 0; k < sizes_r1[i]; k++) {
                tmp_idx = yt[r1[i][k] + 1];
                if (tmp_idx == (j + 1)) {
                    r2[i][j][dynamic_idx++] = r1[i][k];
                    sizes_r2[i][j]++;
                    // printf("dynamic_idx=%i, size = %i\n", dynamic_idx, size);
                }
            }
            double tmp = (double)sizes_r2[i][j] / ((double)(size) - (double)(1.0));
            out2[i][j] =  tmp;
        }
    }

    hh = 0.0;
    for (int i = 0; i < alphabet_size; i++) {
        hh += f_entropy(out2[i], alphabet_size);
    }

    // free(yt);
    // free(out);
    // free(out1);

    // free(sizes_r1);

    // // free nested array
    // for (int i = 0; i < alphabet_size; i++) {
    //     free(r1[i]);
    // }
    // free(r1);
    // // free(sizes_r1);
    
    // for (int i = 0; i < alphabet_size; i++) {
    // //for (int i = alphabet_size - 1; i >= 0; i--) {

    //     free(sizes_r2[i]);
    //     free(out2[i]);
    // }

    // //for (int i = alphabet_size-1; i >= 0 ; i--) {
    // for(int i = 0; i < alphabet_size; i++) {
    //     for (int j = 0; j < alphabet_size; j++) {
    //         free(r2[i][j]);
    //     }
    //     free(r2[i]);
    // }
    
    // free(r2);
    // free(sizes_r2);
    // free(out2);
    
    
    return hh;
    
}

void sb_coarsegrain(const double y[], const int size, const char how[], const int num_groups, int labels[])
{
    int i, j;
    if (simple_strcmp(how, "quantile") == 1) {
        // fprintf(stdout, "ERROR in sb_coarsegrain: unknown coarse-graining method\n");
        exit(1);
    }
    
    /*
    for(int i = 0; i < size; i++){
        printf("yin coarsegrain[%i]=%1.4f\n", i, y[i]);
    }
    */
    
    double th[8];
    double ls[8];
    linspace(0, 1, num_groups + 1, ls);
    for (i = 0; i < num_groups + 1; i++) {
        //double quant = quantile(y, size, ls[i]);
        th[i] = quantile(y, size, ls[i]);
    }
    th[0] -= 1;
    for (i = 0; i < num_groups; i++) {
        for (j = 0; j < size; j++) {
            if (y[j] > th[i] && y[j] <= th[i + 1]) {
                labels[j] = i + 1;
            }
        }
    }
}

double f_entropy(const double a[], const int size)
{
    double f = 0.0;
    for (int i = 0; i < size; i++) {
        if (a[i] > 0) {
            f += a[i] * log(a[i]);
        }
    }
    return -1 * f;
}

void subset(const int a[], int b[], const int start, const int end)
{
    int j = 0;
    for (int i = start; i < end; i++) {
        b[j++] = a[i];
    }
    return;
}

void linspace(double start, double end, int num_groups, double out[])
{
    double step_size = (end - start) / (num_groups - 1);
    for (int i = 0; i < num_groups; i++) {
        out[i] = start;
        start += step_size;
    }
    return;
}

double quantile(const double y[], const int size, const double quant)
{   
    double quant_idx, q, value;
    int idx_left, idx_right;
    double tmp[DATA_SIZE];
    for (int l = 0; l < DATA_SIZE; l++) {
        tmp[l] = y[l];
    }
    sort(tmp, size);
    
    /*
    for(int i=0; i < size; i++){
        printf("y[%i]=%1.4f\n", i, y[i]);
    }
    for(int i=0; i < size; i++){
        printf("sorted[%i]=%1.4f\n", i, tmp[i]);
    }
     */
    
    // out of range limit?
    q = 0.5 / size;
    if (quant < q) {
        value = tmp[0]; // min value
        return value; 
    } else if (quant > (1 - q)) {
        value = tmp[size - 1]; // max value
        return value; 
    }
    
    quant_idx = size * quant - 0.5;
    idx_left = (int)floor(quant_idx);
    idx_right = (int)ceil(quant_idx);
    value = tmp[idx_left] + (quant_idx - idx_left) * (tmp[idx_right] - tmp[idx_left]) / (idx_right - idx_left);
    return value;
}

void sort(double y[], int size) {
    int i, j;
    for (i = 0; i < size - 1; i++) {
        for (j = 0; j < size - i - 1; j++) {
            #pragma HLS LOOP_FLATTEN off
            #pragma HLS PIPELINE
            if (y[j] > y[j + 1]) {
                // Swap elements
                double temp = y[j];
                y[j] = y[j + 1];
                y[j + 1] = temp;
            }
        }
    }
}

int simple_strcmp(const char *str1, const char *str2) {
    while (*str1 && (*str1 == *str2)) {
        str1++;
        str2++;
    }
    return *(const unsigned char*)str1 - *(const unsigned char*)str2;
}


double CO_HistogramAMI_even_2_5(const double y[], const int size)
{
    
    // // NaN check
    // for(int i = 0; i < size; i++)
    // {
    //     if(isnan(y[i]))
    //     {
    //         return NAN;
    //     }
    // }
    
    const int tau = 2;
    const int numBins = 5;
    
    double y1[DATA_SIZE];
    double y2[DATA_SIZE];
    
    for(int i = 0; i < size-tau; i++){
        y1[i] = y[i];
        y2[i] = y[i+tau];
    }
    
    // set bin edges
    const double maxValue = max_(y, size);
    const double minValue = min_(y, size);
    
    double binStep = (maxValue - minValue + 0.2)/5;
    //double binEdges[numBins+1] = {0};
	double binEdges[5+1] = {0};
    for(int i = 0; i < numBins+1; i++){
        binEdges[i] = minValue + binStep*i - 0.1;
        // printf("binEdges[%i] = %1.3f\n", i, binEdges[i]);
    }
    
    
    // count histogram bin contents
    int bins1[DATA_SIZE];
    histbinassign(y1, size-tau, binEdges, numBins+1, bins1);
    
    int bins2[DATA_SIZE];
    histbinassign(y2, size-tau, binEdges, numBins+1, bins2);
    
    /*
    // debug
    for(int i = 0; i < size-tau; i++){
        printf("bins1[%i] = %i, bins2[%i] = %i\n", i, bins1[i], i, bins2[i]);
    }
    */
    
    // joint
    double bins12[DATA_SIZE];
    //double binEdges12[(numBins + 1) * (numBins + 1)] = {0};
	double binEdges12[(5 + 1) * (5 + 1)] = {0};    

    for(int i = 0; i < size-tau; i++){
        bins12[i] = (bins1[i]-1)*(numBins+1) + bins2[i];
        // printf("bins12[%i] = %1.3f\n", i, bins12[i]);
    }
    
    for(int i = 0; i < (numBins+1)*(numBins+1); i++){
        binEdges12[i] = i+1;
        // printf("binEdges12[%i] = %1.3f\n", i, binEdges12[i]);
    }
    
    // fancy solution for joint histogram here
    int jointHistLinear[DATA_SIZE];
    histcount_edges(bins12, size-tau, binEdges12, (numBins + 1) * (numBins + 1), jointHistLinear);
    
    /*
    // debug
    for(int i = 0; i < (numBins+1)*(numBins+1); i++){
        printf("jointHistLinear[%i] = %i\n", i, jointHistLinear[i]);
    }
    */
    
    // transfer to 2D histogram (no last bin, as in original implementation)
    double pij[numBins][numBins];
    int sumBins = 0;
    for(int i = 0; i < numBins; i++){
        for(int j = 0; j < numBins; j++){
            pij[j][i] = jointHistLinear[i*(numBins+1)+j];
            
            // printf("pij[%i][%i]=%1.3f\n", i, j, pij[i][j]);
            
            sumBins += pij[j][i];
        }
    }
    
    // normalise
    for(int i = 0; i < numBins; i++){
        for(int j = 0; j < numBins; j++){
            pij[j][i] /= sumBins;
        }
    }

    // marginals
    //double pi[numBins] = {0};
	double pi[5] = {0};
    //double pj[numBins] = {0};
	double pj[5] = {0};
    for(int i = 0; i < numBins; i++){
        for(int j = 0; j < numBins; j++){
            pi[i] += pij[i][j];
            pj[j] += pij[i][j];
            // printf("pij[%i][%i]=%1.3f, pi[%i]=%1.3f, pj[%i]=%1.3f\n", i, j, pij[i][j], i, pi[i], j, pj[j]);
        }
    }
    
    /*
    // debug
    for(int i = 0; i < numBins; i++){
        printf("pi[%i]=%1.3f, pj[%i]=%1.3f\n", i, pi[i], i, pj[i]);
    }
    */
    
    // mutual information
    double ami = 0;
    for(int i = 0; i < numBins; i++){
        for(int j = 0; j < numBins; j++){
            if(pij[i][j] > 0){
                //printf("pij[%i][%i]=%1.3f, pi[%i]=%1.3f, pj[%i]=%1.3f, logarg=, %1.3f, log(...)=%1.3f\n",
                //       i, j, pij[i][j], i, pi[i], j, pj[j], pij[i][j]/(pi[i]*pj[j]), log(pij[i][j]/(pi[i]*pj[j])));
                ami += pij[i][j] * log(pij[i][j]/(pj[j]*pi[i]));
            }
        }
    }
    
    return ami;
}


void histbinassign(const double y[], const int size, const double binEdges[], const int nEdges, int output[])
{
    
    
    // variable to store counted occurances in
    for(int i = 0; i < size; i++)
    {
        // if not in any bin -> 0
        output[i] = 0;
        
        // go through bin edges
        for(int j = 0; j < nEdges; j++){
            if(y[i] < binEdges[j]){
                output[i] = j;
                break;
            }
        }
    }
    
}

void histcount_edges(const double y[], const int size, const double binEdges[], const int nEdges, int output[])
{
    
    
    for(int i = 0; i < nEdges; i++){
        output[i] = 0;
    }
    
    for(int i = 0; i < size; i++)
    {
        // go through bin edges
        for(int j = 0; j < nEdges; j++){
            if(y[i] <= binEdges[j]){
                output[j] += 1;
                break;
            }
        }
    }  
}


double CO_Embed2_Dist_tau_d_expfit_meandiff(double y[], const int size)
{
    
    // // NaN check
    // for(int i = 0; i < size; i++)
    // {
    //     if(isnan(y[i]))
    //     {
    //         return NAN;
    //     }
    // }
    
    int tau = co_firstzero(y, size, size);
    
    //printf("co_firstzero ran\n");
    
    if (tau > (double)size/10){
        tau = floor((double)size/10);
    }
    //printf("tau = %i\n", tau);
    
    double d[DATA_SIZE];
    for(int i = 0; i < size-tau-1; i++)
    {
        
        d[i] = sqrt((y[i+1]-y[i])*(y[i+1]-y[i]) + (y[i+tau]-y[i+tau+1])*(y[i+tau]-y[i+tau+1]));
        
        // //printf("d[%i]: %1.3f\n", i, d[i]);
        // if (isnan(d[i])){
        //     free(d);
        //     return NAN;
        // }
        
        /*
        if(i<100)
            printf("%i, y[i]=%1.3f, y[i+1]=%1.3f, y[i+tau]=%1.3f, y[i+tau+1]=%1.3f, d[i]: %1.3f\n", i, y[i], y[i+1], y[i+tau], y[i+tau+1], d[i]);
         */
    }
    
    //printf("embedding finished\n");
    
    // mean for exponential fit
    double l = mean(d, size-tau-1);
    
    // count histogram bin contents
    /*
     int * histCounts;
    double * binEdges;
    int nBins = histcounts(d, size-tau-1, -1, &histCounts, &binEdges);
     */
    
    int nBins = num_bins_auto(d, size-tau-1);
    if (nBins == 0){
        return 0;
    }
    int histCounts[DATA_SIZE];
    double binEdges[DATA_SIZE];
    histcounts_preallocated(d, size-tau-1, nBins, histCounts, binEdges);
    
    //printf("histcount ran\n");
    
    // normalise to probability
    double histCountsNorm[DATA_SIZE];
    for(int i = 0; i < nBins; i++){
        //printf("histCounts %i: %i\n", i, histCounts[i]);
        histCountsNorm[i] = (double)histCounts[i]/(double)(size-tau-1);
        //printf("histCounts norm %i: %1.3f\n", i, histCountsNorm[i]);
    }
    
    /*
    for(int i = 0; i < nBins; i++){
        printf("histCounts[%i] = %i\n", i, histCounts[i]);
    }
    for(int i = 0; i < nBins; i++){
        printf("histCountsNorm[%i] = %1.3f\n", i, histCountsNorm[i]);
    }
    for(int i = 0; i < nBins+1; i++){
        printf("binEdges[%i] = %1.3f\n", i, binEdges[i]);
    }
    */
     
    
    //printf("histcounts normed\n");
    
    double d_expfit_diff[DATA_SIZE];
    for(int i = 0; i < nBins; i++){
        double expf = exp(-(binEdges[i] + binEdges[i+1])*0.5/l)/l;
        if (expf < 0){
            expf = 0;
        }
        d_expfit_diff[i] = fabs(histCountsNorm[i]-expf);
        //printf("d_expfit_diff %i: %1.3f\n", i, d_expfit_diff[i]);
    }
    
    double out = mean(d_expfit_diff, nBins);
    
    //printf("out = %1.6f\n", out);
    //printf("reached free statements\n");
    
    
    return out;
    
}

int num_bins_auto(const double y[], const int size){
    
    double maxVal = max_(y, size);
    double minVal = min_(y, size);
    
    if (stddev(y, size) < 0.001){
        return 0;
    }
    
    return ceil((maxVal-minVal)/(3.5*stddev(y, size)/pow(size, 1/3.)));
    
}

double stddev(const double a[], const int size)
{
    double m = mean(a, size);
    double sd = 0.0;
    for (int i = 0; i < size; i++) {
        sd += pow(a[i] - m, 2);
    }
    sd = sqrt(sd / (size - 1));
    return sd;
}

double max_(const double a[], const int size)
{
    double m = a[0];
    for (int i = 1; i < size; i++) {
        if (a[i] > m) {
            m = a[i];
        }
    }
    return m;
}

double min_(const double a[], const int size)
{
    double m = a[0];
    for (int i = 1; i < size; i++) {
        if (a[i] < m) {
            m = a[i];
        }
    }
    return m;
}


int histcounts_preallocated(const double y[], const int size, int nBins, int binCounts[], double binEdges[])
{
    
    int i = 0;
    
    // check min and max of input array
    double minVal = DBL_MAX, maxVal=-DBL_MAX;
    for(int i = 0; i < size; i++)
    {
        // printf("histcountInput %i: %1.3f\n", i, y[i]);
        
        if (y[i] < minVal)
        {
            minVal = y[i];
        }
        if (y[i] > maxVal)
        {
            maxVal = y[i];
        }
    }
    
    // and derive bin width from it
    double binStep = (maxVal - minVal)/nBins;
    
    // variable to store counted occurances in
    for(i = 0; i < nBins; i++)
    {
        binCounts[i] = 0;
    }
    
    for(i = 0; i < size; i++)
    {
        
        int binInd = (y[i]-minVal)/binStep;
        if(binInd < 0)
            binInd = 0;
        if(binInd >= nBins)
            binInd = nBins-1;
        //printf("histcounts, i=%i, binInd=%i, nBins=%i\n", i, binInd, nBins);
        binCounts[binInd] += 1;
        
    }
    
    for(i = 0; i < nBins+1; i++)
    {
        binEdges[i] = i * binStep + minVal;
    }
    
    /*
     // debug
     for(i=0;i<nBins;i++)
     {
     printf("%i: count %i, edge %1.3f\n", i, binCounts[i], binEdges[i]);
     }
     */
    
    return 0;
    
}


double mean(const double a[], const int size)
{
    double m = 0.0;
    for (int i = 0; i < size; i++) {
        m += a[i];
    }
    m /= size;
    return m;
}







// Rounds a value to n decimal places according to the precision
double round(double var, int precision) {
    if (precision < 0) precision = 0;
    double scale = pow(10, precision);
    // Proper rounding: add 0.5 for positive numbers, subtract 0.5 for negative numbers
    double value = (var >= 0) ? floor(var * scale + 0.5) : ceil(var * scale - 0.5);
    return value / scale;
}


double CO_f1ecac(double window[DATA_SIZE])
{
    
    double autocorrs[2 * DATA_SIZE];
    co_autocorrs(window, autocorrs);
    double k[DATA_SIZE];
    for(int i=0; i<DATA_SIZE;i++){
        k[i] = autocorrs[i]/autocorrs[0];
    }
    std::cout << "L jgjg" << k[0] << std::endl;
    std::cout << "L jgjg" << k[1] << std::endl;
    std::cout << "L jgjg" << k[2] << std::endl;
    std::cout << "L jgjg" << k[3] << std::endl;
    // threshold to cross
    double thresh = 0.36;
    double out = (double)DATA_SIZE;
    std::cout << "KERNEL jgjg" << out << std::endl;
    for(int i = 0; i < DATA_SIZE-2; i++){
        // printf("i=%d autocorrs_i=%1.3f\n", i, autocorrs[i]);
        if ( round(k[i+1],2) < thresh ){
            double m = round(k[i+1],2) - round(k[i],2);
            if (m == 0) continue; 
            double dy = thresh - round(k[i],2);
            double dx = dy/m;
            out = ((double)i) + round(dx,2);
            // printf("thresh=%1.3f AC(i)=%1.3f AC(i-1)=%1.3f m=%1.3f dy=%1.3f dx=%1.3f out=%1.3f\n", thresh, autocorrs[i], autocorrs[i-1], m, dy, dx, out);
            return out;
        }
    }
    
    
    return out;
    
}

double FC_LocalSimple_mean1_tauresrat(double y[], const int size){
    return FC_LocalSimple_mean_tauresrat(y, size, 1);
}
int co_firstzero( double y[], int size, int maxtau) {
    double autocorrs[2*DATA_SIZE];
    
    co_autocorrs(y, autocorrs);
    int zerocrossind = 0;
    while(autocorrs[zerocrossind] > 0 && zerocrossind < maxtau)
    {
        zerocrossind += 1;
    }
    std::cout << "kerneladv"<<zerocrossind;
    return zerocrossind;
}

double FC_LocalSimple_mean_tauresrat(double y[], int size, int train_length) {
    double static_res[DATA_SIZE];
    for (int i = 0; i < size - train_length; ++i) {
        double yest = 0;
        for (int j = 0; j < train_length; ++j) {
            yest += y[i + j];
        }
        yest /= train_length;
        static_res[i] = y[i + train_length] - yest;
    }

    double resAC1stZ = co_firstzero(static_res, size - train_length, size - train_length);
    double yAC1stZ = co_firstzero(y, size, size);
    return yAC1stZ != 0 ? resAC1stZ / yAC1stZ : 0; // Avoid division by zero
}


double mean1(double a[])
{
    double m = 0.0;
    for (int i = 0; i < DATA_SIZE; i++) {
        m += a[i];
    }
    m /= DATA_SIZE;
    return m;
}

void dot_multiply(cmplx_type a[DATA_SIZE], cmplx_type b[DATA_SIZE])
{
    for (int i = 0; i < DATA_SIZE; i++) {
        cmplx_type a_conj; 
        CMPXCONJ(a_conj, a[i]);
        CMPXMUL(b[i], a[i], a_conj);
    }
}

cmplx_type cmpxdiv(cmplx_type a, cmplx_type b) {
    cmplx_type a_conj, a_conj_divisor, result;
    CMPXCONJ(a_conj, a);
    CMPXMUL(a_conj_divisor, a, a_conj);
    result.real = b.real/a_conj_divisor.real; 
    result.imag = b.imag/a_conj_divisor.real; 
    return result;
}

void co_autocorrs(double y[DATA_SIZE], double z[DATA_SIZE])
{   
    std::cout << "AUTO COR" << std::endl; 
    double m, nFFT;
    m = mean1(y);
    
    cmplx_type input[2 * DATA_SIZE];
    cmplx_type output[2 * DATA_SIZE];
    
    for (int i = 0; i < DATA_SIZE; i++) {
        input[i].real = y[i] - m;
        input[i].imag = 0.0; 
    }
    for (int i = DATA_SIZE; i < 2 * DATA_SIZE; i++) {
        input[i].real = 0.0;
        input[i].imag = 0.0; 
    }

    pease_fft(input, output);
    dot_multiply(output, input);
    pease_fft(input, output);

    cmplx_type divisor = output[0];
    
    for (int i = 0; i < 2 * DATA_SIZE; i++) {
        input[i] = cmpxdiv(divisor, output[i]); // F[i] / divisor;
    }


    for (int i = 0; i < 2 * DATA_SIZE; i++) {
  
        z[i] = input[i].real;
    }

}


int CO_FirstMin_ac(double window[DATA_SIZE])
{
    // Removed NaN check
    double autocorrs[2 * DATA_SIZE];
    co_autocorrs(window, autocorrs);
    std::cout << "KERNEL jgjg" << autocorrs[0] << std::endl;
    std::cout << "KERNEL jgjg" << autocorrs[1] << std::endl;
    std::cout << "KERNEL jgjg" << autocorrs[2] << std::endl;
    std::cout << "KERNEL jgjg" << autocorrs[3] << std::endl;
    int minInd = DATA_SIZE;
    for(int i = 1; i < DATA_SIZE-1; i++)
    {
        if(autocorrs[i] < autocorrs[i-1] && autocorrs[i] < autocorrs[i+1])
        {
            minInd = i;
            break;
        }
    }    
    return minInd;
    
}

void generate(hls::stream<data_t> &input, data_t y[DATA_SIZE]) {

    for (int i = 0; i < DATA_SIZE-1; i++) {
        y[i] = y[i+1];
    }
    if (!input.empty()) {
        data_t val = input.read();
        y[DATA_SIZE-1] = val;
    }
}

void replicate_stream(hls::stream<data_t> &input, hls::stream<data_t> &input1, hls::stream<data_t> &input2, hls::stream<data_t> &input3) {
    while (!input.empty()) {
        data_t val = input.read();
        input1.write(val);
        input2.write(val);
        input3.write(val);
    }
}



void computation(
        hls::stream<data_t> &input,  hls::stream<data_t> &outp1, hls::stream<data_t> &outp2, hls::stream<data_t> &outp3
        // data_t window[DATA_SIZE], data_t *outp1
        // data_t *outp2, data_t *outp3, data_t *outp4, data_t *outp5, data_t *outp6, data_t *outp7, data_t *outp8, data_t *outp9, data_t *outp10, data_t *outp11, data_t *outp12, data_t *outp13, data_t *outp14, data_t *outp15, data_t *outp16, data_t *outp17, data_t *outp18, data_t *outp19, data_t *outp20, data_t *outp21, data_t *outp22
    ) {

    hls::stream<data_t> input1;
    hls::stream<data_t> input2;
    hls::stream<data_t> input3;

    replicate_stream(input, input1, input2, input3);
    #pragma DATAFLOW
    MD_hrv_classic_pnn40(input1,DATA_SIZE,outp1);
    DN_OutlierInclude_p_001_mdrmd(input2,DATA_SIZE,outp2);
    IN_AutoMutualInfoStats_40_gaussian_fmmi(input3,DATA_SIZE,outp3);
    // *outp4 = CO_FirstMin_ac(window);
    // *outp5 = FC_LocalSimple_mean1_tauresrat(window,DATA_SIZE);
    // *outp6 = CO_f1ecac(window);
    // *outp7 = CO_Embed2_Dist_tau_d_expfit_meandiff(window,DATA_SIZE);
    // *outp8 = CO_HistogramAMI_even_2_5(window,DATA_SIZE);
    // *outp9 = SB_MotifThree_quantile_hh(window,DATA_SIZE);
    // *outp10 = SB_TransitionMatrix_3ac_sumdiagcov(window,DATA_SIZE);
    // *outp11 = DN_HistogramMode_5(window,DATA_SIZE);
    // *outp12 = DN_HistogramMode_10(window,DATA_SIZE);
    // *outp13 = SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1(window,DATA_SIZE);
    // *outp14 = SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1(window,DATA_SIZE);
    // *outp15 = PD_PeriodicityWang_th0_01(window,DATA_SIZE);
    // *outp16 = FC_LocalSimple_mean3_stderr(window,DATA_SIZE);
    // *outp17 = SP_Summaries_welch_rect_area_5_1(window,DATA_SIZE);
    // *outp18 = SP_Summaries_welch_rect_centroid(window,DATA_SIZE);
    // *outp19 = DN_OutlierInclude_n_001_mdrmd(window,DATA_SIZE);
    // *outp20 = SB_BinaryStats_diff_longstretch0(window,DATA_SIZE);
    // *outp21 = SB_BinaryStats_mean_longstretch1(window,DATA_SIZE);
    // *outp22 = CO_trev_1_num(window,DATA_SIZE);
}



extern "C" void krnl(hls::stream<data_t> &input, hls::stream<data_t> &outp1, hls::stream<data_t> &outp2, hls::stream<data_t> &outp3) {

    #pragma HLS INTERFACE mode=axis port=input
    #pragma HLS INTERFACE mode=axis port=outp1
    #pragma HLS INTERFACE mode=axis port=outp2
    #pragma HLS INTERFACE mode=axis port=outp3
    // static data_t window[DATA_SIZE];
    // data_t outp1;
    // data_t outp2;
    // data_t outp3;
    // data_t outp4;
    // data_t outp5;
    // data_t outp6;
    // data_t outp7;
    // data_t outp8;
    // data_t outp9;
    // data_t outp10;
    // data_t outp11;
    // data_t outp12;
    // data_t outp13;
    // data_t outp14;
    // data_t outp15;
    // data_t outp16;
    // data_t outp17;
    // data_t outp18;
    // data_t outp19;
    // data_t outp20;
    // data_t outp21;
    // data_t outp22;
    // static int w = 0; 

    // /* Reading from DDR*/
    // for (int i = 0; i < DATA_SIZE; i++) {
    //     // #pragma HLS UNROLL
    //     window[i] = input[i];
    // }

    computation(input, outp1, outp2, outp3);


    
    /* Feature Extraction */
    // result = CO_FirstMin_ac(window);
    // result = FC_LocalSimple_mean1_tauresrat(window,DATA_SIZE);
    // result = CO_f1ecac(window);
    // result = CO_Embed2_Dist_tau_d_expfit_meandiff(window,DATA_SIZE);
    // result = CO_HistogramAMI_even_2_5(window,DATA_SIZE);
    // result = SB_MotifThree_quantile_hh(window,DATA_SIZE);
    // result = SB_TransitionMatrix_3ac_sumdiagcov(window,DATA_SIZE);
    // result = DN_HistogramMode_5(window,DATA_SIZE);
    // result = DN_HistogramMode_10(window,DATA_SIZE);
    // result =SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1(window,DATA_SIZE);
    // result =SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1(window,DATA_SIZE);
    // result =PD_PeriodicityWang_th0_01(window,DATA_SIZE);
    // result =FC_LocalSimple_mean3_stderr(window,DATA_SIZE);
    // result =SP_Summaries_welch_rect_area_5_1(window,DATA_SIZE);
    // result =SP_Summaries_welch_rect_centroid(window,DATA_SIZE);
    // result =DN_OutlierInclude_n_001_mdrmd(window,DATA_SIZE);
    // result =DN_OutlierInclude_p_001_mdrmd(window,DATA_SIZE);
    // result =SB_BinaryStats_diff_longstretch0(window,DATA_SIZE);
    // result = SB_BinaryStats_mean_longstretch1(window,DATA_SIZE);
    // result = IN_AutoMutualInfoStats_40_gaussian_fmmi(window,DATA_SIZE);
    // result = CO_trev_1_num(window,DATA_SIZE);
    // result = MD_hrv_classic_pnn40(window,DATA_SIZE);
    /* Writing to DDR */
    // output[0] = result;

    // output[0] = outp1;
    // output[1] = outp2;
    // output[2] = outp3;
    // output[3] = outp4;
    // output[4] = outp5;
    // output[5] = outp6;
    // output[6] = outp7;
    // output[7] = outp8;
    // output[8] = outp9;
    // output[9] = outp10;
    // output[10] = outp11;
    // output[11] = outp12;
    // output[12] = outp13;
    // output[13] = outp14;
    // output[14] = outp15;
    // output[15] = outp16;
    // output[16] = outp17;
    // output[17] = outp18;
    // output[18] = outp19;
    // output[19] = outp20;
    // output[20] = outp21;
    // output[21] = outp22;
}
