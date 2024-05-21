#include "host.h"
#include <stdio.h>
#include "constants.h"
#include <vector>
#include <cmath>
#include <cstdlib>
#include <float.h>
#include <tgmath.h>
#include <complex>
#include <iostream>
#include <math.h>
#include <string.h>
#include <time.h>
#include <complex.h>
#include <stdlib.h>
#include <algorithm>
#include <cfloat>
#include <cstring>
#include <ctime>
#include <complex>
#define pow2(x) (1 << x)
#define nCoeffs 3
#define nPoints 4
#define pieces 2
#define nBreaks 3
#define deg 3
#define nSpline 4
#define piecesExt 8 //3 * deg - 1


#define pow2(x) (1 << x)
typedef std::complex< double > cplx;
double FC_LocalSimple_mean_tauresrat(const double y[], const int size, const int train_length);
int co_firstzero(const double y[], const int size, const int maxtau);
double * co_autocorrs(const double y[], const int size);
int nextpow2(int n);
void twiddles(cplx a[], int size);
void fft(cplx a[], int size, cplx tw[]);
static void _fft(cplx a[], cplx out[], int size, int step, cplx tw[]);
void dot_multiply(cplx a[], cplx b[], int size);
double mean(const double a[], const int size);
double round(double var, int precision);
int num_bins_auto(const double y[], const int size);
double stddev(const double a[], const int size);
int histcounts_preallocated(const double y[], const int size, int nBins, int * binCounts, double * binEdges);
double max_(const double a[], const int size);
double min_(const double a[], const int size);
int * histbinassign(const double y[], const int size, const double binEdges[], const int nEdges);
int * histcount_edges(const double y[], const int size, const double binEdges[], const int nEdges);
void sb_coarsegrain(const double y[], const int size, const char how[], const int num_groups, int labels[]);
double f_entropy(const double a[], const int size);
void subset(const int a[], int b[], const int start, const int end);
void linspace(double start, double end, int num_groups, double out[]);
double quantile(const double y[], const int size, const double quant);
void sort(double y[], int size);
static int compare (const void * a, const void * b);
double cov(const double x[], const double y[], const int size);
int histcounts(const double y[], const int size, int nBins, int ** binCounts, double ** binEdges);
int linreg(const int n, const double x[], const double y[], double* m, double* b);
double norm_(const double a[], const int size);
double SC_FluctAnal_2_50_1_logi_prop_r1(const double y[], const int size, const int lag, const char how[]);
int iLimit(int x, int lim);
void icumsum(const int a[], const int size, int b[]);
void lsqsolve_sub(const int sizeA1, const int sizeA2, const double *A, const int sizeb, const double *b, double *x);
void matrix_multiply(const int sizeA1, const int sizeA2, const double *A, const int sizeB1, const int sizeB2, const double *B, double *C);
void matrix_times_vector(const int sizeA1, const int sizeA2, const double *A, const int sizeb, const double *b, double *c);
void gauss_elimination(int size, double *A, double *b, double *x);
int splinefit(const double *y, const int size, double *yOut);
double cov_mean(const double x[], const double y[], const int size);
double autocov_lag(const double x[], const int size, const int lag);
double FC_LocalSimple_mean_stderr(const double y[], const int size, const int train_length);
void cumsum(const double a[], const int size, double b[]);
int welch(const double y[], const int size, const int NFFT, const double Fs, const double window[], const int windowWidth, double ** Pxx, double ** f);
double SP_Summaries_welch_rect(const double y[], const int size, const char what[]);
double median(const double a[], const int size);
double DN_OutlierInclude_np_001_mdrmd(const double y[], const int size, const int sign);
double corr(const double x[], const double y[], const int size);
double autocorr_lag(const double x[], const int size, const int lag);
void diff(const double a[], const int size, double b[]);

double MD_hrv_classic_pnn40(const double y[], const int size){
    
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
    double * Dy = (double*)malloc((size-1) * sizeof(double));
    diff(y, size, Dy);
    
    double pnn40 = 0;
    for(int i = 0; i < size-1; i++){
        if(fabs(Dy[i])*1000 > pNNx){
            pnn40 += 1;
        }
    }
    
    free(Dy);
    
    return pnn40/(size-1);
}

void diff(const double a[], const int size, double b[])
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
    
    double * diffTemp = (double*)malloc((size-1) * sizeof * diffTemp);
    
    for(int i = 0; i < size-tau; i++)
    {
        diffTemp[i] = pow(y[i+1] - y[i],3);
    }
    
    double out;
    
    out = mean(diffTemp, size-tau);
    
    free(diffTemp);
    
    return out;
}

double IN_AutoMutualInfoStats_40_gaussian_fmmi(const double y[], const int size)
{
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
    double * ami = (double*)malloc(size * sizeof(double));
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
    
    free(ami);
    
    return fmmi;
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

double autocorr_lag(const double x[], const int size, const int lag){
    
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
    int * yBin = (int*)malloc((size-1) * sizeof(int));
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
    
    free(yBin);
    
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
    int * yBin = (int*)malloc((size-1) * sizeof(int));
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
    
    free(yBin);
    
    return maxstretch1;
}

double DN_OutlierInclude_p_001_mdrmd(const double y[], const int size)
{
    return DN_OutlierInclude_np_001_mdrmd(y, size, 1.0);
}

double DN_OutlierInclude_n_001_mdrmd(const double y[], const int size)
{
    return DN_OutlierInclude_np_001_mdrmd(y, size, -1.0);
}

double DN_OutlierInclude_np_001_mdrmd(const double y[], const int size, const int sign)
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
    double * yWork = (double*)malloc(size * sizeof(double));
    
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
        free(yWork);
        return 0; // if constant, return 0
    }
    
    // find maximum (or minimum, depending on sign)
    double maxVal = max_(yWork, size);
    
    // maximum value too small? return 0
    if(maxVal < inc){
        free(yWork);
        return 0;
    }
    
    int nThresh = maxVal/inc + 1;
    
    // save the indices where y > threshold
    double * r = (double*)malloc(size * sizeof * r);
    
    // save the median over indices with absolute value > threshold
    double * msDti1 = (double*)malloc(nThresh * sizeof(double));
    double * msDti3 = (double*)malloc(nThresh * sizeof(double));
    double * msDti4 = (double*)malloc(nThresh * sizeof(double));
    
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
        double * Dt_exc = (double*)malloc(highSize * sizeof(double));
        
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
        
        free(Dt_exc);
        
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
    
    free(r);
    free(yWork);
    free(msDti1);
    free(msDti3);
    free(msDti4);
    
    return outputScalar;
}

double median(const double a[], const int size)
{
    double m;
    double * b = (double*)malloc(size * sizeof *b);
    memcpy(b, a, size * sizeof *b);
    sort(b, size);
    if (size % 2 == 1) {
        m = b[size / 2];
    } else {
        int m1 = size / 2;
        int m2 = m1 - 1;
        m = (b[m1] + b[m2]) / (double)2.0;
    }
    free(b);
    return m;
}



double SP_Summaries_welch_rect_area_5_1(const double y[], const int size)
{
    return SP_Summaries_welch_rect(y, size, "area_5_1");
}
double SP_Summaries_welch_rect_centroid(const double y[], const int size)
{
    return SP_Summaries_welch_rect(y, size, "centroid");
    
}

int welch(const double y[], const int size, const int NFFT, const double Fs, const double window[], const int windowWidth, double ** Pxx, double ** f){
    
    double dt = 1.0/Fs;
    double df = 1.0/(nextpow2(windowWidth))/dt;
    double m = mean(y, size);
    
    // number of windows, should be 1
    int k = floor((double)size/((double)windowWidth/2.0))-1;
    
    // normalising scale factor
    double KMU = k * pow(norm_(window, windowWidth),2);
    
    double * P =(double*) malloc(NFFT * sizeof(double));
    for(int i = 0; i < NFFT; i++){
        P[i] = 0;
    }
    
    // fft variables
    cplx * F = (cplx*)malloc(NFFT * sizeof *F);
    cplx * tw = (cplx*)malloc(NFFT * sizeof *tw);
    twiddles(tw, NFFT);
    
    double * xw = (double*)malloc(windowWidth * sizeof(double));
    for(int i = 0; i<k; i++){
        
        // apply window
        for(int j = 0; j<windowWidth; j++){
            xw[j] = window[j]*y[j + (int)(i*(double)windowWidth/2.0)];
        }
        
        // initialise F (
        for (int i = 0; i < windowWidth; i++) {
            
	    // #if defined(__GNUC__) || defined(__GNUG__)
		// cplx tmp = xw[i] - m + 0.0 * I;
	    // #elif defined(_MSC_VER)
		    cplx tmp = { xw[i] - m, 0.0 };
	    // #endif
            
            
            F[i] = tmp; // CMPLX(xw[i] - m, 0.0);
        }
        for (int i = windowWidth; i < NFFT; i++) {
            // F[i] = CMPLX(0.0, 0.0);
            //cplx tmp = { 0.0, 0.0 };
        //     #if defined(__GNUC__) || defined(__GNUG__)
		// cplx tmp = 0.0 + 0.0 * I;
	    // #elif defined(_MSC_VER)
		    cplx tmp = { 0.0 , 0.0 };
	    // #endif
            F[i] = tmp;
        }
        
        fft(F, NFFT, tw);
        // std::cout << "hostF0" << F[0];
        // std::cout << "hostF1" << F[1];
        
        /*
        for(int i = 0; i < NFFT; i++){
            printf("F1[%i] real: %1.3f, imag: %1.3f\n", i, creal(F[i]), cimag(F[i]));
        }
         */
        
        for(int l = 0; l < NFFT; l++){
            P[l] += round(pow(abs(F[l]),2),5);
        }
        std::cout << "hostp0" << P[0];
        std::cout << "hostp1" << P[1];
        /*
        /*
        for(int i = 0; i < NFFT; i++){
            printf("P[%i]: %1.3f\n", i, P[i]);
        }
         */
        
    }
    
    int Nout = (NFFT/2+1);
    *Pxx = (double*)malloc(Nout * sizeof(double));
    for(int i = 0; i < Nout; i++){
        (*Pxx)[i] = round(P[i]/KMU*dt,5);
        if(i>0 & i < Nout-1){
            (*Pxx)[i] *= 2;
        }
    }
    /*
    for(int i = 0; i < Nout; i++){
        printf("Pxx[%i]: %1.3f\n", i, Pxx[i]);
    }
     */
    
    *f = (double*)malloc(Nout * sizeof(double));
    for(int i = 0; i < Nout; i++){
        (*f)[i] = round((double)i*df,5);
    }
    std::cout << "hostf0" << (*f)[0];
    std::cout << "hostf1" << (*f)[1];
    /*
    for(int i = 0; i < Nout; i++){
        printf("f[%i]: %1.3f\n", i, (*f)[i]);
    }
     */
    
    free(P);
    free(F);
    free(tw);
    free(xw);
    
    return Nout;
}

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
    double * window = (double*)malloc(size * sizeof(double));
    for(int i = 0; i < size; i++){
        window[i] = 1;
    }
    
    double Fs = 1.0; // sampling frequency
    int N = nextpow2(size);
    
    double * S;
    double * f;
    
    // compute Welch-power
    int nWelch = welch(y, size, N, Fs, window, size, &S, &f);
    free(window);
    
    // angualr frequency and spectrum on that
    double * w = (double*)malloc(nWelch * sizeof(double));
    double * Sw = (double*)malloc(nWelch * sizeof(double));
    
    double PI = 3.14;
    for(int i = 0; i < nWelch; i++){
        w[i] = round(2*PI*f[i],1);
        Sw[i] = round(S[i]/(2*PI),1);
        //printf("w[%i]=%1.3f, Sw[%i]=%1.3f\n", i, w[i], i, Sw[i]);
        if(isinf(Sw[i]) | isinf(-Sw[i])){
            return 0;
        }
    }
    
    double dw = w[1] - w[0];
    
    double * csS = (double*)malloc(nWelch * sizeof(double));
    cumsum(Sw, nWelch, csS);
    /*
    for(int i=0; i<nWelch; i++)
    {
        printf("csS[%i]=%1.3f\n", i, csS[i]);
    }
     */
    
    double output = 0;
    
    if(strcmp(what, "centroid") == 0){
        
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
    else if(strcmp(what, "area_5_1") == 0){
        double area_5_1 = 0;;
        for(int i=0; i<nWelch/5; i++){
            area_5_1 += Sw[i];
        }
        area_5_1 *= dw;
        
        output = round(area_5_1,5);
    }
    
    free(w);
    free(Sw);
    free(csS);
    free(f);
    free(S);
    
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



double FC_LocalSimple_mean3_stderr(const double y[], const int size)
{
    return FC_LocalSimple_mean_stderr(y, size, 3);
}

double FC_LocalSimple_mean_stderr(const double y[], const int size, const int train_length)
{
    // // NaN check
    // for(int i = 0; i < size; i++)
    // {
    //     if(isnan(y[i]))
    //     {
    //         return NAN;
    //     }
    // }
    
    double * res = (double*)malloc((size - train_length) * sizeof *res);
    
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
    
    free(res);
    return output;
    
}


int PD_PeriodicityWang_th0_01(const double * y, const int size){
    
    
    const double th = 0.01;
    
    double * ySpline = (double*)malloc(size * sizeof(double));
    
    // fit a spline with 3 nodes to the data
    splinefit(y, size, ySpline);
    
    //printf("spline fit complete.\n");
    
    // subtract spline from data to remove trend
    double * ySub = (double*)malloc(size * sizeof(double));
    for(int i = 0; i < size; i++){
        ySub[i] = y[i] - ySpline[i];
        //printf("ySub[%i] = %1.5f\n", i, ySub[i]);
    }
    
    // compute autocorrelations up to 1/3 of the length of the time series
    int acmax = (int)ceil((double)size/3);
    
    double * acf = (double*)malloc(acmax*sizeof(double));
    for(int tau = 1; tau <= acmax; tau++){
        // correlation/ covariance the same, don't care for scaling (cov would be more efficient)
        acf[tau-1] = autocov_lag(ySub, size, tau);
        //printf("acf[%i] = %1.9f\n", tau-1, acf[tau-1]);
    }
    
    //printf("ACF computed.\n");
    
    // find troughts and peaks
    double * troughs = (double*)malloc(acmax * sizeof(double));
    double * peaks = (double*)malloc(acmax * sizeof(double));
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
    
    free(ySpline);
    free(ySub);
    free(acf);
    free(troughs);
    free(peaks);
    
    return out;
    
}

int iLimit(int x, int lim){
    return x<lim ? x : lim;
}

void icumsum(const int a[], const int size, int b[])
{
    b[0] = a[0];
    for (int i = 1; i < size; i++) {
        b[i] = a[i] + b[i-1];
        //printf("b[%i]%1.3f = a[%i]%1.3f + b[%i-1]%1.3f\n", i, b[i], i, a[i], i, a[i-1]);
    }
    
}

void lsqsolve_sub(const int sizeA1, const int sizeA2, const double *A, const int sizeb, const double *b, double *x)
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
    
    double * AT = (double*)malloc(sizeA2 * sizeA1 * sizeof(double));
    double * ATA = (double*)malloc(sizeA2 * sizeA2 * sizeof(double));
    double * ATb = (double*)malloc(sizeA2 * sizeof(double));
    
    
    for(int i = 0; i < sizeA1; i++){
        for(int j = 0; j < sizeA2; j++){
            //AT[i,j] = A[j,i]
            AT[j * sizeA1 + i] = A[i * sizeA2 + j];
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
     
    
    matrix_multiply(sizeA2, sizeA1, AT, sizeA1, sizeA2, A, ATA);
    
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
    
    free(AT);
    free(ATA);
    free(ATb);
    
}

void matrix_multiply(const int sizeA1, const int sizeA2, const double *A, const int sizeB1, const int sizeB2, const double *B, double *C){
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
            
            //C[i][j] = 0;
            C[i*sizeB2 + j] = 0;
            for(int k = 0; k < sizeB1; k++){
                // C[i][j] += A[i][k]*B[k][j];
                C[i*sizeB2 + j] += A[i * sizeA2 + k]*B[k * sizeB2 + j];
                //printf("C[%i][%i] (k=%i) = %1.3f\n", i, j, k, C[i * sizeB2 + j]);
            }
            
        }
    }
    
}

void matrix_times_vector(const int sizeA1, const int sizeA2, const double *A, const int sizeb, const double *b, double *c){ //c[sizeb]
    
    if(sizeA2 != sizeb){
        return;
    }
    
    // row
    for(int i = 0; i < sizeA1; i++){
        
        // column
        c[i] = 0;
        for(int k = 0; k < sizeb; k++){
            c[i] += A[i * sizeA2 + k]*b[k];
        }
        
    }
    
}

void gauss_elimination(int size, double *A, double *b, double *x){
// void gauss_elimination(int size, double A[size][size], double b[size], double x[size]){
    
    double factor;
    
    // create temp matrix and vector
    // double *AElim[size];
    double* AElim[nSpline + 1];
    for (int i = 0; i < size; i++)
        AElim[i] = (double *)malloc(size * sizeof(double));
    double * bElim = (double*)malloc(size * sizeof(double));
    
    // -- create triangular matrix
    
    // initialise to A and b
    for(int i = 0; i < size; i++){
        for(int j = 0; j < size; j++){
            AElim[i][j] = A[i*size + j];
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
    
    for (int i = 0; i < size; i++)
        free(AElim[i]);
    free(bElim);
}





int splinefit(const double *y, const int size, double *yOut)
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
    int * xsB = (int*)malloc((size*nSpline)* sizeof(int));
    int * indexB = (int*)malloc((size*nSpline) * sizeof(int));
    
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
    
    double * vB = (double*)malloc((size*nSpline) * sizeof(double));
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
    
    
    double * A = (double*)malloc(size*(nSpline+1) * sizeof(double));
    
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
    
    
    
    double * x = (double*)malloc((nSpline+1)*sizeof(double));
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
     
    free(xsB);
    free(indexB);
    free(vB);
    free(A);
    free(x);
    
    return 0;
    
}

double cov_mean(const double x[], const double y[], const int size){
    
    double covariance = 0;
    
    for(int i = 0; i < size; i++){
        // double xi =x[i];
        // double yi =y[i];
        covariance += x[i] * y[i];
        
    }
    
    return covariance/size;
    
}

double autocov_lag(const double x[], const int size, const int lag){
    
    return cov_mean(x, &(x[lag]), size-lag);
    
}


double SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1(const double y[], const int size){
    return SC_FluctAnal_2_50_1_logi_prop_r1(y, size, 2, "dfa");
}

double SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1(const double y[], const int size){
    return SC_FluctAnal_2_50_1_logi_prop_r1(y, size, 1, "rsrangefit");
}



double SC_FluctAnal_2_50_1_logi_prop_r1(const double y[], const int size, const int lag, const char how[])
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
    double * yCS = (double*)malloc(sizeCS * sizeof(double));
    
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
    double * xReg = (double*)malloc(tau[nTau-1] * sizeof * xReg);
    for(int i = 0; i < tau[nTau-1]; i++)
    {
        xReg[i] = i+1;
    }
    
    // iterate over taus, cut signal, detrend and save amplitude of remaining signal
    double * F = (double*)malloc(nTau * sizeof * F);
    for(int i = 0; i < nTau; i++)
    {
        int nBuffer = sizeCS/tau[i];
        double * buffer = (double*)malloc(tau[i] * sizeof * buffer);
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
            
            if (strcmp(how, "rsrangefit") == 0) {
                F[i] += pow(max_(buffer, tau[i]) - min_(buffer, tau[i]), 2);
            }
            else if (strcmp(how, "dfa") == 0) {
                for(int k = 0; k<tau[i]; k++){
                    F[i] += buffer[k]*buffer[k];
                }
            }
            else{
                return 0.0;
            }
        }
        
        if (strcmp(how, "rsrangefit") == 0) {
            F[i] = sqrt(F[i]/nBuffer);
        }
        else if (strcmp(how, "dfa") == 0) {
            F[i] = sqrt(F[i]/(nBuffer*tau[i]));
        }
        //printf("F[%i]=%1.3f\n", i, F[i]);
        
        free(buffer);
        
    }
    
    double * logtt = (double*)malloc(nTau * sizeof * logtt);
    double * logFF = (double*)malloc(nTau * sizeof * logFF);
    int ntt = nTau;
    
    for (int i = 0; i < nTau; i++)
    {
        logtt[i] = log(tau[i]);
        logFF[i] = log(F[i]);
    }
    
    int minPoints = 6;
    int nsserr = ntt - 2*minPoints + 1;
    double * sserr = (double*)malloc(nsserr * sizeof * sserr);
    double * buffer = (double*)malloc((ntt - minPoints + 1) * sizeof * buffer);
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
    
    free(yCS); // new
    
    free(xReg);
    free(F);
    free(logtt);
    free(logFF);
    free(sserr);
    free(buffer);
    
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



double DN_HistogramMode_5(const double y[], const int size)
{
    
   
    const int nBins = 5;
    
    int * histCounts;
    double * binEdges;
    
    histcounts(y, size, nBins, &histCounts, &binEdges);
    
    /*
    for(int i = 0; i < nBins; i++){
        printf("histCounts[%i] = %i\n", i, histCounts[i]);
    }
    for(int i = 0; i < nBins+1; i++){
        printf("binEdges[%i] = %1.3f\n", i, binEdges[i]);
    }
     */
    
    double maxCount = 0;
    int numMaxs = 1;
    double out = 0;;
    for(int i = 0; i < nBins; i++)
    {
        // printf("binInd=%i, binCount=%i, binEdge=%1.3f \n", i, histCounts[i], binEdges[i]);
        
        if (histCounts[i] > maxCount)
        {
            maxCount = histCounts[i];
            numMaxs = 1;
            out = (binEdges[i] + binEdges[i+1])*0.5;
        }
        else if (histCounts[i] == maxCount){
            
            numMaxs += 1;
            out += (binEdges[i] + binEdges[i+1])*0.5;
        }
    }
    out = out/numMaxs;
    
    // arrays created dynamically in function histcounts
    free(histCounts);
    free(binEdges);
    
    return out;
}

double DN_HistogramMode_10(const double y[], const int size)
{
    
    // // NaN check
    // for(int i = 0; i < size; i++)
    // {
    //     if(isnan(y[i]))
    //     {
    //         return NAN;
    //     }
    // }
    
    const int nBins = 10;
    
    int * histCounts;
    double * binEdges;
    
    histcounts(y, size, nBins, &histCounts, &binEdges);
    
    /*
    for(int i = 0; i < nBins; i++){
        printf("histCounts[%i] = %i\n", i, histCounts[i]);
    }
    for(int i = 0; i < nBins+1; i++){
        printf("binEdges[%i] = %1.3f\n", i, binEdges[i]);
    }
     */
    
    double maxCount = 0;
    int numMaxs = 1;
    double out = 0;;
    for(int i = 0; i < nBins; i++)
    {
        // printf("binInd=%i, binCount=%i, binEdge=%1.3f \n", i, histCounts[i], binEdges[i]);
        
        if (histCounts[i] > maxCount)
        {
            maxCount = histCounts[i];
            numMaxs = 1;
            out = (binEdges[i] + binEdges[i+1])*0.5;
        }
        else if (histCounts[i] == maxCount){
            
            numMaxs += 1;
            out += (binEdges[i] + binEdges[i+1])*0.5;
        }
    }
    out = out/numMaxs;
    
    // arrays created dynamically in function histcounts
    free(histCounts);
    free(binEdges);
    
    return out;
}

int histcounts(const double y[], const int size, int nBins, int ** binCounts, double ** binEdges)
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
    
    // // if no number of bins given, choose spaces automatically
    // if (nBins <= 0){
    //     nBins = ceil((maxVal-minVal)/(3.5*stddev(y, size)/pow(size, 1/3.)));
    // }
    
    // and derive bin width from it
    double binStep = (maxVal - minVal)/nBins;
    
    // variable to store counted occurances in
    *binCounts = (int*) malloc(nBins * sizeof(int));
    for(i = 0; i < nBins; i++)
    {
        (*binCounts)[i] = 0;
    }
    
    for(i = 0; i < size; i++)
    {
        
        int binInd = (y[i]-minVal)/binStep;
        if(binInd < 0)
            binInd = 0;
        if(binInd >= nBins)
            binInd = nBins-1;
        (*binCounts)[binInd] += 1;
        
    }
    
    *binEdges = (double*)malloc((nBins+1) * sizeof(double));
    for(i = 0; i < nBins+1; i++)
    {
        (*binEdges)[i] = i * binStep + minVal;
    }
   
    /*
    // debug
    for(i=0;i<nBins;i++)
    {
        printf("%i: count %i, edge %1.3f\n", i, binCounts[i], binEdges[i]);
    }
    */
    
    return nBins;
    
}

double SB_TransitionMatrix_3ac_sumdiagcov(const double y[], const int size)
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
    
    double * yFilt = (double*)malloc(size * sizeof(double));
    
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
    double * yDown = (double*)malloc(nDown * sizeof(double));
    
    for(int i = 0; i < nDown; i++){
        yDown[i] = yFilt[i*tau];
    }
    
    /*
    for(int i = 0; i < nDown; i++){
        printf("yDown[%i]=%1.4f\n", i, yDown[i]);
    }
     */
    
    
    // transfer to alphabet
    int * yCG = (int*)malloc(nDown * sizeof(double));
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
    
    double column1[3] = {0};
    double column2[3] = {0};
    double column3[3] = {0};
    
    for(int i = 0; i < numGroups; i++){
        column1[i] = T[i][0];
        column2[i] = T[i][1];
        column3[i] = T[i][2];
        // printf("column3(%i) = %1.3f\n", i, column3[i]);
    }
    
    double *columns[3];
    columns[0] = &(column1[0]);
    columns[1] = &(column2[0]);
    columns[2] = &(column3[0]);
    
    
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
    
    free(yFilt);
    free(yDown);
    free(yCG);
    
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
    int * yt = (int*)malloc(size * sizeof(yt)); // alphabetized array
    double hh; // output
    double * out = (double*)malloc(124 * sizeof(out)); // output array
    
    // transfer to alphabet
    sb_coarsegrain(y, size, "quantile", 3, yt);
    
    // words of length 1
    array_size = alphabet_size;
    int ** r1 = (int**)malloc(array_size * sizeof(*r1));
    int * sizes_r1 = (int*)malloc(array_size * sizeof(sizes_r1));
    double * out1 = (double*)malloc(array_size * sizeof(out1));
    for (int i = 0; i < alphabet_size; i++) {
        r1[i] = (int*)malloc(size * sizeof(r1[i])); // probably can be rewritten
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
            int* tmp_ar =(int*) malloc(sizes_r1[i] * sizeof(tmp_ar));
            subset(r1[i], tmp_ar, 0, sizes_r1[i]);
            memcpy(r1[i], tmp_ar, (sizes_r1[i] - 1) * sizeof(tmp_ar));
            sizes_r1[i]--;
            free(tmp_ar);
        }
    }
    
    /*
    int *** r2 = malloc(array_size * sizeof(**r2));
    int ** sizes_r2 = malloc(array_size * sizeof(*sizes_r2));
    double ** out2 = malloc(array_size * sizeof(*out2));
    */
    int*** r2 = (int***)malloc(alphabet_size * sizeof(**r2));
    int** sizes_r2 = (int**)malloc(alphabet_size * sizeof(*sizes_r2));
    double** out2 = (double**)malloc(alphabet_size * sizeof(*out2));
    

    // allocate separately
    for (int i = 0; i < alphabet_size; i++) {
        r2[i] = (int**)malloc(alphabet_size * sizeof(*r2[i]));
        sizes_r2[i] = (int*)malloc(alphabet_size * sizeof(*sizes_r2[i]));
        //out2[i] = malloc(alphabet_size * sizeof(out2[i]));
        out2[i] = (double*)malloc(alphabet_size * sizeof(**out2));
        for (int j = 0; j < alphabet_size; j++) {
            r2[i][j] = (int*)malloc(size * sizeof(*r2[i][j]));
        }
    }

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

    free(yt);
    free(out);
    free(out1);

    free(sizes_r1);

    // free nested array
    for (int i = 0; i < alphabet_size; i++) {
        free(r1[i]);
    }
    free(r1);
    // free(sizes_r1);
    
    for (int i = 0; i < alphabet_size; i++) {
    //for (int i = alphabet_size - 1; i >= 0; i--) {

        free(sizes_r2[i]);
        free(out2[i]);
    }

    //for (int i = alphabet_size-1; i >= 0 ; i--) {
    for(int i = 0; i < alphabet_size; i++) {
        for (int j = 0; j < alphabet_size; j++) {
            free(r2[i][j]);
        }
        free(r2[i]);
    }
    
    free(r2);
    free(sizes_r2);
    free(out2);
    
    
    return hh;
    
}

void sb_coarsegrain(const double y[], const int size, const char how[], const int num_groups, int labels[])
{
    int i, j;
    if (strcmp(how, "quantile") == 1) {
        fprintf(stdout, "ERROR in sb_coarsegrain: unknown coarse-graining method\n");
        exit(1);
    }
    
    /*
    for(int i = 0; i < size; i++){
        printf("yin coarsegrain[%i]=%1.4f\n", i, y[i]);
    }
    */
    
    double * th = (double*)malloc((num_groups + 1) * 2 * sizeof(th));
    double * ls = (double*)malloc((num_groups + 1) * 2 * sizeof(th));
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
    
    free(th);
    free(ls);
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
    double * tmp = (double*)malloc(size * sizeof(*y));
    memcpy(tmp, y, size * sizeof(*y));
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
        free(tmp);
        return value; 
    } else if (quant > (1 - q)) {
        value = tmp[size - 1]; // max value
        free(tmp);
        return value; 
    }
    
    quant_idx = size * quant - 0.5;
    idx_left = (int)floor(quant_idx);
    idx_right = (int)ceil(quant_idx);
    value = tmp[idx_left] + (quant_idx - idx_left) * (tmp[idx_right] - tmp[idx_left]) / (idx_right - idx_left);
    free(tmp);
    return value;
}

void sort(double y[], int size)
{
    qsort(y, size, sizeof(*y), compare);
}

static int compare (const void * a, const void * b)
{
    if (*(double*)a < *(double*)b) {
        return -1;
    } else if (*(double*)a > *(double*)b) {
        return 1;
    } else {
        return 0;
    }
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
    
    double * y1 = (double*)malloc((size-tau) * sizeof(double));
    double * y2 = (double*)malloc((size-tau) * sizeof(double));
    
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
    int * bins1;
    bins1 = histbinassign(y1, size-tau, binEdges, numBins+1);
    
    int * bins2;
    bins2 = histbinassign(y2, size-tau, binEdges, numBins+1);
    
    /*
    // debug
    for(int i = 0; i < size-tau; i++){
        printf("bins1[%i] = %i, bins2[%i] = %i\n", i, bins1[i], i, bins2[i]);
    }
    */
    
    // joint
    double * bins12 = (double*)malloc((size-tau) * sizeof(double));
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
    int * jointHistLinear;
    jointHistLinear = histcount_edges(bins12, size-tau, binEdges12, (numBins + 1) * (numBins + 1));
    
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
    
    free(bins1);
    free(bins2);
    free(jointHistLinear);
    
    free(y1);
    free(y2);
    free(bins12);
    
    return ami;
}


int * histbinassign(const double y[], const int size, const double binEdges[], const int nEdges)
{
    
    
    // variable to store counted occurances in
    int * binIdentity = (int*)malloc(size * sizeof(int));
    for(int i = 0; i < size; i++)
    {
        // if not in any bin -> 0
        binIdentity[i] = 0;
        
        // go through bin edges
        for(int j = 0; j < nEdges; j++){
            if(y[i] < binEdges[j]){
                binIdentity[i] = j;
                break;
            }
        }
    }
    
    return binIdentity;
    
}

int * histcount_edges(const double y[], const int size, const double binEdges[], const int nEdges)
{
    
    
    int * histcounts = (int*)malloc(nEdges * sizeof(int));
    for(int i = 0; i < nEdges; i++){
        histcounts[i] = 0;
    }
    
    for(int i = 0; i < size; i++)
    {
        // go through bin edges
        for(int j = 0; j < nEdges; j++){
            if(y[i] <= binEdges[j]){
                histcounts[j] += 1;
                break;
            }
        }
    }
    
    return histcounts;
    
}


double CO_Embed2_Dist_tau_d_expfit_meandiff(const double y[], const int size)
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
    
    double * d = (double*)malloc((size-tau) * sizeof(double));
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
    int * histCounts = (int*)malloc(nBins * sizeof(double));
    double * binEdges = (double*)malloc((nBins + 1) * sizeof(double));
    histcounts_preallocated(d, size-tau-1, nBins, histCounts, binEdges);
    
    //printf("histcount ran\n");
    
    // normalise to probability
    double * histCountsNorm = (double*)malloc(nBins * sizeof(double));
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
    
    double * d_expfit_diff = (double*)malloc(nBins * sizeof(double));
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
    
    // arrays created dynamically in function histcounts
    free(d);
    free(d_expfit_diff);
    free(binEdges);
    free(histCountsNorm);
    free(histCounts);
    
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


int histcounts_preallocated(const double y[], const int size, int nBins, int * binCounts, double * binEdges)
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





// Rounds a value to n decimal places according to the precision
double round(double var, int precision) {
    if (precision < 0) precision = 0;
    double scale = pow(10, precision);
    // Proper rounding: add 0.5 for positive numbers, subtract 0.5 for negative numbers
    double value = (var >= 0) ? floor(var * scale + 0.5) : ceil(var * scale - 0.5);
    return value / scale;
}



double CO_f1ecac(const double y[], const int size)
{
    
    // compute autocorrelations
    double * autocorrs = co_autocorrs(y, size);
    // threshold to cross
    double thresh = 0.36;

    double out = (double)size;
    std::cout << "HOST" << out<< std::endl;
    for(int i = 0; i < size-2; i++){
        // printf("i=%d autocorrs_i=%1.3f\n", i, autocorrs[i]);
        if ( round(autocorrs[i+1],2) < thresh ){
            double m = round(autocorrs[i+1],2) - round(autocorrs[i],2);
            if (m == 0) continue; 
            double dy = thresh - round(autocorrs[i],2);
            double dx = dy/m;
            out = ((double)i) + round(dx,2);
            // printf("thresh=%1.3f AC(i)=%1.3f AC(i-1)=%1.3f m=%1.3f dy=%1.3f dx=%1.3f out=%1.3f\n", thresh, autocorrs[i], autocorrs[i-1], m, dy, dx, out);
            free(autocorrs);
            return out;
        }
    }
    
    free(autocorrs);
    
    return out;
    
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

cplx _Cmulcc(cplx x, cplx y) {
    /*double a = x._Val[0];
    double b = x._Val[1];

    double c = y._Val[0];
    double d = y._Val[1];

    cplx result = { (a * c - b * d), (a * d + c * b) };
        */
    return x*y;
}

cplx _Cminuscc(cplx x, cplx y) {
    //cplx result = { x._Val[0] - y._Val[0], x._Val[1] - y._Val[1] };
    return x - y;
}

cplx _Caddcc(cplx x, cplx y) {
    // cplx result = { x._Val[0] + y._Val[0], x._Val[1] + y._Val[1] };
    return x + y;
}

cplx _Cdivcc(cplx x, cplx y) {
    
    double a = real(x);
    double b = imag(x);

    double c = real(y);
    double d = imag(y);

    cplx result;
    result.real((a*c + b*d) / (c*c + d*d));
    result.imag((b*c - a*d)/(c*c + d*d));
    
    return result;
        
    // return x / y;
}

void twiddles(cplx a[], int size)
{

    double PI = 3.14159265359;

    for (int i = 0; i < size; i++) {
        // cplx tmp = { 0, -PI * i / size };
	    cplx tmp = {0.0, -PI * i / size };
        a[i] = exp(tmp);
        //a[i] = cexp(-I * M_PI * i / size);
    }
}

static void _fft(cplx a[], cplx out[], int size, int step, cplx tw[])
{   
    if (step < size) {
        _fft(out, a, size, step * 2, tw);
        _fft(out + step, a + step, size, step * 2, tw);

        for (int i = 0; i < size; i += 2 * step) {
            //cplx t = tw[i] * out[i + step];
            cplx t = _Cmulcc(tw[i], out[i + step]);
            a[i / 2] = _Caddcc(out[i], t);
            a[(i + size) / 2] = _Cminuscc(out[i], t);
        }
    }
}

void fft(cplx a[], int size, cplx tw[])
{
    cplx * out = (cplx*) malloc(size * sizeof(cplx));
    memcpy(out, a, size * sizeof(cplx));
    _fft(a, out, size, 1, tw);
    free(out);
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

void dot_multiply(cplx a[], cplx b[], int size)
{
    for (int i = 0; i < size; i++) {
        a[i] = _Cmulcc(a[i], conj(b[i]));
    }
}


double * co_autocorrs(const double y[], const int size)
{
    double m, nFFT;
    m = mean(y, size);
    nFFT = nextpow2(size) << 1;
    
    cplx * F = (cplx*) malloc(nFFT * 2 * sizeof *F);
    cplx * tw = (cplx*) malloc(nFFT * 2 * sizeof *tw);
    for (int i = 0; i < size; i++) {
        
        cplx tmp = { y[i] - m, 0.0 };
        F[i] = tmp;
        
    }
    for (int i = size; i < nFFT; i++) {
        
        cplx tmp = { 0.0, 0.0 };
        F[i] = tmp;

    }
    //size = nFFT;
    
    twiddles(tw, nFFT);
    fft(F, nFFT, tw);
    dot_multiply(F, F, nFFT);
    fft(F, nFFT, tw);
    cplx divisor = F[0];
    for (int i = 0; i < nFFT; i++) {
        F[i] = _Cdivcc(F[i], divisor); // F[i] / divisor;
    }
    
    double * out = (double*) malloc(nFFT * 2 * sizeof(out));
    for (int i = 0; i < nFFT; i++) {
        out[i] = real(F[i]);
    }
    free(F);
    free(tw);
    return out;
}


int CO_FirstMin_ac(const double y[], const int size)
{
    
    // NaN check
    // for(int i = 0; i < size; i++)
    // {
    //     if(isnan(y[i]))
    //     {
    //         return 0;
    //     }
    // }
    
    double * autocorrs = co_autocorrs(y, size);
    std::cout << "L jgjg" << autocorrs[0] << std::endl;
    std::cout << "L jgjg" << autocorrs[1] << std::endl;
    std::cout << "L jgjg" << autocorrs[2] << std::endl;
    std::cout << "L jgjg" << autocorrs[3] << std::endl;
    int minInd = size;
    for(int i = 1; i < size-1; i++)
    {
        if(autocorrs[i] < autocorrs[i-1] && autocorrs[i] < autocorrs[i+1])
        {
            minInd = i;
            break;
        }
    }
    
    free(autocorrs);
    
    return minInd;
    
}

int co_firstzero(const double y[], const int size, const int maxtau) {
    if (!y || size <= 0 || maxtau < 0) return -1;

    double* autocorrs = co_autocorrs(y, size);
    if (!autocorrs) return -1; // Error in autocorrelation computation

    int zerocrossind = 0;
    while(autocorrs[zerocrossind] > 0 && zerocrossind < maxtau)
    {
        zerocrossind += 1;
    }

    delete[] autocorrs;
    std::cout << "hostadv" <<zerocrossind;
    return zerocrossind;
}

double FC_LocalSimple_mean1_tauresrat(const double y[], const int size){
    return FC_LocalSimple_mean_tauresrat(y, size, 1);
}



double FC_LocalSimple_mean_tauresrat(const double y[], const int size, const int train_length) {
    // Sanitize inputs
    if (!y || size <= 0 || train_length <= 0 || size <= train_length) {
        fprintf(stderr, "Invalid input parameters.\n");
        return 0.0;
    }

    double* res = new double[size - train_length];
    for (int i = 0; i < size - train_length; i++) {
        double yest = 0.0;
        for (int j = 0; j < train_length; j++) {
            yest += y[i + j];
        }
        yest /= train_length;
        res[i] = y[i + train_length] - yest;
    }

    double resAC1stZ = co_firstzero(res, size - train_length, size - train_length);
    double yAC1stZ = co_firstzero(y, size, size);
    double output = (yAC1stZ != 0) ? resAC1stZ / yAC1stZ : 0.0; // Avoid division by zero

    delete[] res;
    return output;
}




int main(int argc, char** argv) {


    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <XCLBIN File>" << std::endl;
        return EXIT_FAILURE;
    }
   
    clock_t htod, dtoh, comp;


    /*====================================================CL===============================================================*/
    std::string binaryFile = argv[1];
    cl_int err;
    cl::Context context;
    cl::Kernel krnl1, krnl2, krnl3;
    cl::CommandQueue q;
   
    auto devices = get_xil_devices();
    auto fileBuf = read_binary_file(binaryFile);
    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
    bool valid_device = false;
    for (unsigned int i = 0; i < devices.size(); i++) {
        auto device = devices[i];
        OCL_CHECK(err, context = cl::Context(device, nullptr, nullptr, nullptr, &err));
        OCL_CHECK(err, q = cl::CommandQueue(context, device, 0, &err));
        std::cout << "Trying to program device[" << i << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
        cl::Program program(context, {device}, bins, nullptr, &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to program device[" << i << "] with xclbin file!\n";
        } else {
            std::cout << "Device[" << i << "]: program successful!\n";
            std::cout << "Setting CU(s) up..." << std::endl;
            OCL_CHECK(err, krnl1 = cl::Kernel(program, "krnl", &err));
            OCL_CHECK(err, krnl2 = cl::Kernel(program, "load", &err));
            OCL_CHECK(err, krnl3 = cl::Kernel(program, "store", &err));
            valid_device = true;
            break; // we break because we found a valid device
        }
    }
    if (!valid_device) {
        std::cout << "Failed to program any device found, exit!\n";
        exit(EXIT_FAILURE);
    }


    /*====================================================INIT INPUT/OUTPUT VECTORS===============================================================*/
    std::vector<data_t, aligned_allocator<data_t> > input(DATA_SIZE);
    data_t *input_sw = (data_t*) malloc(sizeof(data_t) * (2 * DATA_SIZE));
    std::vector<data_t, aligned_allocator<data_t> > output1_hw(1);
    std::vector<data_t, aligned_allocator<data_t> > output2_hw(1);
    std::vector<data_t, aligned_allocator<data_t> > output3_hw(1);
    data_t *output_sw = (data_t*) malloc(sizeof(data_t) * (22));


    for (int i = 0; i < DATA_SIZE-1; i++) {
        input_sw[i] = 0;
    }

    for (int i = 0; i < DATA_SIZE; i++) {
        input[i] = data[i];
        input_sw[DATA_SIZE + i - 1] = data[i];
    }

    /*====================================================SW VERIFICATION===============================================================*/


    output_sw[3] = CO_FirstMin_ac(input_sw, DATA_SIZE);
    output_sw[4] = FC_LocalSimple_mean1_tauresrat(input_sw, DATA_SIZE);
    output_sw[5] = CO_f1ecac(input_sw, DATA_SIZE);
    output_sw[6] = CO_Embed2_Dist_tau_d_expfit_meandiff(input_sw, DATA_SIZE);
    output_sw[7] = CO_HistogramAMI_even_2_5(input_sw, DATA_SIZE);
    output_sw[8] = SB_MotifThree_quantile_hh(input_sw, DATA_SIZE);
    output_sw[9] = SB_TransitionMatrix_3ac_sumdiagcov(input_sw, DATA_SIZE);
    output_sw[10] = DN_HistogramMode_5(input_sw, DATA_SIZE);
    output_sw[11] = DN_HistogramMode_10(input_sw, DATA_SIZE);
    output_sw[13] = SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1(input_sw, DATA_SIZE);
    output_sw[12] = SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1(input_sw, DATA_SIZE);
    output_sw[14] = PD_PeriodicityWang_th0_01(input_sw, DATA_SIZE);
    output_sw[15] = FC_LocalSimple_mean3_stderr(input_sw, DATA_SIZE);
    output_sw[16] = SP_Summaries_welch_rect_area_5_1(input_sw, DATA_SIZE);
    output_sw[17] = SP_Summaries_welch_rect_centroid(input_sw, DATA_SIZE);
    output_sw[1] = DN_OutlierInclude_p_001_mdrmd(input_sw, DATA_SIZE);
    output_sw[18] = DN_OutlierInclude_n_001_mdrmd(input_sw, DATA_SIZE);
    output_sw[19] = SB_BinaryStats_diff_longstretch0(input_sw, DATA_SIZE);
    output_sw[20] = SB_BinaryStats_mean_longstretch1(input_sw, DATA_SIZE);
    output_sw[2] = IN_AutoMutualInfoStats_40_gaussian_fmmi(input_sw, DATA_SIZE);
    output_sw[21] = CO_trev_1_num(input_sw, DATA_SIZE);
    output_sw[0] = MD_hrv_classic_pnn40(input_sw, DATA_SIZE);


    /*====================================================Setting up kernel I/O===============================================================*/

    /* INPUT BUFFERS */
    OCL_CHECK(err, cl::Buffer buffer_input(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(data_t) * DATA_SIZE, input.data(), &err));  


    /* OUTPUT BUFFERS */
    OCL_CHECK(err, cl::Buffer buffer_output1(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(data_t) * 1, output1_hw.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_output2(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(data_t) * 1, output2_hw.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_output3(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(data_t) * 1, output3_hw.data(), &err));

    /* SETTING INPUT PARAMETERS */
    OCL_CHECK(err, err = krnl2.setArg(0, buffer_input));
    OCL_CHECK(err, err = krnl3.setArg(3, buffer_output1));
    OCL_CHECK(err, err = krnl3.setArg(4, buffer_output2));
    OCL_CHECK(err, err = krnl3.setArg(5, buffer_output3));
    /*====================================================KERNEL===============================================================*/
    /* HOST -> DEVICE DATA TRANSFER*/
    std::cout << "HOST -> DEVICE" << std::endl;
    htod = clock();
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_input}, 0 /* 0 means from host*/));
    q.finish();
    htod = clock() - htod;
   
    /*STARTING KERNEL(S)*/
    std::cout << "STARTING KERNEL(S)" << std::endl;
    comp = clock();
   
    OCL_CHECK(err, err = q.enqueueTask(krnl2));
    OCL_CHECK(err, err = q.enqueueTask(krnl1));
    OCL_CHECK(err, err = q.enqueueTask(krnl3));
    q.finish();
    comp = clock() - comp;
    std::cout << "KERNEL(S) FINISHED" << std::endl;


    /*DEVICE -> HOST DATA TRANSFER*/
    std::cout << "HOST <- DEVICE" << std::endl;
    dtoh = clock();
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_output1}, CL_MIGRATE_MEM_OBJECT_HOST));
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_output2}, CL_MIGRATE_MEM_OBJECT_HOST));
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_output3}, CL_MIGRATE_MEM_OBJECT_HOST));
    q.finish();
    dtoh = clock() - dtoh;


    /*====================================================VERIFICATION & TIMING===============================================================*/

    printf("Host -> Device : %lf ms\n", 1000.0 * htod/CLOCKS_PER_SEC);
    printf("Device -> Host : %lf ms\n", 1000.0 * dtoh/CLOCKS_PER_SEC);
    printf("Computation : %lf ms\n",  1000.0 * comp/CLOCKS_PER_SEC);
    std::cout << std::endl; 
   
    bool match = true;

    // std::cout << "HW:" << output_hw[0] << " SW:" << output_sw[0] << std::endl;
    // if (output_hw[0] != output_sw[0]) match = false;
  
    std::cout << "HW:" << output1_hw[0] << " SW:" << output_sw[0] << std::endl;
    if (output1_hw[0] != output_sw[0]) match = false;
    std::cout << "HW:" << output2_hw[0] << " SW:" << output_sw[1] << std::endl;
    if (output2_hw[0] != output_sw[1]) match = false;
    std::cout << "HW:" << output3_hw[0] << " SW:" << output_sw[2] << std::endl;
    if (output3_hw[0] != output_sw[2]) match = false;

    std::cout << std::endl << "TEST " << (match ? "PASSED" : "FAILED") << std::endl;

    free(output_sw);
    free(input_sw);


    return (match ? EXIT_SUCCESS : EXIT_FAILURE);
}


