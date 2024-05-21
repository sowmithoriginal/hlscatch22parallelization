// Author: Guanglin Xu (guanglix@andrew.cmu.edu)
//         Carnegie Mellon University
// Date: 05/20/2017
// Copyright reserved.
// Distribution and commercial uses are not allowed without permission.

#ifndef _HEADER_N_H
#define _HEADER_N_H

#include "header.h"
// N1 is the problem size.
//  N1 can be power of 2. The tested range are 64-4096
//  N1<64 may fail because of pipeline hazards
// R1 is the radix
// SW1 is the streaming width. SW1=N1*R1 (N1>=1) must be satisfied
#define N1 128
#define R1 2
#define SW1 4
// must be 2

#define LOGrN1 7
#define LOG2N1 7
#define LOG2R1 1
#define LOG2SW1 2
#define LOG2_LOGrN1 3

#include "stride_perm_num_stage1.h"
#include "dig_rev_perm_num_stage1.h"

template <int NUM_STAGE>
struct perm_config1{
	int init_perm_idx[SW1];

	int w_switch_connection_idx[NUM_STAGE][SW1];
	int w_switch_control_bit[NUM_STAGE];
	int w_addr_bit_seq[LOG2N1-LOG2SW1];

	int r_switch_connection_idx[NUM_STAGE][SW1];
	int r_switch_control_bit[NUM_STAGE];
};


// #define USE_FFTW
// #define DOUBLE
// #define FIXED_POINT_PREC 32

// #define PRAGMA_SUB(x) _Pragma (#x)
// #define DO_PRAGMA(x) PRAGMA_SUB(x)


// #ifdef SINGLE
// typedef float VAL_TYPE;
// #endif
// #ifdef DOUBLE
// typedef double VAL_TYPE;
// #endif
// #ifdef FIXED_POINT
// #include "ap_fixed.h"
// typedef ap_fixed<FIXED_POINT_PREC,4> VAL_TYPE;
// #endif

// typedef struct {
// 	float real;
// 	float imag;
// } float_cmplx_type;

// typedef struct {
// 	VAL_TYPE real;
// 	VAL_TYPE imag;
// } cmplx_type;

#include "ap_int.h"

typedef struct {
	cmplx_type data1;
	ap_uint<LOG2N1-LOG2SW1+1> addr1;
} content_addr1;

#include "hls_stream.h"
void disp_complex_type1(char* name,  cmplx_type x[N1]);
extern "C" void pease_fft_N(cmplx_type X[N1], cmplx_type Y[N1]);
void digit_rev1(unsigned int in, unsigned int* out, unsigned int bitwid);

void buf_read_addr_generation1(ap_uint<LOG2N1-LOG2SW1> out_count, bool flip, ap_uint<LOG2N1-LOG2SW1+1> ridx[2]);
void buf_read1(cmplx_type buf[SW1][N1*2/SW1], cmplx_type in_from_buff_pre_switch[SW1], ap_uint<LOG2N1-LOG2SW1+1> ridx_post_switch[SW1]);
void buf_write_addr_generation1(ap_uint<LOG2N1-LOG2SW1> in_count, bool flip, int bit_seq[LOG2N1-LOG2SW1], ap_uint<LOG2N1-LOG2SW1+1> widx[2]);
void combine_addr_data1(content_addr1 combination[SW1], ap_uint<LOG2N1-LOG2SW1+1> addr1[SW1],
		cmplx_type data1[SW1]);
template <typename T>
void switch_network_write1(T X[SW1], T Y[SW1], ap_uint<LOG2N1-LOG2SW1> bits, int init_perm_idx[SW1], int idx_in_or_out[STRIDE_PERM_SWITCH_NUM_STAGE][SW1], int control_bit[STRIDE_PERM_SWITCH_NUM_STAGE]);
template <typename T>
void switch_network_read1(T X[SW1], T Y[SW1], ap_uint<LOG2N1-LOG2SW1> bits, int idx_in_or_out[STRIDE_PERM_SWITCH_NUM_STAGE][SW1], int control_bit[STRIDE_PERM_SWITCH_NUM_STAGE]);
void buf_write1(cmplx_type buf[SW1][N1*2/SW1], content_addr1 in_post_switch[SW1]);

#define CMPXADD(Z, X, Y)  ({ \
	(Z).real = (X).real + (Y).real;   \
	(Z).imag = (X).imag + (Y).imag; \
	})
#define CMPXSUB(Z, X, Y)  ({ \
	(Z).real = (X).real - (Y).real; \
	(Z).imag = (X).imag - (Y).imag; \
	})
#define CMPXMUL(Z, X, Y)  ({ \
	(Z).real = (X).real * (Y).real - (X).imag * (Y).imag; \
	(Z).imag = (X).real * (Y).imag + (X).imag * (Y).real; \
	})

#define CMPXDIV(Z, X, Y)  ({ \
	(Z).real = (X).real * (Y).real - (X).imag * (Y).imag; \
	(Z).imag = (X).real * (Y).imag + (X).imag * (Y).real; \
	})

#define CMPXCONJ(Y, X)  ({ \
	(Y).real = (X).real; \
	(Y).imag = -(X).imag; \
	})

#define CMPXSUB_MUL_NI(Z, X, Y)  ({ \
	(Z).imag = -((X).real - (Y).real); \
	(Z).real = (X).imag - (Y).imag; \
	})
#define CMPXSUB_MUL_07R_N07I(Z, X, Y, T)  ({ \
		(T).real = (X).real - (Y).real; \
		(T).imag = (X).imag - (Y).imag; \
		(Z).real = (FP_TYPE)0.70710678118654757 * ((T).real + (T).imag); \
		(Z).imag = (FP_TYPE)0.70710678118654757 * ((T).imag - (T).real); \
	})
#define CMPXSUB_MUL_N07R_N07I(Z, X, Y, T)  ({ \
		(T).real = (X).real - (Y).real; \
		(T).imag = (X).imag - (Y).imag; \
		(Z).real = (FP_TYPE)0.70710678118654757 * ((T).imag - (T).real); \
		(Z).imag = (FP_TYPE)-0.70710678118654757 * ((T).real + (T).imag); \
	})



// 2 sub
#define CMPX_NI_MUL_X_SUB_Y(Z, X, Y)  ({ \
	(Z).real = (X).imag - (Y).imag; \
	(Z).imag = -((X).real - (Y).real); \
	})
// 2 add
#define CMPX_NI_MUL_X_ADD_Y(Z, X, Y)  ({ \
	(Z).real = (X).imag + (Y).imag; \
	(Z).imag = -((X).real + (Y).real); \
	})
// 2 sub
#define CMPX_I_MUL_X_SUB_Y(Z, X, Y)  ({ \
	(Z).real = -((X).imag - (Y).imag); \
	(Z).imag = (X).real - (Y).real; \
	})
// 2 add
#define CMPX_I_MUL_X_ADD_Y(Z, X, Y)  ({ \
	(Z).real = -((X).imag + (Y).imag); \
	(Z).imag = (X).real + (Y).real; \
	})

// 1 add, 5 sub, 2 mul
#define CMPX_TWR_TWR_MUL_X_SUB_Y(Z, TW_R, X, Y )  ({ \
	(Z).real = (FP_TYPE)TW_R * (((X).real - (Y).real) - ((X).imag - (Y).imag)); \
	(Z).imag = (FP_TYPE)TW_R * (((X).real - (Y).real) + ((X).imag - (Y).imag)); \
	})
// 5 add, 1 sub, 2 mul
#define CMPX_TWR_TWR_MUL_X_ADD_Y(Z, TW_R, X, Y)  ({ \
	(Z).real = (FP_TYPE)TW_R * (((X).real + (Y).real) - ((X).imag + (Y).imag)); \
	(Z).imag = (FP_TYPE)TW_R * (((X).real + (Y).real) + ((X).imag + (Y).imag)); \
	})
// 1 add, 5 sub, 2 mul
#define CMPX_TWR_NTWR_MUL_X_SUB_Y(Z, TW_R, X, Y)  ({ \
	(Z).real = (FP_TYPE)TW_R * (((X).real - (Y).real) + ((X).imag - (Y).imag)); \
	(Z).imag = (FP_TYPE)TW_R * (((X).imag - (Y).imag) - ((X).real - (Y).real)); \
	})
// 5 add, 1 sub, 2 mul
#define CMPX_TWR_NTWR_MUL_X_ADD_Y(Z, TW_R, X, Y)  ({ \
	(Z).real = (FP_TYPE)TW_R * (((X).real + (Y).real) + ((X).imag + (Y).imag)); \
	(Z).imag = (FP_TYPE)TW_R * (((X).imag + (Y).imag) - ((X).real + (Y).real)); \
	})

// 1 add, 5 sub, 4 mul
#define CMPX_TWR_TWI_MUL_X_SUB_Y(Z, TW_R, TW_I, X, Y)  ({ \
	(Z).real = (FP_TYPE)TW_R * ((X).real - (Y).real) - TW_I * ((X).imag - (Y).imag); \
	(Z).imag = (FP_TYPE)TW_R * ((X).imag - (Y).imag) + TW_I * ((X).real - (Y).real); \
	})
// 5 add, 1 sub, 4 mul
#define CMPX_TWR_TWI_MUL_X_ADD_Y(Z, TW_R, TW_I, X, Y)  ({ \
	(Z).real = (FP_TYPE)TW_R * ((X).real + (Y).real) - TW_I * ((X).imag + (Y).imag); \
	(Z).imag = (FP_TYPE)TW_R * ((X).imag + (Y).imag) + TW_I * ((X).real + (Y).real); \
	})


#define MAX(X,Y) ({ \
		(X>Y)?(X):(Y); \
	})

#endif
