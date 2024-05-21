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
// LOSS OF USE, DATA1, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// The views and conclusions contained in the software and documentation are those
// of the authors and should not be interpreted as representing official policies,
// either expressed or implied, of the SPIRAL project.

#include <math.h>
#include "ap_int.h"
#include "header_N.h"

template <typename T>
void switch_2_by_21(T in[2], T out[2], bool switch_on) {
	out[0] = in[(switch_on==true)?(1):(0)];
	out[1] = in[(switch_on==true)?(0):(1)];
}

template <typename T>
void spatial_permutation1(T spat_in[SW1], T spat_out[SW1], int y_idx[SW1]) {
	int i;
	for (i=0; i<SW1; i++) {
		spat_out[y_idx[i]] = spat_in[i];
	}
}

template <typename T>
void copy_vector1(T from[SW1], T to[SW1]) {
	int i;
	for (i=0; i<SW1; i++) {
		to[i] = from[i];
	}
}

template <typename T>
void connect_to_switch1(T con_to_swh_in[SW1], T con_to_swh_out[SW1/2][2], int x_idx[SW1]) {
#pragma HLS ARRAY_PARTITION variable=con_to_swh_in,con_to_swh_out,x_idx complete dim=1
#pragma HLS ARRAY_PARTITION variable=con_to_swh_out complete dim=2
#pragma HLS INLINE
	int i,j;
	for (i=0; i<SW1/2; i++) {
#pragma HLS UNROLL
		for (j=0; j<2; j++) {
			con_to_swh_out[i][j] = con_to_swh_in[x_idx[2*i+j]];
		}
	}
}

template <typename T>
void connect_from_switch1(T con_from_swh_in[SW1/2][2], T con_from_swh_out[SW1], int y_idx[SW1]) {
#pragma HLS ARRAY_PARTITION variable=con_from_swh_in,con_from_swh_out,y_idx complete dim=1
#pragma HLS ARRAY_PARTITION variable=con_from_swh_in complete dim=2
#pragma HLS INLINE
	int i,j;
	for (i=0; i<SW1/2; i++) {
#pragma HLS UNROLL
		for (j=0; j<2; j++) {
			con_from_swh_out[y_idx[2*i+j]] = con_from_swh_in[i][j];
		}
	}
}

template <typename T>
void switch_array1(int num, T swh_array_in[SW1/2][2], T swh_array_out[SW1/2][2], bool switch_on) {
	int i;
	for (i=0; i<num; i++) {
#pragma HLS UNROLL
		switch_2_by_21 <T> (swh_array_in[i], swh_array_out[i], switch_on);
	}
}

template <typename T>
void onestage1(T PRE_IN[SW1], T NEXT_PRE_IN[SW1],  bool control, int idx_in_or_out[SW1]) {
#pragma HLS INLINE
    T IN[SW1/2][2], OUT[SW1/2][2];

    connect_to_switch1 <T> (PRE_IN, IN, idx_in_or_out);
    switch_array1 <T> (SW1/2, IN, OUT, control);
    connect_from_switch1 <T> (OUT, NEXT_PRE_IN, idx_in_or_out);
}

template <typename T, int NUM_STAGE>
void switch_network_write1(T X[SW1], T Y[SW1], ap_uint<LOG2N1-LOG2SW1> bits, int init_perm_idx[SW1], int idx_in_or_out[NUM_STAGE][SW1], int control_bit[NUM_STAGE]) {
#pragma HLS ARRAY_PARTITION variable=X,Y complete dim=1
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS INLINE off

	T PRE_IN[NUM_STAGE+1][SW1];

	// initial permutation
	spatial_permutation1 <T> (X, PRE_IN[0], init_perm_idx);

	int i;
	for (i=0; i<NUM_STAGE; i++) {
#pragma HLS UNROLL
	    onestage1 <T> (PRE_IN[i], PRE_IN[i+1], bits.get_bit(control_bit[i]-LOG2SW1), idx_in_or_out[i]);
	}

	// output port
	copy_vector1 <T> (PRE_IN[NUM_STAGE], Y);
}

void buf_write_addr_generation1(ap_uint<LOG2N1-LOG2SW1> in_count, bool flip, int bit_seq[LOG2N1-LOG2SW1], ap_uint<LOG2N1-LOG2SW1+1> widx[2]) {
#pragma HLS INLINE
	int i, j;
	for (i=0; i<SW1; i++) {
	#pragma HLS UNROLL
		ap_uint<LOG2N1> bits = SW1*in_count+i;

		for (j=0; j<LOG2N1-LOG2SW1; j++) {
	#pragma HLS UNROLL
			bool bit = bits.get_bit( bit_seq[LOG2N1-LOG2SW1-1-j] );
			widx[i].set_bit(j, bit);
		}
		widx[i].set_bit(LOG2N1-LOG2SW1, flip);
	}
}

void buf_read_addr_generation1(ap_uint<LOG2N1-LOG2SW1> out_count, bool flip, ap_uint<LOG2N1-LOG2SW1+1> ridx[2]) {
	int i;
	for (i=0; i<SW1; i++) {
#pragma HLS UNROLL
		ridx[i].range(LOG2N1-LOG2SW1-1, 0) = out_count.range(LOG2N1-LOG2SW1-1, 0);
		ridx[i].set_bit(LOG2N1-LOG2SW1, flip);
	}
}

template <typename T, int NUM_STAGE>
void switch_network_read1(T X[SW1], T Y[SW1], ap_uint<LOG2N1-LOG2SW1> bits, int idx_in_or_out[NUM_STAGE][SW1], int control_bit[NUM_STAGE]) {
#pragma HLS ARRAY_PARTITION variable=X,Y complete dim=1
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS INLINE off

	T PRE_IN[NUM_STAGE+1][SW1];

//	// no initial permutation for the read switch network.
	copy_vector1 <T> (X, PRE_IN[0]);

	int i;
	for (i=0; i<NUM_STAGE; i++) {
#pragma HLS UNROLL
	    onestage1 <T> (PRE_IN[i], PRE_IN[i+1], bits.get_bit(control_bit[i]-LOG2SW1), idx_in_or_out[i]);
	}

	// output port
	copy_vector1 <T> (PRE_IN[NUM_STAGE], Y);
}

void buf_write1(cmplx_type buf[SW1][N1*2/SW1], content_addr1 in_post_switch[SW1]) {
	// buf write
	int i;
	for (i=0; i<SW1; i++) {
#pragma HLS UNROLL
		buf[i][in_post_switch[i].addr1] = in_post_switch[i].data1;
	}
}

void buf_read1(cmplx_type buf[SW1][N1*2/SW1], cmplx_type in_from_buff_pre_switch[SW1], ap_uint<LOG2N1-LOG2SW1+1> ridx_post_switch[SW1]) {
	// buf read
	int i;
	for (i=0; i<SW1; i++) {
//#pragma HLS UNROLL
		in_from_buff_pre_switch[i] = buf[i][ridx_post_switch[i]];
	}
}

void combine_addr_data1(content_addr1 combination[SW1], ap_uint<LOG2N1-LOG2SW1+1> addr1[SW1],
		cmplx_type data1[SW1]) {
	int i;
	for (i = 0; i < SW1; i++) {
#pragma HLS UNROLL
		combination[i].data1 = data1[i];
		combination[i].addr1 = addr1[i];
	}
}


void fixed_point_scale_after_dft1(cmplx_type x[R1], cmplx_type y[R1], ap_uint<LOG2_LOGrN1> i__) {
	int i;
	for (i=0; i<R1; i++) {

//		y[i].real = x[i].real >> ((i__!=9)?LOG2R1:3);
//		y[i].imag = x[i].imag >> ((i__!=9)?LOG2R1:3);
//		y[i].real = x[i].real ;
//		y[i].imag = x[i].imag ;
		y[i].real = x[i].real /exp2(LOG2R1); //>> LOG2R1;
		y[i].imag = x[i].imag /exp2(LOG2R1); // >> LOG2R1;
	}
}

#if R1 == 2
void dft1(cmplx_type x[2], cmplx_type y[2]) {
	CMPXADD(y[0], x[0], x[1]);
	CMPXSUB(y[1], x[0], x[1]);
}

#elif R1 == 4
void dft1(cmplx_type x[4], cmplx_type y[4]) {
	cmplx_type tmp[4];

	CMPXADD(tmp[0], x[0], x[2]);
	CMPXSUB(tmp[1], x[0], x[2]);
	CMPXADD(tmp[2], x[1], x[3]);
//	CMPXSUB_MUL_NI(tmp[3], x[1], x[3]);
	CMPX_NI_MUL_X_SUB_Y(tmp[3], x[1], x[3]);

	CMPXADD(y[0], tmp[0], tmp[2]);
	CMPXSUB(y[2], tmp[0], tmp[2]);
	CMPXADD(y[1], tmp[1], tmp[3]);
	CMPXSUB(y[3], tmp[1], tmp[3]);

}

#elif R1 == 8
void dft1(cmplx_type X[8], cmplx_type Y[8]) {
    cmplx_type s16, s17, s18, s19, s20, t66, t67, t68
            , t69, t70, t71, t72, t73, t74, t75, t76;
    CMPXADD(t66, X[0], X[4]);
    CMPXSUB(t67, X[0], X[4]);
    CMPXADD(t68, X[1], X[5]);
    CMPX_TWR_NTWR_MUL_X_SUB_Y(s16, 0.70710678118654757, X[1], X[5]);
    CMPXADD(t69, X[2], X[6]);
    CMPX_NI_MUL_X_SUB_Y(s17, X[2], X[6]);
    CMPXADD(t70, X[3], X[7]);
    CMPX_TWR_TWR_MUL_X_SUB_Y(s18, -0.70710678118654757, X[3], X[7]);
    CMPXADD(t71, t66, t69);
    CMPXSUB(t72, t66, t69);
    CMPXADD(t73, t68, t70);
    CMPX_NI_MUL_X_SUB_Y(s19, t68, t70);
    CMPXADD(Y[0], t71, t73);
    CMPXSUB(Y[4], t71, t73);
    CMPXADD(Y[2], t72, s19);
    CMPXSUB(Y[6], t72, s19);
    CMPXADD(t74, t67, s17);
    CMPXSUB(t75, t67, s17);
    CMPXADD(t76, s16, s18);
    CMPX_NI_MUL_X_SUB_Y(s20, s16, s18);
    CMPXADD(Y[1], t74, t76);
    CMPXSUB(Y[5], t74, t76);
    CMPXADD(Y[3], t75, s20);
    CMPXSUB(Y[7], t75, s20);

}
#endif


void convert_2d_to_1d1(cmplx_type X[SW1/R1][R1], cmplx_type Y[SW1]) {
	int i,j;
	for (i=0; i<SW1/R1; i++) {
		for (j=0; j<R1; j++) {
			Y[R1*i+j] = X[i][j];
		}
	}
}

void convert_1d_to_2d1(cmplx_type X[SW1], cmplx_type Y[SW1/R1][R1]) {
	int i,j;
	for (i=0; i<SW1/R1; i++) {
		for (j=0; j<R1; j++) {
			Y[i][j] = X[R1*i+j];
		}
	}
}

void dft_bundle_module_old1(cmplx_type dft_in[SW1], cmplx_type dft_out[SW1]) {
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS INLINE off
#pragma HLS PIPELINE

	cmplx_type data_dft_in[SW1/R1][R1], data_dft_out[SW1/R1][R1];
	cmplx_type data_scaled[SW1/R1][R1];

	convert_1d_to_2d1(dft_in, data_dft_in);

	int i;
	for (i=0; i<SW1/R1; i++) {
#pragma HLS UNROLL
		dft1(data_dft_in[i], data_dft_out[i]);
#ifdef FIXED_POINT
//		fixed_point_scale_after_dft(data_dft_out[i], data_scaled[i]);
#endif
	}

#ifdef FIXED_POINT
	convert_2d_to_1d1(data_scaled, dft_out);
#else
	convert_2d_to_1d1(data_dft_out, dft_out);
#endif
}


void dft_bundle_module1(cmplx_type dft_in[SW1], cmplx_type dft_out[SW1], ap_uint<LOG2_LOGrN1> i__) {
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS INLINE off
#pragma HLS PIPELINE

	cmplx_type data_dft_in[SW1/R1][R1], data_dft_out[SW1/R1][R1];
	cmplx_type data_scaled[SW1/R1][R1];

	convert_1d_to_2d1(dft_in, data_dft_in);

	int i;
	for (i=0; i<SW1/R1; i++) {
#pragma HLS UNROLL
		dft1(data_dft_in[i], data_dft_out[i]);
#ifdef FIXED_POINT
		fixed_point_scale_after_dft1(data_dft_out[i], data_scaled[i], i__);
#endif
	}

#ifdef FIXED_POINT
	convert_2d_to_1d1(data_scaled, dft_out);
#else
	convert_2d_to_1d1(data_dft_out, dft_out);
#endif
}

void twidscale_module1(cmplx_type data_pre_twid[SW1], cmplx_type data_post[SW1], ap_uint<LOG2_LOGrN1> i) {
#pragma HLS INTERFACE ap_ctrl_none port=return
#pragma HLS INLINE off
#pragma HLS PIPELINE

	const VAL_TYPE twid_table_R[N1] ={
#ifdef FIXED_POINT
	#include "twidR_fixed_point1.txt"
#else
	#include "twidR1.txt"
#endif
	};
//#pragma HLS RESOURCE variable=twid_table_R core=ROM_nP_BRAM
//#pragma HLS ARRAY_MAP variable=twid_table_R instance=twid_RI vertical
	const VAL_TYPE twid_table_I[N1] ={
#ifdef FIXED_POINT
	#include "twidI_fixed_point1.txt"
#else
	#include "twidI1.txt"
#endif
	};
//#pragma HLS RESOURCE variable=twid_table_I core=ROM_nP_BRAM
//#pragma HLS ARRAY_MAP variable=twid_table_I instance=twid_RI vertical

	static ap_uint<LOG2N1-LOG2SW1> j_count = N1/SW1-1;

	if (j_count==N1/SW1-1) {
		j_count = 0;
	}
	else {
		j_count++;
	}


	ap_uint<LOG2N1> twid_idx_ofset;
	twid_idx_ofset.range(LOG2R1-1, 0) = R1-1;
	twid_idx_ofset.range(LOG2N1-1, LOG2R1) = (N1/R1-1) << ( LOG2R1*(LOGrN1-1-i)  );

	int l,j;
	ap_uint<LOG2N1> ld_idx_base = SW1*j_count;

	for (j=0; j<SW1/R1; j++) {
		data_post[R1*j + 0] = data_pre_twid[R1*j + 0];

		for (l=1; l<R1; l++) {
			int ld_idx = ld_idx_base + j*R1 + l;
			cmplx_type twid;
			twid.real = twid_table_R[ld_idx & twid_idx_ofset];
			twid.imag = twid_table_I[ld_idx & twid_idx_ofset];

			CMPXMUL(data_post[j*R1 + l], twid, data_pre_twid[j*R1 + l]);
		}
	}
}





#define LATENCY 46
#define LOG_LATENCY 6


void digit_rev1(unsigned int in, unsigned int* out, unsigned int bitwid) {
#pragma HLS INLINE
	int j;
	ap_uint<LOG2R1> dummy1 = ~0;
	unsigned int idx = 0;
	for (j=0; j<bitwid; j+=LOG2R1) {
#pragma HLS UNROLL
		idx = R1*idx + ((in>>j)&dummy1);
	}
//	*out = (bitwid==0)?in:idx;
	*out = idx;
}




#if SW1*SW1<=N1
//lighter compute
void digit_rev_perm_N1(cmplx_type x[N1],cmplx_type y[N1]) {
#pragma HLS INLINE off
DO_PRAGMA(HLS ARRAY_PARTITION variable=x cyclic factor=SW1 dim=1)
DO_PRAGMA(HLS ARRAY_PARTITION variable=y cyclic factor=SW1 dim=1)
#pragma HLS RESOURCE variable=x,y core=RAM_S2P_BRAM
	unsigned int i;
	unsigned int in_mod, in_mul, out_prt;
	unsigned int out_mod, out_mul, in_prt;
	int in_idx, out_idx;
	cycle_modulo: for (in_mod=0; in_mod<N1/SW1/SW1; in_mod++) {
DO_PRAGMA(HLS PIPELINE II=SW1 rewind)
		digit_rev1(in_mod, &out_mod, LOG2N1 - 2*LOG2SW1);
		in_prt=0; out_prt=0;
		cycle_multiplier_prt: for (i=0; i<SW1*SW1; i++) {
			out_prt = i%SW1;
//			in_prt = (out_prt + i>>LOGSW)%SW1; HLS DOESN'T PIPELINE WELL WITH THIS.

			digit_rev1(out_prt, &in_mul, LOG2SW1);
			digit_rev1(in_prt, &out_mul, LOG2SW1);

			in_idx = (in_mul*N1/SW1) + (in_mod*SW1) + in_prt;
			out_idx = (out_mul*N1/SW1) + (out_mod*SW1) + out_prt;

//			printf("in_mul=%d, in_mod=%d, in_prt=%d\n1", in_mul, in_mod, in_prt);
//			printf("ou_mul=%d, ou_mod=%d, ou_prt=%d\n1", out_mul, out_mod, out_prt);
//			printf("x[%2d] -> y[%2d], port: [%d] -> [%d]\n1", in_idx, out_idx, in_idx%SW1, out_idx%SW1);

			y[out_idx] = x[in_idx];

//			out_prt = (out_prt+1)%SW1;
			in_prt = (in_prt+1)%SW1;
			if (i%SW1==SW1-1)
				in_prt = (in_prt+1)%SW1;
		}
	}
}

#else
void digit_rev_perm_N1(cmplx_type x[N1],cmplx_type y[N1]) {
DO_PRAGMA(HLS PIPELINE II=N_O_SW rewind)
DO_PRAGMA(HLS ARRAY_PARTITION variable=x cyclic factor=SW1 dim=1)
DO_PRAGMA(HLS ARRAY_PARTITION variable=y cyclic factor=SW1 dim=1)
#pragma HLS RESOURCE variable=x,y core=RAM_S2P_BRAM
	unsigned int i, in_mul, out_prt, pad, pad_rev, out_prt_rev;
//	ap_uint<LOGN-LOGSW> pad;

//	unsigned int mask = (1<<(2*LOGSW-LOGN))-1;

	int in_idx, out_idx;
	padd: for (i=0; i<N1/SW1; i++) {
// doesn't work because modulo scheduling is impossible. (port in -> port out pairs must be the same for each iteration)
//DO_PRAGMA(HLS PIPELINE II=SW1 rewind)
		pad = i;
		all_prts: for (out_prt=0; out_prt<SW1; out_prt++) {
			digit_rev1(pad, &pad_rev, LOGrN1-LOG2SW1);
			digit_rev1(out_prt, &out_prt_rev, LOG2SW1);

//			unsigned int in_prt = ( (out_prt_rev & mask) <<(LOGN-LOGSW) ) + pad_rev;
////			printf("in_prt2 = %d\n1", in_prt); // correct
////			unsigned int in_idx2;
//			in_idx = ((out_prt_rev>>(2*LOGSW-LOGN))<<LOGSW) + in_prt;
////			printf("in_idx2 = %d\n1", in_idx2);

			in_idx = (out_prt_rev *N1/SW1) + pad_rev;
			out_idx = (pad *SW1) + out_prt;
//			printf("x[%2d] -> y[%2d], port: [%d] -> [%d]\n1", in_idx, out_idx, in_idx%SW1, out_idx%SW1);
			y[out_idx] = x[in_idx];

			pad = (pad+1)% (N1/SW1);
		}
	}
}
#endif

void digit_rev_perm_switch_write1(cmplx_type data_out_switched[SW1], cmplx_type bram[SW1][N1*2/SW1], bool wt_offset) {
	static ap_uint<LOG2N1-LOG2SW1> j_com = N1/SW1-1;
	if (j_com==N1/SW1-1) {
		j_com = 0;
	}
	else {
		j_com++;
	}

	perm_config1<DIGIT_REV_NUM_STAGE> config = {
#include "dig_rev_perm_config1.dat"
	};

	// buf write addr1 generation
	ap_uint<LOG2N1-LOG2SW1+1> w_addr[SW1];

	buf_write_addr_generation1(j_com, wt_offset, config.w_addr_bit_seq, w_addr);



	// input switch (addr1 + data1)
	// compose the combination of write addr1 and data1.
	content_addr1 in_pre_switch[SW1], in_post_switch[SW1];
	combine_addr_data1(in_pre_switch, w_addr, data_out_switched);

	switch_network_write1 <content_addr1, DIGIT_REV_NUM_STAGE> (in_pre_switch, in_post_switch, j_com, config.init_perm_idx, config.w_switch_connection_idx, config.w_switch_control_bit);

	// buf write
	buf_write1(bram, in_post_switch);
}


void digit_rev_perm_read_switch1(cmplx_type bram[SW1][N1*2/SW1], bool rd_offset, bool first_stage, cmplx_type data_in[SW1]) {
	static ap_uint<LOG2N1-LOG2SW1> j_readbuf = N1/SW1-1;

	if (j_readbuf==N1/SW1-1) {
		j_readbuf = 0;
	}
	else {
		j_readbuf++;
	}

	perm_config1<DIGIT_REV_NUM_STAGE> config = {
#include "dig_rev_perm_config1.dat"
	};

	// buf read addr1 generation
	ap_uint<LOG2N1-LOG2SW1+1> r_addr[SW1], r_addr_post_switch[SW1];
	buf_read_addr_generation1(j_readbuf, rd_offset, r_addr);

	// switch the read address
	//  (actually unnecessary because they are always the same!!)

	// buf read
	cmplx_type in_from_buff_pre_switch[SW1];
	buf_read1(bram, in_from_buff_pre_switch, r_addr);

	// switch out data1
	if (first_stage) {
		copy_vector1(in_from_buff_pre_switch, data_in);
	}
	else {
		switch_network_read1 <cmplx_type, DIGIT_REV_NUM_STAGE>(in_from_buff_pre_switch, data_in, j_readbuf, config.r_switch_connection_idx, config.r_switch_control_bit);
	}
}

void digrev_or_stride_perm_read_switch1(cmplx_type bram[SW1][N1*2/SW1], bool rd_offset, bool first_stage, cmplx_type data_in[SW1]) {
	static ap_uint<LOG2N1-LOG2SW1> j_readbuf = N1/SW1-1;

	if (j_readbuf==N1/SW1-1) {
		j_readbuf = 0;
	}
	else {
		j_readbuf++;
	}

	perm_config1<STRIDE_PERM_SWITCH_NUM_STAGE> config = {
#include "stride_perm_config1.dat"
	};
	perm_config1<DIGIT_REV_NUM_STAGE> dig_rev_config = {
#include "dig_rev_perm_config1.dat"
	};

	// buf read addr1 generation
	ap_uint<LOG2N1-LOG2SW1+1> r_addr[SW1], r_addr_post_switch[SW1];
	buf_read_addr_generation1(j_readbuf, rd_offset, r_addr);

	// switch the read address
	//  (actually unnecessary because they are always the same!!)

	// buf read
	cmplx_type in_from_buff_pre_switch[SW1];
	buf_read1(bram, in_from_buff_pre_switch, r_addr);

	// switch out data1
	if (first_stage) {
		copy_vector1(in_from_buff_pre_switch, data_in);
		switch_network_read1 <cmplx_type, DIGIT_REV_NUM_STAGE>(in_from_buff_pre_switch, data_in, j_readbuf, dig_rev_config.r_switch_connection_idx, dig_rev_config.r_switch_control_bit);
	}
	else {
		switch_network_read1 <cmplx_type, STRIDE_PERM_SWITCH_NUM_STAGE>(in_from_buff_pre_switch, data_in, j_readbuf, config.r_switch_connection_idx, config.r_switch_control_bit);
	}
}

void stride_perm_switch_write1(cmplx_type data_out_switched[SW1], cmplx_type bram[SW1][N1*2/SW1], bool wt_offset) {
	static ap_uint<LOG2N1-LOG2SW1> j_com = N1/SW1-1;
	if (j_com==N1/SW1-1) {
		j_com = 0;
	}
	else {
		j_com++;
	}

	perm_config1<STRIDE_PERM_SWITCH_NUM_STAGE> config = {
#include "stride_perm_config1.dat"
	};

	// buf write addr1 generation
	ap_uint<LOG2N1-LOG2SW1+1> w_addr[SW1];

	buf_write_addr_generation1(j_com, wt_offset, config.w_addr_bit_seq, w_addr);


	// input switch (addr1 + data1)
	// compose the combination of write addr1 and data1.
	content_addr1 in_pre_switch[SW1], in_post_switch[SW1];
	combine_addr_data1(in_pre_switch, w_addr, data_out_switched);

	switch_network_write1 <content_addr1, STRIDE_PERM_SWITCH_NUM_STAGE> (in_pre_switch, in_post_switch, j_com, config.init_perm_idx, config.w_switch_connection_idx, config.w_switch_control_bit);

	// buf write
	buf_write1(bram, in_post_switch);
}

void final_stride_perm_switch1(cmplx_type bram[SW1][N1*2/SW1], bool rd_offset, cmplx_type output[SW1]) {
	bool wt_offset = !rd_offset;
	static ap_uint<LOG2N1-LOG2SW1> j_readbuf = N1/SW1-1;

	if (j_readbuf==N1/SW1-1) {
		j_readbuf = 0;
	}
	else {
		j_readbuf++;
	}

	perm_config1<STRIDE_PERM_SWITCH_NUM_STAGE> config = {
#include "stride_perm_config1.dat"
	};

	// buf read addr1 generation
	ap_uint<LOG2N1-LOG2SW1+1> r_addr[SW1], r_addr_post_switch[SW1];
	buf_read_addr_generation1(j_readbuf, rd_offset, r_addr);

	// switch the read address
	//  (actually unnecessary because they are always the same!!)

	// buf read
	cmplx_type in_from_buff_pre_switch[SW1], data_in[SW1];
	buf_read1(bram, in_from_buff_pre_switch, r_addr);

	// switch out data1
	switch_network_read1 <cmplx_type, STRIDE_PERM_SWITCH_NUM_STAGE>(in_from_buff_pre_switch, output, j_readbuf, config.r_switch_connection_idx, config.r_switch_control_bit);


}

extern "C" void pease_fft_N(cmplx_type X[N1], cmplx_type Y[N1]) {
#pragma HLS DATA_PACK variable=X
#pragma HLS DATA_PACK variable=Y
#pragma HLS INTERFACE axis port=X
#pragma HLS INTERFACE axis port=Y
DO_PRAGMA(HLS ARRAY_PARTITION variable=X cyclic factor=SW1 dim=1)
DO_PRAGMA(HLS ARRAY_PARTITION variable=Y cyclic factor=SW1 dim=1)

	int i;                          // loop iterator for Log(R1)N1 stages
	int j;                          // loop iterator for N1/SW1 butterfly bundles
	int k;                          // loop iterator for SW1/R1 butterflies
	cmplx_type in[SW1];     			// temporary variable for data1 stream in
	cmplx_type data_in[SW1];  		// temporary variable for data1 pre-twiddled
	cmplx_type data_twiddled[SW1]; 	// temporary variable for data1 post-twiddled
	cmplx_type data_out[SW1]; 		// temporary variable for data1 post base dft1
	cmplx_type out[SW1];   			// temporary variable for data1 stream out
	cmplx_type buf[SW1][N1*2/SW1];     // buffer for iterative calculation (num of ports == SW1; size == 2N)
//#pragma HLS DATA_PACK variable=buf
#pragma HLS RESOURCE variable=buf core=RAM_S2P_BRAM
#pragma HLS ARRAY_PARTITION variable=buf complete dim=1
	bool digit_rev_flip;            // whether to flip BRAM when writing for digit-rev permutation.
	bool rd_flip, wt_flip;          // whether to flip BRAM for R1/W for stride permutation.
	bool final_rd_flip;             // whether to flip BRAM for R1 for the final stride permutation.

	digit_rev_flip= false;          // write streamed in data1 to BRAM without addr1 offset
	for (j=0; j<N1/SW1; j++) {
#pragma HLS PIPELINE
		for (k=0; k<SW1; k++) {
			in[k] = X[SW1*j+k];
		}
		digit_rev_perm_switch_write1(in, buf, digit_rev_flip);
	}

	for (i=0; i<LOGrN1; i++) {
		for (j=0; j<N1/SW1; j++) {
#pragma HLS DEPENDENCE variable=buf inter false
#pragma HLS PIPELINE
			rd_flip = ((i%2==0)?(false):(true));
			wt_flip = !rd_flip;

			digrev_or_stride_perm_read_switch1(buf, rd_flip, i==0, data_in);

			twidscale_module1(data_in, data_twiddled, i);
			dft_bundle_module1(data_twiddled, data_out, i);

			stride_perm_switch_write1(data_out, buf, wt_flip);
		}
	}

	final_rd_flip = ((LOGrN1%2==0)?(false):(true));
	for (j=0; j<N1/SW1; j++) {
#pragma HLS PIPELINE
		final_stride_perm_switch1(buf, final_rd_flip, out);
		for (k=0; k<SW1; k++) {
			Y[SW1*j+k] = out[k];
		}
	}
}
