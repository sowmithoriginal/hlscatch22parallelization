#include "constants.h"

extern "C" {
    void store(hls::stream<data_t> &outp1,
        hls::stream<data_t> &outp2,
        hls::stream<data_t> &outp3,
        hls::stream<data_t> &outp4,
        hls::stream<data_t> &outp5,
        hls::stream<data_t> &outp6,
        hls::stream<data_t> &outp7,
        hls::stream<data_t> &outp8,
        hls::stream<data_t> &outp9,
        hls::stream<data_t> &outp10,
        hls::stream<data_t> &outp11,
        hls::stream<data_t> &outp12,
        hls::stream<data_t> &outp13,
        hls::stream<data_t> &outp14,
        hls::stream<data_t> &outp15,
        hls::stream<data_t> &outp16,
        hls::stream<data_t> &outp17,
        hls::stream<data_t> &outp18,
        hls::stream<data_t> &outp19,
        hls::stream<data_t> &outp20,
        hls::stream<data_t> &outp21,
        hls::stream<data_t> &outp22,
                data_t* output1,
                data_t* output2,
                data_t* output3,
                data_t* output4,
                data_t* output5,
                data_t* output6,
                data_t* output7,
                data_t* output8,
                data_t* output9,
                data_t* output10,
                data_t* output11,
                data_t* output12,
                data_t* output13,
                data_t* output14,
                data_t* output15,
                data_t* output16,
                data_t* output17,
                data_t* output18,
                data_t* output19,
                data_t* output20,
                data_t* output21,
                data_t* output22
                ) {
        #pragma HLS INTERFACE axis port = outp1
        #pragma HLS INTERFACE axis port = outp2
        #pragma HLS INTERFACE axis port = outp3
        #pragma HLS INTERFACE axis port = outp4
        #pragma HLS INTERFACE axis port = outp5
        #pragma HLS INTERFACE axis port = outp6
        #pragma HLS INTERFACE axis port = outp7
        #pragma HLS INTERFACE axis port = outp8
        #pragma HLS INTERFACE axis port = outp9
        #pragma HLS INTERFACE axis port = outp10
        #pragma HLS INTERFACE axis port = outp11
        #pragma HLS INTERFACE axis port = outp12
        #pragma HLS INTERFACE axis port = outp13
        #pragma HLS INTERFACE axis port = outp14
        #pragma HLS INTERFACE axis port = outp15
        #pragma HLS INTERFACE axis port = outp16
        #pragma HLS INTERFACE axis port = outp17
        #pragma HLS INTERFACE axis port = outp18
        #pragma HLS INTERFACE axis port = outp19
        #pragma HLS INTERFACE axis port = outp20
        #pragma HLS INTERFACE axis port = outp21
        #pragma HLS INTERFACE axis port = outp22
        #pragma HLS INTERFACE m_axi port = output1 bundle = gmem1
        #pragma HLS INTERFACE m_axi port = output2 bundle = gmem1
        #pragma HLS INTERFACE m_axi port = output3 bundle = gmem1
        #pragma HLS INTERFACE m_axi port = output4 bundle = gmem1
        #pragma HLS INTERFACE m_axi port = output5 bundle = gmem1
        #pragma HLS INTERFACE m_axi port = output6 bundle = gmem1
        #pragma HLS INTERFACE m_axi port = output7 bundle = gmem1
        #pragma HLS INTERFACE m_axi port = output8 bundle = gmem1
        #pragma HLS INTERFACE m_axi port = output9 bundle = gmem1
        #pragma HLS INTERFACE m_axi port = output10 bundle = gmem1
        #pragma HLS INTERFACE m_axi port = output11 bundle = gmem1
        #pragma HLS INTERFACE m_axi port = output12 bundle = gmem1
        #pragma HLS INTERFACE m_axi port = output13 bundle = gmem1
        #pragma HLS INTERFACE m_axi port = output14 bundle = gmem1
        #pragma HLS INTERFACE m_axi port = output15 bundle = gmem1
        #pragma HLS INTERFACE m_axi port = output16 bundle = gmem1
        #pragma HLS INTERFACE m_axi port = output17 bundle = gmem1
        #pragma HLS INTERFACE m_axi port = output18 bundle = gmem1
        #pragma HLS INTERFACE m_axi port = output19 bundle = gmem1
        #pragma HLS INTERFACE m_axi port = output20 bundle = gmem1
        #pragma HLS INTERFACE m_axi port = output21 bundle = gmem1
        #pragma HLS INTERFACE m_axi port = output22 bundle = gmem1
       

        #pragma HLS DATAFLOW
        
        static int counter = 0; 
        outp1 >> output1[counter];
        outp2 >> output2[counter];
        outp3 >> output3[counter];
        outp4 >> output4[counter];
        outp5 >> output5[counter];
        outp6 >> output6[counter];
        outp7 >> output7[counter];
        outp8 >> output8[counter];
        outp9 >> output9[counter];
        outp10 >> output10[counter];
        outp11 >> output11[counter];
        outp12 >> output12[counter];
        outp13 >> output13[counter];
        outp14 >> output14[counter];
        outp15 >> output15[counter];
        outp16 >> output16[counter];
        outp17 >> output17[counter];
        outp18 >> output18[counter];
        outp19 >> output19[counter];
        outp20 >> output20[counter];
        outp21 >> output21[counter];
        outp22 >> output22[counter];
        counter+=1;
    }
}
