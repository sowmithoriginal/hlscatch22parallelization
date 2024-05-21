#include "constants.h"

extern "C" {
    void store(hls::stream<data_t> &outp1,
        hls::stream<data_t> &outp2,
               hls::stream<data_t> &outp3,
    //            hls::stream<data_t> &coarseg,
    //            hls::stream<data_t> &hrvpnn,
    //            hls::stream<data_t> &cotrev1,
    //            hls::stream<data_t> &gaussian, 
                data_t* output1,
                data_t* output2,
                data_t* output3
            //     data_t* ar_cotrev1,
            //     data_t* ar_gaussian
                ) {
        #pragma HLS INTERFACE axis port = outp2
        #pragma HLS INTERFACE axis port = outp3
        // #pragma HLS INTERFACE axis port = coarseg
        // #pragma HLS INTERFACE axis port = hrvpnn
        // #pragma HLS INTERFACE axis port = cotrev1
        // #pragma HLS INTERFACE axis port = gaussian
        #pragma HLS INTERFACE axis port = outp1
        #pragma HLS INTERFACE m_axi port = output1 bundle = gmem1
        #pragma HLS INTERFACE m_axi port = output2 bundle = gmem1
        #pragma HLS INTERFACE m_axi port = output3 bundle = gmem1
        // #pragma HLS INTERFACE m_axi port = ar_cotrev1 bundle = gmem2
        // #pragma HLS INTERFACE m_axi port = ar_gaussian bundle = gmem2

        #pragma HLS DATAFLOW
        
        static int counter = 0; 
        // longs0 >> ar_longs0[counter];
        // longs1 >> ar_longs1[counter];
        // coarseg >> ar_coarseg[counter];
        // hrvpnn >> ar_hrvpnn[counter];
        outp1 >> output1[counter];
        outp2 >> output2[counter];
        outp3 >> output3[counter];
        counter+=1;
    }
}
