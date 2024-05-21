#include "constants.h"

extern "C" {
    void load(data_t* input, hls::stream<data_t> &output) {
        #pragma HLS INTERFACE m_axi port = input bundle = gmem0
        #pragma HLS INTERFACE axis port = output

        #pragma HLS DATAFLOW
        
        static int counter = 0; 
        output << input[counter];
        counter+=1; 
    }
}
