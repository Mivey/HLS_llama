#include "mha_forward.h"
#include "swiglu.h"

void swiglu_kernel(s_fdata_v_t &output, fdata_v_t *w1w3){

	s_fdata_v_t s_w1, s_w3, out;
	#pragma HLS STREAM variable=s_w1 depth=(MODEL_HIDDEN_DIM/SM_FL_ELEM)
	#pragma HLS STREAM variable=s_w3 depth=(MODEL_HIDDEN_DIM/SM_FL_ELEM)
	#pragma HLS BIND_STORAGE variable=s_w1 type=fifo impl=bram
	#pragma HLS BIND_STORAGE variable=s_w3 type=fifo impl=bram
	#pragma HLS DATAFLOW

	mm2s_input_data(s_w1, w1w3, MODEL_HIDDEN_DIM/SM_FL_ELEM, 0);
	mm2s_input_data(s_w3, w1w3, MODEL_HIDDEN_DIM/SM_FL_ELEM, 1);
	swiglu(output, s_w1, s_w3);
	return;
}