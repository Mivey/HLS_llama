#include "../forward.h"

void swiglu_kernel(fdata_v_t *output, fdata_v_t *w1, fdata_v_t *w3){

	#pragma HLS INTERFACE mode=m_axi port=w1					bundle=sg_in_out		depth=MODEL_HIDDEN_DIM	offset=slave max_read_burst_length=(4096/SM_DW * 8)
	#pragma HLS INTERFACE mode=m_axi port=w3					bundle=sg_in_out		depth=MODEL_HIDDEN_DIM	offset=slave max_read_burst_length=(4096/SM_DW * 8)
	#pragma HLS INTERFACE mode=m_axi port=output			bundle=sg_in_out		depth=MODEL_HIDDEN_DIM	offset=slave max_write_burst_length=(4096/SM_DW * 8)
	#pragma HLS INTERFACE mode=s_axilite port=w1			bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=w3			bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=output	bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=return	bundle=control
	s_fdata_v_t s_w1, s_w3, out;
	#pragma HLS STREAM variable=s_w1 depth=(MODEL_HIDDEN_DIM/SM_FL_ELEM)
	#pragma HLS STREAM variable=s_w3 depth=(MODEL_HIDDEN_DIM/SM_FL_ELEM)
	#pragma HLS BIND_STORAGE variable=s_w1 type=fifo impl=bram
	#pragma HLS BIND_STORAGE variable=s_w3 type=fifo impl=bram
	#pragma HLS DATAFLOW

	// w1w3_load_input(w1, w3, in);
	
	tok_load_input(s_w1, w1, MODEL_HIDDEN_DIM);
	tok_load_input(s_w3, w3, MODEL_HIDDEN_DIM);
	swiglu(out, s_w1, s_w3);
	store_output(output, out, MODEL_HIDDEN_DIM);
	return;
}