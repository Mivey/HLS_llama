#include "rmsnorm.h"
#include "../forward.h"
// #include "rmsnorm.h"


void rmsnorm(s_fdata_v_t &o, s_fdata_v_t &t, s_fdata_v_t &d, s_fdata_v_t &x, s_fdata_v_t &w){
  // #pragma HLS DATAFLOW
	constexpr int UF = 4;
  fdata_v_t arr[MODEL_ELEMENTS/SM_FL_ELEM] = {0};
#pragma HLS ARRAY_PARTITION variable=arr type=complete
  fdata_v_t ss = 0.0f;// = {0.0f}; // <----- added init value 0.0f 10/3 while working on MHA
  
  rms_mac_loop:
  for (int i = 0; i < (MODEL_ELEMENTS / SM_FL_ELEM); i++) {
    #pragma HLS PIPELINE II=1
    fdata_v_t tempval = x.read() + d.read();
    ss += tempval * tempval;
		t.write(tempval);
		arr[i] = tempval * w.read();
  }

  my_float_t fss = (ss.reduce_add() / MODEL_ELEMENTS + 1e-5);
	
  fss = 1.0f/hls::sqrtf(fss);

  // fdata_v_t tmp_o;
  data_out_loop:
  for (int i = 0 ; i < MODEL_ELEMENTS/SM_FL_ELEM; i++) {
    #pragma HLS PIPELINE II=1
    // fdata_v_t tmp_o = w.read() * arr[i] * fss;
    o.write(arr[i] * fss);
		// t.write(arr[i]);
  }
}


void rmsnorm_kernel(fdata_v_t *output, fdata_v_t *tokens_o, fdata_v_t *tokens_i, fdata_v_t *diff, fdata_v_t *weights, const int CURR_LAYER){

	constexpr int RMS_DEPTH = MODEL_ELEMENTS / SM_FL_ELEM;
	// constexpr int RMS_DEPTH = MODEL_ELEMENTS * MODEL_NUM_LAYERS / SM_FL_ELEM;

	#pragma HLS INTERFACE mode=m_axi port=tokens_i						bundle=rms_tok	depth=RMS_DEPTH			offset=slave max_read_burst_length=(4096/SM_DW * 8)
	#pragma HLS INTERFACE mode=m_axi port=tokens_o						bundle=rms_tok	depth=RMS_DEPTH			offset=slave max_write_burst_length=(4096/SM_DW * 8)
	#pragma HLS INTERFACE mode=m_axi port=output						bundle=rms_out_w	depth=RMS_DEPTH		offset=slave max_write_burst_length=(4096/SM_DW * 8)
	#pragma HLS INTERFACE mode=m_axi port=weights						bundle=rms_out_w 	depth=RMS_DEPTH		offset=slave max_read_burst_length=(4096/SM_DW * 8)
	#pragma HLS INTERFACE mode=m_axi port=diff							bundle=rms_out_w 	depth=RMS_DEPTH		offset=slave max_read_burst_length=(4096/SM_DW * 8)

	#pragma HLS INTERFACE mode=s_axilite port=tokens_i				bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=tokens_o				bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=output				bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=weights				bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=diff				bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=CURR_LAYER 		bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=return 				bundle=control
	#pragma HLS DATAFLOW
	s_fdata_v_t s_weights, s_tokens, s_tokens_out, s_diff_out, s_diff, s_res_rms;
	
	#pragma HLS STREAM variable=s_tokens depth=(MODEL_ELEMENTS / SM_FL_ELEM)
	// #pragma HLS STREAM variable=s_res_rms depth=(MODEL_ELEMENTS / SM_FL_ELEM)
	#pragma HLS STREAM variable=s_diff depth=(MODEL_ELEMENTS / SM_FL_ELEM)
	#pragma HLS STREAM variable=s_tokens_out depth=(MODEL_ELEMENTS / SM_FL_ELEM)
	#pragma HLS STREAM variable=s_diff_out depth=(MODEL_ELEMENTS / SM_FL_ELEM)
	#pragma HLS STREAM variable=s_weights depth=(MODEL_ELEMENTS / SM_FL_ELEM)
	
	#pragma HLS BIND_STORAGE variable=s_tokens type=fifo impl=bram
	// #pragma HLS BIND_STORAGE variable=s_diff_out type=fifo impl=bram
	// #pragma HLS BIND_STORAGE variable=s_diff type=fifo impl=bram
	// #pragma HLS BIND_STORAGE variable=s_res_rms type=fifo impl=bram
	#pragma HLS BIND_STORAGE variable=s_diff type=fifo impl=bram
	#pragma HLS BIND_STORAGE variable=s_tokens_out type=fifo impl=bram
	#pragma HLS BIND_STORAGE variable=s_diff_out type=fifo impl=bram
	#pragma HLS BIND_STORAGE variable=s_weights type=fifo impl=bram
	
	// tok_load_input(s_tokens, tokens);
	mm2s_input_data(s_tokens, tokens_i, MODEL_ELEMENTS/SM_FL_ELEM);
	mm2s_input_data(s_diff, diff, MODEL_ELEMENTS / SM_FL_ELEM);
	// tok_load_input(s_diff, diff);
	// resid_conn(s_res_rms, s_rescon_out, s_rescon, s_tokens);
	// store_output(rescon, s_rescon_out, MODEL_ELEMENTS);
	rms_load_input(s_weights, weights, CURR_LAYER);

	rmsnorm(s_tokens_out, s_diff_out, s_diff, s_tokens, s_weights);
	s2mm_output_data(output, s_tokens_out, MODEL_ELEMENTS / SM_FL_ELEM, 0);
	s2mm_output_data(tokens_o, s_diff_out, MODEL_ELEMENTS / SM_FL_ELEM, 0);
	// store_output(output, s_tokens_out, MODEL_ELEMENTS);
	return;
}