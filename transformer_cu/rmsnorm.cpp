#include "rmsnorm.h"
#include "mha_forward.h"


void rmsnorm(s_fdata_v_t &o, s_fdata_v_t &d, fdata_v_t x[MODEL_ELEMENTS/SM_FL_ELEM], s_fdata_v_t &w, const int INIT){
  // #pragma HLS DATAFLOW
	constexpr int UF = 4;
  fdata_v_t arr[MODEL_ELEMENTS/SM_FL_ELEM] = {0};
	// #pragma HLS ARRAY_PARTITION variable=arr type=complete
	const int acc_lag = 8;
  my_float_t ss[acc_lag] = {0.0f};// = {0.0f}; // <----- added init value 0.0f 10/3 while working on MHA
  
  rms_mac_loop:
  for (int i = 0; i < (MODEL_ELEMENTS / SM_FL_ELEM); i++) {
    #pragma HLS PIPELINE

		if (INIT == 0) {
			x[i] = d.read();
		}else {
			x[i] += d.read();
		}
    fdata_v_t tss = x[i] * x[i];
		ss[i % acc_lag] += tss.reduce_add();
		arr[i] = x[i] * w.read();
  }

	for (int stride = (acc_lag >>1); stride > 0; stride >>=1) {
		#pragma HLS UNROLL
		for (int i = 0; i < stride; i++) {
			#pragma HLS UNROLL
			ss[i] += ss[stride + i];
		}
	}
  my_float_t fss = (ss[0] / MODEL_ELEMENTS + 1e-5);
	
  fss = 1.0f/hls::sqrtf(fss);

  data_out_loop:
  for (int i = 0 ; i < MODEL_ELEMENTS/SM_FL_ELEM; i++) {
    #pragma HLS PIPELINE II=1
    o.write(arr[i] * fss);
  }
}


void rmsnorm_kernel(s_fdata_v_t &s_tokens_out, fdata_v_t internal_tokens[MODEL_ELEMENTS/SM_FL_ELEM], fdata_v_t *diff, fdata_v_t *weights, const int CURR_LAYER, const int INIT){

	#pragma HLS DATAFLOW
	s_fdata_v_t s_weights, s_tokens, s_diff_out, s_diff, s_res_rms;
	
	#pragma HLS STREAM variable=s_tokens depth=(MODEL_ELEMENTS / SM_FL_ELEM)
	#pragma HLS STREAM variable=s_diff depth=(MODEL_ELEMENTS / SM_FL_ELEM)
	#pragma HLS STREAM variable=s_tokens_out depth=(MODEL_ELEMENTS / SM_FL_ELEM)
	#pragma HLS STREAM variable=s_diff_out depth=(MODEL_ELEMENTS / SM_FL_ELEM)
	#pragma HLS STREAM variable=s_weights depth=(MODEL_ELEMENTS / SM_FL_ELEM)
	
	#pragma HLS BIND_STORAGE variable=s_tokens type=fifo impl=bram
	#pragma HLS BIND_STORAGE variable=s_diff type=fifo impl=bram
	#pragma HLS BIND_STORAGE variable=s_tokens_out type=fifo impl=bram
	#pragma HLS BIND_STORAGE variable=s_diff_out type=fifo impl=bram
	#pragma HLS BIND_STORAGE variable=s_weights type=fifo impl=bram
	
	mm2s_input_data(s_diff, diff, MODEL_ELEMENTS / SM_FL_ELEM);
	// rms_load_input(s_weights, weights, CURR_LAYER);
	mm2s_input_data(s_weights, weights, MODEL_ELEMENTS/SM_FL_ELEM, CURR_LAYER);

	rmsnorm(s_tokens_out, s_diff_out, internal_tokens, s_weights, INIT);
	return;
}