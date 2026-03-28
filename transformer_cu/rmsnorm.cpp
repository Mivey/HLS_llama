#include "rmsnorm.h"
#include "mha_forward.h"

#ifdef  __DEBUG__
	#include "hls_print.h"
#endif


void rmsnorm(s_fdata_v_t &o, s_fdata_v_t &d, fdata_v_t x[MODEL_ELEMENTS/SM_FL_ELEM], s_fdata_v_t &w, const int INIT){
  // #pragma HLS DATAFLOW
	constexpr int UF = 4;
  fdata_v_t arr[MODEL_ELEMENTS/SM_FL_ELEM] = {0};
	// #pragma HLS ARRAY_PARTITION variable=arr type=complete
	const int acc_lag = 16;
  my_float_t ss[acc_lag] = {0.0f};
  
  rms_mac_loop:
  for (int i = 0; i < (MODEL_ELEMENTS / SM_FL_ELEM); i++) {
    #pragma HLS PIPELINE

		if (INIT == 1) {
			x[i] = d.read();
		}else {
			x[i] += d.read();
		}
    fdata_v_t tss = x[i] * x[i];
		ss[i % acc_lag] += tss.reduce_add();
		arr[i] = x[i];
  }
	my_float_t ftss = 0.0f;
	for (int i = 0; i < acc_lag; i++) {
		#pragma HLS UNROLL
		ftss += ss[i];
	}

  my_float_t fss = (ftss / MODEL_ELEMENTS + 1e-5);
	
  fss = 1.0f/hls::sqrtf(fss);

  data_out_loop:
  for (int i = 0 ; i < MODEL_ELEMENTS/SM_FL_ELEM; i++) {
    #pragma HLS PIPELINE II=1
		fdata_v_t tw = w.read();
    o.write(arr[i] * fss * tw);
  }
}


void rmsnorm_kernel(s_fdata_v_t &s_tokens_out, fdata_v_t *diff, fdata_v_t *weights, const int CURR_LAYER, const int INIT, const int offset){

	#pragma HLS DATAFLOW
	s_fdata_v_t s_weights, s_tokens, s_diff;
	const int ratio = MODEL_ELEMENTS / SM_FL_ELEM;
	
	#pragma HLS STREAM variable=s_tokens depth=ratio
	#pragma HLS STREAM variable=s_diff depth=ratio
	#pragma HLS STREAM variable=s_tokens_out depth=ratio
	#pragma HLS STREAM variable=s_weights depth=ratio
	
	#pragma HLS BIND_STORAGE variable=s_tokens type=fifo impl=bram
	#pragma HLS BIND_STORAGE variable=s_diff type=fifo impl=bram
	#pragma HLS BIND_STORAGE variable=s_tokens_out type=fifo impl=bram
	#pragma HLS BIND_STORAGE variable=s_weights type=fifo impl=bram
	
	static fdata_v_t internal_tokens[ratio];
	#pragma HLS BIND_STORAGE variable=internal_tokens type=ram_t2p impl=bram
	#pragma HLS ARRAY_PARTITION variable=internal_tokens dim=1 type=cyclic factor=2

	mm2s_input_data(s_diff, diff, ratio);
	// rms_load_input(s_weights, weights, CURR_LAYER);
	mm2s_input_data(s_weights, weights, ratio, CURR_LAYER, offset);

	rmsnorm(s_tokens_out, s_diff, internal_tokens, s_weights, INIT);
		
	return;
}