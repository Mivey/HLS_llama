#include "rmsnorm.h"
#include "mha_forward.h"
#include <cmath>

void rms_mm2s_data(s_fdata_v_t &out, fdata_v_t *in, const int cnt){
	// #pragma HLS PIPELINE off
	mm2s_input_data(out, in, cnt/2, 0, 0 );
	mm2s_input_data(out, in, cnt/2, 0, (INTERNAL_DATA_SIZE / (SM_FL_ELEM * 2)));
}

void rmsnorm(s_fdata_v_t &o, s_fdata_v_t &d, fdata_v_t *x, s_fdata_v_t &w, const bool INIT){
  
  fdata_v_t arr[MODEL_ELEMENTS/SM_FL_ELEM];
	const int acc_lag = 16;
  float_t ss[acc_lag] = {0.0f}; 
  
  rms_mac:
  for (int i = 0; i < (MODEL_ELEMENTS / SM_FL_ELEM); i++) {
    #pragma HLS PIPELINE

		// if (INIT) { 	x[i] = d.read();	}
		// else { 				x[i] += d.read(); }
		fdata_v_t tmp_d = d.read();
		fdata_v_t tmp_x = x[i];
		fdata_v_t tmp_dx;
		float_t ss_ra = 0;
		for (int j = 0 ; j < SM_FL_ELEM; j++) {
			#pragma HLS UNROLL
			float_t foo = tmp_d[j] + tmp_x[j];
			float_t bar = foo * foo;
			tmp_dx[j] = foo;
			ss_ra += bar;
		}
		
		x[i] = tmp_dx;
    // fdata_v_t tss = tmp_tss;
		
		ss[i % acc_lag] += ss_ra;
		// float_t foo;
		// for (int j = 0; j < SM_FL_ELEM; j++) {
		// 	#pragma hls UNROLL
		// 	float_t bar = tss[j];
		// 	foo += bar;
		// }
		arr[i] = x[i];
  }
	float_t ftss = 0.0f;
	
	rms_sum:
	for (int i = 0; i < acc_lag; i++) {
		#pragma HLS UNROLL
		ftss += ss[i];
	}

  float_t fss = (ftss / MODEL_ELEMENTS + 1e-5);
  fss = 1.0f/hls::sqrtf(fss);

  data_out:
  for (int i = 0 ; i < MODEL_ELEMENTS/SM_FL_ELEM; i++) {
    #pragma HLS PIPELINE II=1
		fdata_v_t tw = w.read();
		fdata_v_t rw = arr[i];
		fdata_v_t o_data;
		for (int j = 0; j < SM_FL_ELEM; j++) {
			#pragma HLS UNROLL
			float_t foo = rw[j] * tw[j] * fss;
			o_data[j] = foo;
		}
    o.write(o_data);//arr[i] * fss * tw);
  }
}


void rmsnorm_kernel(s_fdata_v_t &s_tokens_out, fdata_v_t *diff, fdata_v_t *weights, fdata_v_t *res_con, const int CURR_LAYER, const bool INIT, const int offset){

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
	
	rms_mm2s_data(s_diff, diff, ratio);
	mm2s_input_data(s_weights, weights, ratio, CURR_LAYER, offset);
	rmsnorm(s_tokens_out, s_diff, res_con, s_weights, INIT);
		
	return;
}