#include "rmsnorm.h"
#include "../forward.h"
#include <cstdint>
#include <hls_math.h>

void new_rmsnorm(hls::stream<my_float_t> &c_rms, s_fdata_v_t &o, s_fdata_v_t &d, fdata_v_t x[MODEL_ELEMENTS/SM_FL_ELEM], s_fdata_v_t &w, const int INIT){
  // #pragma HLS DATAFLOW
	constexpr int UF = 4;
  fdata_v_t arr[MODEL_ELEMENTS/SM_FL_ELEM] = {0};
	// #pragma HLS ARRAY_PARTITION variable=arr type=complete
	const int acc_lag = 16;
  my_float_t ss[acc_lag] = {0.0f};
  
  rms_mac:
  for (int i = 0; i < (MODEL_ELEMENTS / SM_FL_ELEM); i++) {
    #pragma HLS PIPELINE II=1
		fdata_v_t tmp;

		if (INIT == 1) {
			tmp = d.read();
		}else {
			tmp = x[i] + d.read();
		}
		
    fdata_v_t tss = tmp * tmp;
		fdata_v_t wss = tmp * w.read();
		arr[i] = tmp;
		
		o.write(wss);
		
		ss[i % acc_lag] += tss.reduce_add();
  }
	my_float_t ftss = 0.0f;
	for (int i = 0; i < acc_lag; i++) {
		#pragma HLS UNROLL
		ftss += ss[i];
	}

  my_float_t fss = (ftss / MODEL_ELEMENTS + 1e-5);	
  c_rms.write(1.0f/hls::sqrtf(fss));

	array_write:
	for (int i = 0; i < (MODEL_ELEMENTS / SM_FL_ELEM); i++) {
		#pragma HLS PIPELINE II=1
		x[i] = arr[i];
	}
}

void rms_max_finder(hls::stream<my_float_t> &max_val, s_fdata_v_t &tokens_out, s_fdata_v_t &tokens_in){
	
	const my_float_t Q_MAX = 1.0f / 127.0f;
	const int cnt = MODEL_SCALING_FACTOR / SM_FL_ELEM;
	my_float_t c_val[MODEL_SCALING_FACTOR];
	#pragma HLS ARRAY_PARTITION variable=c_val dim=1 type=complete
	//here we store token_out and then assign token_out[i] to c_val[i * MAX_FL_ELEM + k] = hls::absf(token_out[i][k])
	
	for (int j = 0; j < MODEL_NUM_HEADS; j++) {
		
		mf_intake:
		for (int i = 0; i < cnt; i++) {
			#pragma HLS PIPELINE II=1
			fdata_v_t val = tokens_in.read();
			tokens_out.write(val);
			for (int k = 0; k < SM_FL_ELEM; k++) {
				c_val[i * SM_FL_ELEM + k] = hls::absf(val[k]);
			}
		}

		for (int stride = (MODEL_SCALING_FACTOR>>1); stride > 0; stride >>=1) {
			#pragma HLS UNROLL
			for (int i = 0; i < stride; i++) {
				#pragma HLS UNROLL
				c_val[i] = (c_val[i]  > c_val[i + stride] ) ? c_val[i] : c_val[i + stride];
			}
		}	
		max_val.write(c_val[0] * Q_MAX);
	}
}

void rms_quant_out( hls::stream<my_float_t> &tok_sf_out, s_idata_v_t &tok_out, s_fdata_v_t &tokens_in, hls::stream<my_float_t> &max_val, hls::stream<my_float_t> &c_rms){
	
	const size_t TOK_COUNT = MODEL_SCALING_FACTOR / SM_FL_ELEM;
		my_float_t ds_arr[MODEL_NUM_HEADS]{};

	for (int i = 0;  i < MODEL_NUM_HEADS; i++) {
	
		my_float_t dscale = max_val.read(); 
		my_float_t scale = hls::recipf(dscale);//Q_MAX / max_val;
		idata_v_t quant_tmp; // not an array anymore
		
		create_q_val:
		for (size_t j = 0; j < TOK_COUNT; j++) {
			#pragma HLS PIPELINE
			fdata_v_t proc_tok = tokens_in.read();
			
			create_q_val_ppl:	
			for (size_t k = 0; k < SM_FL_ELEM; k++) {
				#pragma HLS UNROLL
				quant_tmp[j * SM_FL_ELEM + k] = (my_quant_data_t) hls::roundf(proc_tok[k] * scale);
			}
		}
		
		tok_out.write(quant_tmp);
		ds_arr[i] = dscale;
		// tok_sf_out.write(dscale * c_r);
	}
	my_float_t c_r = c_rms.read();
	for (int i = 0;  i < MODEL_NUM_HEADS; i++) {
		#pragma HLS PIPELINE II=1
		tok_sf_out.write(ds_arr[i] * c_r);
	}
}


// void rmsnorm_kernel(fdata_v_t *output, fdata_v_t *tokens_o, fdata_v_t *tokens_i, fdata_v_t *diff, fdata_v_t *weights, const int CURR_LAYER){
void rmsnorm_kernel(s_idata_v_t &s_w, hls::stream<my_float_t> &s_sf, fdata_v_t *diff, fdata_v_t *weights, const int CURR_LAYER, const int INIT, const int offset){

	constexpr int RMS_DEPTH = MODEL_ELEMENTS / SM_FL_ELEM;
	
	#pragma HLS INTERFACE mode=m_axi port=weights						bundle=rms_out_w 	depth=RMS_DEPTH		offset=slave max_read_burst_length=(4096/SM_DW * 8)
	#pragma HLS INTERFACE mode=m_axi port=diff							bundle=rms_out_w 	depth=RMS_DEPTH		offset=slave max_read_burst_length=(4096/SM_DW * 8)

	#pragma HLS INTERFACE mode=s_axilite port=weights			bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=diff				bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=CURR_LAYER 	bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=INIT				bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=offset 			bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=return 			bundle=control
	#pragma HLS DATAFLOW
	s_fdata_v_t s_weights, s_tokens, s_diff, s_tokens_out, s_abs_out;
	hls::stream<my_float_t> c_rms;
	hls::stream<my_float_t> q_sf;
	hls::stream<my_float_t> max_val;
	const int ratio = MODEL_ELEMENTS / SM_FL_ELEM;
	
	#pragma HLS STREAM variable=s_tokens depth=ratio
	#pragma HLS STREAM variable=q_sf depth=MODEL_ELEMENTS/MODEL_SCALING_FACTOR
	#pragma HLS STREAM variable=s_diff depth=ratio
	#pragma HLS STREAM variable=s_tokens_out depth=ratio
	#pragma HLS STREAM variable=s_abs_out depth=ratio
	#pragma HLS STREAM variable=s_weights depth=ratio/4
	
	#pragma HLS BIND_STORAGE variable=s_tokens type=fifo impl=bram
	#pragma HLS BIND_STORAGE variable=s_diff type=fifo impl=bram
	#pragma HLS BIND_STORAGE variable=s_tokens_out type=fifo impl=bram
	#pragma HLS BIND_STORAGE variable=s_weights type=fifo impl=bram
	
	static fdata_v_t internal_tokens[ratio];
	#pragma HLS BIND_STORAGE variable=internal_tokens type=ram_t2p impl=bram
	#pragma HLS ARRAY_PARTITION variable=internal_tokens dim=1 type=cyclic factor=2

	// rms_mm2s_data(s_diff, diff, ratio);
	mm2s_input_data(s_diff, diff, ratio);
	mm2s_input_data(s_weights, weights, ratio, CURR_LAYER, offset);
	new_rmsnorm(c_rms, s_tokens_out, s_diff, internal_tokens, s_weights, INIT);
	rms_max_finder(max_val, s_abs_out, s_tokens_out);
	rms_quant_out(s_sf, s_w, s_abs_out, max_val, c_rms);
	return;
}