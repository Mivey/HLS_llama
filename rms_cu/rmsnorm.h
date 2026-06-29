#ifndef MARK_RMS
#define MARK_RMS
#include "../forward.h"

// void rmsnorm_kernel(fdata_v_t *tokens, fdata_v_t *stokens, fdata_v_t *weights/*, const int CURR_LAYER*/);
// void rmsnorm_kernel(fdata_v_t *output, fdata_v_t *tokens, fdata_v_t *diff, fdata_v_t *weights, const int CURR_LAYER);
void rmsnorm_kernel(s_idata_v_t &s_w, hls::stream<my_float_t> &s_sf, fdata_v_t *diff, fdata_v_t *weights, const int CURR_LAYER, const int INIT, const int offset);


template<typename T, size_t N>
void max_finder(hls::stream<T> &max_val, hls::stream<hls::vector<T, N>> &tokens_out, hls::stream<hls::vector<T, N>> &tokens_in){
	
	const T Q_MAX = 1.0f / 127.0f;
	const int cnt = MODEL_SCALING_FACTOR / N;
	T c_val[MODEL_SCALING_FACTOR];
	#pragma HLS ARRAY_PARTITION variable=c_val dim=1 type=complete
	//here we store token_out and then assign token_out[i] to c_val[i * MAX_FL_ELEM + k] = hls::absf(token_out[i][k])
	
	
	mf_intake:
	for (int i = 0; i < cnt; i++) {
		#pragma HLS PIPELINE II=1
		hls::vector<T, N> val = tokens_in.read();
		tokens_out.write(val);
		for (int k = 0; k < N; k++) {
			c_val[i * N + k] = hls::absf(val[k]);
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


template<typename T, size_t N>
void quant_out( hls::stream<T> &tok_sf_out, s_idata_v_t &tok_out, hls::stream<hls::vector<T, N>> &tokens_in, hls::stream<my_float_t> &max_val){
	
	const size_t TOK_COUNT = MODEL_SCALING_FACTOR / N;
	T dscale = max_val.read(); 
	T scale = hls::recipf(dscale);//Q_MAX / max_val;
	idata_v_t quant_tmp; // not an array anymore
	// fdata_v_t tok_arr[TOK_COUNT];
	
	create_q_val:
	for (size_t j = 0; j < TOK_COUNT; j++) {
		// #pragma HLS PIPELINE
		hls::vector<T, N> proc_tok = tokens_in.read();
		
		create_q_val_ppl:	
		for (size_t k = 0; k < N; k++) {
			#pragma HLS PIPELINE
			quant_tmp[j * N + k] = (my_quant_data_t) hls::roundf(proc_tok[k] * scale);
		}
	}
}

template<typename T, size_t N>
void quant_out( hls::stream<T> &tok_sf_out, s_idata_v_t &tok_out, hls::stream<hls::vector<T, N>> &tokens_in, hls::stream<my_float_t> &max_val, hls::stream<my_float_t> &c_rms){
	
	const size_t TOK_COUNT = MODEL_SCALING_FACTOR / N;
	my_float_t c_r = c_rms.read();
	T dscale = max_val.read(); 
	T scale = hls::recipf(dscale);//Q_MAX / max_val;
	idata_v_t quant_tmp; // not an array anymore
	// fdata_v_t tok_arr[TOK_COUNT];
	
	create_q_val:
	for (size_t j = 0; j < TOK_COUNT; j++) {
		// #pragma HLS PIPELINE
		hls::vector<T, N> proc_tok = tokens_in.read();
		
		create_q_val_ppl:	
		for (size_t k = 0; k < N; k++) {
			#pragma HLS PIPELINE
			quant_tmp[j * N + k] = (my_quant_data_t) hls::roundf(proc_tok[k] * scale);
		}
	}
	tok_out.write(quant_tmp);
	tok_sf_out.write(dscale * c_r);
}


	
#endif
