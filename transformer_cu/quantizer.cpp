
#include "quantizer.h"
#include "mha_forward.h"


void abs_intake(s_fdata_v_t &tokens_out, s_fdata_v_t &abs_tokens, s_fdata_v_t &tokens_in){
	
	const size_t TOK_COUNT = MODEL_SCALING_FACTOR / SM_FL_ELEM;
	my_float_t max_val = 0.0f;
	
	group_scaling:
	for (size_t j = 0; j < TOK_COUNT; j++) {
		#pragma HLS PIPELINE II=1
		
		fdata_v_t val = tokens_in.read();
		tokens_out.write(val);
		
		fdata_v_t c_val;
		for (int k = 0; k < SM_FL_ELEM; k++) {
			c_val[k] = hls::absf(val[k]);
		}		
		abs_tokens.write(c_val);
	}
	
}

void max_finder(hls::stream<my_float_t> &max_val, s_fdata_v_t & abs_tokens){
	
	const my_float_t Q_MAX = 1.0f / 127.0f;
	const int cnt = MODEL_SCALING_FACTOR / SM_FL_ELEM;
	my_float_t c_val[MODEL_SCALING_FACTOR];
	#pragma HLS ARRAY_PARTITION variable=c_val dim=1 type=complete
	
	mf_intake:
	for (int i = 0; i < cnt; i++) {
		#pragma HLS PIPELINE II=1
		fdata_v_t val = abs_tokens.read();
		for (int k = 0; k < SM_FL_ELEM; k++) {
			c_val[i * SM_FL_ELEM + k] = val[k];
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


void quant_out( hls::stream<my_float_t> &tok_sf_out, s_idata_v_t &tok_out, s_fdata_v_t &tokens_in, hls::stream<my_float_t> &max_val){
	
	const size_t TOK_COUNT = MODEL_SCALING_FACTOR / SM_FL_ELEM;
	my_float_t dscale = max_val.read(); 
	my_float_t scale = hls::recipf(dscale);//Q_MAX / max_val;
	idata_v_t quant_tmp; // not an array anymore
	// fdata_v_t tok_arr[TOK_COUNT];
	
	create_quant_val:
	for (size_t j = 0; j < TOK_COUNT; j++) {
		#pragma HLS PIPELINE
		fdata_v_t proc_tok = tokens_in.read();
		
		create_quant_val_pipeline_loop:	
		for (size_t k = 0; k < SM_FL_ELEM; k++) {
			#pragma HLS UNROLL
			quant_tmp[j * SM_FL_ELEM + k] = (my_quant_data_t) hls::roundf(proc_tok[k] * scale);
		}
	}
	
	tok_out.write(quant_tmp);
	tok_sf_out.write(dscale);
}

void quantizer_kernel(hls::stream<my_float_t>  &tok_sf_out, s_idata_v_t &tok_out, s_fdata_v_t &tokens, const int N_DIM){
	
	const size_t SF_COUNT = N_DIM / MODEL_SCALING_FACTOR;
	const size_t TOK_COUNT = MODEL_SCALING_FACTOR / SM_FL_ELEM;
	// #pragma HLS STREAM variable=tok_out depth=64
	// #pragma HLS STREAM variable=tok_sf_out depth=64
	for (int i = 0; i < SF_COUNT; i++) {
		#pragma HLS LOOP_TRIPCOUNT max=MODEL_HIDDEN_DIM / MODEL_SCALING_FACTOR min=MODEL_ELEMENTS / MODEL_SCALING_FACTOR
		#pragma HLS DATAFLOW
		hls::stream<my_float_t> max_val;
		s_fdata_v_t tokens_out, abs_tokens;
		#pragma HLS STREAM variable=tokens_out depth=64
		#pragma HLS STREAM variable=max_val depth=TOK_COUNT
		#pragma HLS STREAM variable=abs_tokens depth=64
		
		abs_intake(tokens_out, abs_tokens, tokens);
		max_finder(max_val, abs_tokens);
		quant_out(tok_sf_out, tok_out, tokens_out, max_val);
	}
}
