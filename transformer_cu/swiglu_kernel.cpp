#include "mha_forward.h"
#include "quantizer.h"
#include "swiglu.h"

const int Q_LOOP = MODEL_HIDDEN_DIM / MODEL_SCALING_FACTOR;
void swi_max_finder(hls::stream<my_float_t> &max_val, s_fdata_v_t &tokens_out, s_fdata_v_t &tokens_in){
	
	const my_float_t Q_MAX = 1.0f / 127.0f;
	const int cnt = MODEL_SCALING_FACTOR / SM_FL_ELEM;
	my_float_t c_val[MODEL_SCALING_FACTOR];
	#pragma HLS ARRAY_PARTITION variable=c_val dim=1 type=complete
	//here we store token_out and then assign token_out[i] to c_val[i * MAX_FL_ELEM + k] = hls::absf(token_out[i][k])
	
	for (int j = 0; j < Q_LOOP; j++) {
		
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

void swi_quant_out( hls::stream<my_float_t> &tok_sf_out, s_idata_v_t &tok_out, s_fdata_v_t &tokens_in, hls::stream<my_float_t> &max_val){
	
	const size_t TOK_COUNT = MODEL_SCALING_FACTOR / SM_FL_ELEM;
	my_float_t ds_arr[Q_LOOP]{};

	for (int i = 0;  i < Q_LOOP; i++) {
	
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
	}
	
	for (int i = 0;  i < MODEL_NUM_HEADS; i++) {
		#pragma HLS PIPELINE II=1
		tok_sf_out.write(ds_arr[i]);
	}
}


void swiglu_kernel(hls::stream<my_float_t> &tok_sf_out, s_idata_v_t &tok_out, fdata_v_t *w1w3){

	s_fdata_v_t s_w1, s_w3, out;
	hls::stream<my_float_t> max_val;
	s_fdata_v_t tokens_out;
	#pragma HLS STREAM variable=tokens_out depth = 16
	#pragma HLS STREAM variable=max_val depth = 16
	#pragma HLS STREAM variable=s_w1 depth=(MODEL_HIDDEN_DIM/SM_FL_ELEM)
	#pragma HLS STREAM variable=s_w3 depth=(MODEL_HIDDEN_DIM/SM_FL_ELEM)
	#pragma HLS BIND_STORAGE variable=s_w1 type=fifo impl=bram
	#pragma HLS BIND_STORAGE variable=s_w3 type=fifo impl=bram
	#pragma HLS DATAFLOW

	// mm2s_input_data(s_w1, w1w3, MODEL_HIDDEN_DIM/SM_FL_ELEM, 0);
	// mm2s_input_data(s_w3, w1w3, MODEL_HIDDEN_DIM/SM_FL_ELEM, 0, MODEL_TOKENS / (SM_FL_ELEM * 2));
	mm2s_input_data(s_w1, w1w3, MODEL_HIDDEN_DIM/SM_FL_ELEM, 0);
	mm2s_input_data(s_w3, w1w3, MODEL_HIDDEN_DIM/SM_FL_ELEM, 1);
	swiglu(out, s_w1, s_w3);
	quantizer_kernel(tok_sf_out, tok_out, out, MODEL_HIDDEN_DIM);
	// swi_max_finder(max_val, tokens_out, out);
	// swi_quant_out(tok_sf_out, tok_out, tokens_out, max_val);
	return;
}