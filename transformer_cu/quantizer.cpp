
#include "quantizer.h"

void quantizer_kernel(s_fdata_v_t &tok_sf_out, s_idata_v_t &tok_out, s_fdata_v_t &tokens, const int N_DIM){
	/********************************* WILL NEED TO FIX CREATE_QUANT_VAL IF MAX_FL_ELEM != 16 *******************************/
	const size_t SF_COUNT = N_DIM / MODEL_SCALING_FACTOR;
	const size_t TOK_COUNT = MODEL_SCALING_FACTOR / SM_FL_ELEM;
	const my_float_t Q_MAX = 127.0f;
	
	fdata_v_t tok_arr[TOK_COUNT];
	#pragma HLS ARRAY_PARTITION variable=tok_arr dim=1 type=complete 
	fdata_v_t tmp_sf_out;
	int wout = 0;
	quantizer_main:
	for (size_t i = 0; i < SF_COUNT; i++) {
		
		#pragma HLS LOOP_TRIPCOUNT max=MODEL_HIDDEN_DIM / MODEL_SCALING_FACTOR
		my_float_t max_val = 0.0f;
		my_float_t c_val[MODEL_SCALING_FACTOR] = {0.0f};
			#pragma HLS ARRAY_PARTITION variable=c_val dim=1 type=complete 
		group_scaling:
		for (size_t j = 0; j < TOK_COUNT; j++) {
			#pragma HLS PIPELINE II=1
			
			fdata_v_t val = tokens.read();
			tok_arr[j] = val;
			my_float_t a_val[SM_FL_ELEM];
			#pragma HLS ARRAY_PARTITION variable=a_val dim=1 type=complete 

			for (int k = 0; k < SM_FL_ELEM; k++) {
				c_val[j * SM_FL_ELEM + k] = hls::absf(val[k]);
			}		
		}

		for (int stride = (MODEL_SCALING_FACTOR >> 1); stride > 0; stride >>=1) {
			#pragma HLS UNROLL
			for (int j = 0; j < stride; j++) {
				#pragma HLS UNROLL
				c_val[j] = (c_val[j] < c_val[j + stride]) ? c_val[j + stride] : c_val[j];
			}
		}
		max_val = c_val[0];
		
		my_float_t dscale = max_val * ( 1.0f / Q_MAX); 
		my_float_t scale = hls::recipf(dscale);//Q_MAX / max_val;
		idata_v_t quant_tmp; // not an array anymore
		
		int n = 0;
		create_quant_val:
		for (size_t j = 0; j < TOK_COUNT; j++) {
			fdata_v_t proc_tok = tok_arr[j];
			
			create_quant_val_pipeline_loop:	
			for (size_t k = 0; k < SM_FL_ELEM; k++) {
			#pragma HLS PIPELINE
			#pragma HLS UNROLL factor=2
				quant_tmp[n + k] = (my_quant_data_t) hls::roundf(proc_tok[k] * scale);
			}
			n += SM_FL_ELEM;
		}
		
		tok_out.write(quant_tmp);
			
		tmp_sf_out[wout] = dscale;
		wout++;
		
		if (wout == SM_FL_ELEM) {
			tok_sf_out.write(tmp_sf_out);
			wout = 0;
		}
	}
}