
#include "matmult.h"
#include <cstddef>
#include <cstdint>
// #include "forward.h"
#include "hls_task.h"

constexpr size_t TOK_QUANT_MAX =  (MODEL_HIDDEN_DIM / MAX_QUANT_ELEM);
constexpr size_t TOK_SF_MAX = (MODEL_HIDDEN_DIM / (MODEL_SCALING_FACTOR * SM_FL_ELEM));
constexpr int UF = 1;

void quantizer_kernel(s_fdata_v_t &tok_sf_out, s_idata_v_t &tok_out, s_fdata_v_t &tokens, const int N_DIM){
	
	const size_t SF_COUNT = N_DIM / MODEL_SCALING_FACTOR;
	const size_t TOK_COUNT = MODEL_SCALING_FACTOR / SM_FL_ELEM;
	const my_float_t Q_MAX = 127.0f;
	
	fdata_v_t tok_arr[TOK_COUNT];
	#pragma HLS ARRAY_PARTITION variable=tok_arr dim=1 type=complete 
	fdata_v_t tmp_sf_out;
	
	quantizer_main_loop:
	for (size_t i = 0; i < SF_COUNT; i++) {
		#pragma HLS LOOP_TRIPCOUNT max=MODEL_HIDDEN_DIM / MODEL_SCALING_FACTOR
		my_float_t max_val = 0.0f;
		
		group_scaling_loop:
		for (size_t j = 0; j < TOK_COUNT; j++) {
			// #pragma HLS PIPELINE II=1
			
			fdata_v_t val = tokens.read();
			tok_arr[j] = val;

			quantizer_abs_val_loop:
			for (size_t k = 0; k < SM_FL_ELEM; k++) {	
				#pragma HLS PIPELINE
				my_float_t a_val = hls::absf(val[k]);
				if (max_val < a_val) {max_val = a_val; }
			}
		}
		
		my_float_t dscale = max_val / Q_MAX; 
		my_float_t scale = Q_MAX / max_val;
		idata_v_t quant_tmp_arr[MODEL_SCALING_FACTOR / MAX_QUANT_ELEM]; // not an array anymore
	#pragma HLS ARRAY_PARTITION variable=quant_tmp_arr dim=1 type=complete 
		
		create_quant_val_loop:
		for (size_t j = 0; j < TOK_COUNT; j++) {
			fdata_v_t proc_tok = tok_arr[j];// * scale;
			int n = (SM_FL_ELEM * j) % MAX_QUANT_ELEM; 
			int t = (SM_FL_ELEM * j ) / MAX_QUANT_ELEM;
				
			create_quant_val_pipeline_loop:	
			for (size_t k = 0; k < SM_FL_ELEM; k++) {
			#pragma HLS PIPELINE
			#pragma HLS UNROLL factor=2
				quant_tmp_arr[t][n + k] = (my_quant_data_t) hls::roundf(proc_tok[k] * scale);
			}
		}
		for (size_t p = 0; p < (MODEL_SCALING_FACTOR / MAX_QUANT_ELEM); p++) {
			#pragma HLS UNROLL
			tok_out.write(quant_tmp_arr[p]);
		}
			
		tmp_sf_out[i % SM_FL_ELEM] = dscale;
		
		if ((i % SM_FL_ELEM) == (SM_FL_ELEM - 1)) {
			tok_sf_out.write(tmp_sf_out);
		}
	}
}

void alt_mat_mult_main(hls::stream<my_float_t> &out, s_idata_v_t &w, s_fdata_v_t &w_sf, \
											s_idata_v_t &tok, s_fdata_v_t &tok_sf, const int N_DIM, const int M_DIM){

	const int sfCount = N_DIM / (SM_FL_ELEM * MODEL_SCALING_FACTOR);
	const int TOK_ARR_SIZE = N_DIM / MAX_QUANT_ELEM;
	const int SUM_FACTOR = MODEL_SCALING_FACTOR / MAX_QUANT_ELEM;
	// const int SF_2_Q_RATIO = MODEL_SCALING_FACTOR / MAX_QUANT_ELEM;

	//for now, assume idvt is 512 and only 512. 256 and 128 would require amm_calc to have 
	// another factor that handles 
	
	fdata_v_t arr_sf[TOK_SF_MAX];
	idata_v_t arr[TOK_QUANT_MAX];
  #pragma HLS BIND_STORAGE variable=arr_sf type=ram_2p impl=bram
  // #pragma HLS BIND_STORAGE variable=arr impl=srl

	amm_tok_sf:
	for (size_t i = 0; i < sfCount; i++){ // vCount here is 1/4 vCount in send_wtok!!
		#pragma HLS PIPELINE II=1
  	#pragma HLS LOOP_TRIPCOUNT max = TOK_SF_MAX min=MODEL_ELEMENTS/(MODEL_SCALING_FACTOR * SM_FL_ELEM )  
		arr_sf[i] = tok_sf.read();
		for (size_t j = 0; j < ( SM_FL_ELEM); j++) {
			arr[i * (SM_FL_ELEM) + j] = tok.read();
		}
	}


	// amm_tok:
	// for(size_t i = 0; i < (TOK_ARR_SIZE); i++){
	// 	#pragma HLS PIPELINE II=1
	// 	#pragma HLS LOOP_TRIPCOUNT max = TOK_QUANT_MAX min=MODEL_ELEMENTS/(MAX_QUANT_ELEM)  
	// 	arr[i] = tok.read();
	// }	
	
	amm_calc:
	for (size_t i = 0; i < M_DIM; i++) {
		#pragma HLS LOOP_TRIPCOUNT max=MODEL_TOKENS min=MODEL_ELEMENTS
		//output M_DIM float elements
		my_float_t sum_out = 0;
		for (size_t j = 0 ; j < sfCount; j++) {
  	#pragma HLS LOOP_TRIPCOUNT max = TOK_SF_MAX min=MODEL_ELEMENTS/(MODEL_SCALING_FACTOR * SM_FL_ELEM )  
			//read the next set of scaling factors
			fdata_v_t vec_tok_sf = arr_sf[j];
			fdata_v_t vec_w_sf = w_sf.read();
			// my_float_t tmp_sum = 0.0f;
			for (size_t k = 0; k < SM_FL_ELEM; k++) {
				//do our calculations
				#pragma HLS PIPELINE 
				
				my_float_t cur_tok_sf = vec_tok_sf[k];
				my_float_t cur_w_sf = vec_w_sf[k];
				
				//read the next set of weights
				idata_v_t curr_tok;
				idata_v_t curr_w;
				
				int32_t prod;
				// int32_t comb_prod = 0;
				
				curr_w = w.read();
				curr_tok = arr[j * SM_FL_ELEM + k];
				prod = 0;
				
				for (size_t m = 0; m < MAX_QUANT_ELEM; m++) {
					prod += (int32_t) curr_w[m] * curr_tok[m];
				}
				sum_out += (float)prod * cur_tok_sf * cur_w_sf;
			}
		}
		out.write(sum_out);
	}
	
}


void double_matmult_kernel(fdata_v_t *out, fdata_v_t *fl_tok, fdata_v_t *w_sf, idata_v_t *w, const int N_DIM, const int M_DIM, const int CURR_LAYER, const int W_Off){

	constexpr int HD_QUANT_DEPTH = MODEL_HIDDEN_DIM * MODEL_ELEMENTS * MODEL_NUM_LAYERS  / MAX_QUANT_ELEM;
	constexpr int HD_SF_DEPTH = MODEL_HIDDEN_DIM * MODEL_ELEMENTS * MODEL_NUM_LAYERS / (MODEL_SCALING_FACTOR * SM_FL_ELEM);
	constexpr int TOK_DEPTH = MODEL_HIDDEN_DIM / SM_FL_ELEM;
	constexpr int TOK_OUT_DEPTH = MODEL_HIDDEN_DIM / SM_FL_ELEM;

	
	#pragma HLS INTERFACE mode=m_axi port=out 		bundle=D_TOK_W_SF 	depth=TOK_OUT_DEPTH 	offset=slave max_write_burst_length=128	
	#pragma HLS INTERFACE mode=m_axi port=fl_tok 	bundle=D_TOK_W_SF 	depth=TOK_DEPTH 			offset=slave max_read_burst_length=64	
	#pragma HLS INTERFACE mode=m_axi port=w_sf 		bundle=D_TOK_W_SF 	depth=HD_SF_DEPTH 		offset=slave max_read_burst_length=64	
	#pragma HLS INTERFACE mode=m_axi port=w 			bundle=D_W_GEMM 	depth=HD_QUANT_DEPTH	offset=slave max_read_burst_length=256	


	#pragma HLS INTERFACE mode=s_axilite port=out 				bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=fl_tok 			bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=w_sf 				bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=w 					bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=N_DIM 			bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=M_DIM 			bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=CURR_LAYER 			bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=W_Off 			bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=return			bundle=control

	constexpr int mm_thr = 2;
	const int num = N_DIM * M_DIM ;
	const int num_sf = N_DIM * M_DIM / (MODEL_SCALING_FACTOR );
	const int sfCount = N_DIM / (MODEL_SCALING_FACTOR * SM_FL_ELEM);
	const int qCount = N_DIM / MAX_QUANT_ELEM;
	
	s_fdata_v_t tokens("tokens");
	#pragma HLS STREAM variable=tokens depth = 16// MODEL_HIDDEN_DIM/MAX_FL_ELEM
	s_fdata_v_t s_wsf("s_wsf");
	s_idata_v_t s_w("s_w");
	#pragma HLS BIND_STORAGE variable=s_w type=fifo impl=srl
	#pragma HLS STREAM variable=s_w type=fifo depth=16
	s_fdata_v_t dist_wsf[mm_thr];
	s_idata_v_t dist_w[mm_thr];
	
	s_fdata_v_t tok_sf;
  // #pragma HLS BIND_STORAGE variable=tok_sf type=ram_2p impl=bram
	s_idata_v_t tok;
	#pragma HLS STREAM variable=tok type=fifo depth=32
  // #pragma HLS BIND_STORAGE variable=tok type=ram_2p impl=uram
	idata_v_t w_arr[TOK_QUANT_MAX];

	hls::stream<my_float_t> wtok;
	hls::stream<my_float_t> wtok_sf;
	hls::stream<my_float_t> out_thread[mm_thr];
	s_fdata_v_t d_tok_sf[mm_thr];
	s_idata_v_t d_tok[mm_thr];
	s_fdata_v_t d_wsf[mm_thr];
	s_idata_v_t d_w[mm_thr];
	#pragma HLS STREAM variable=d_wsf depth = 16// MODEL_HIDDEN_DIM/MAX_FL_ELEM
	#pragma HLS STREAM variable=d_w depth = 16// MODEL_HIDDEN_DIM/MAX_FL_ELEM
	#pragma HLS BIND_STORAGE variable=d_w type=fifo impl=srl
	s_fdata_v_t s_out("s_out");
	

	#pragma HLS DATAFLOW
	// tok_load_input(tokens, fl_tok, N_DIM);
	mm2s_input_data(tokens, fl_tok, N_DIM / SM_FL_ELEM);
	
	quantizer_kernel(tok_sf, tok, tokens, N_DIM);
	inf_split_tee(d_tok_sf, tok_sf, (N_DIM / (MODEL_SCALING_FACTOR * SM_FL_ELEM)));
	inf_split_tee(d_tok, tok, (N_DIM / MAX_QUANT_ELEM));
	
	mm_load_input(s_wsf, w_sf, num_sf, CURR_LAYER);
	mm_tok_load_input(s_w, w, num, CURR_LAYER);

	inf_round_robin(d_wsf, s_wsf, (N_DIM / (MODEL_SCALING_FACTOR * SM_FL_ELEM)), M_DIM);
	inf_round_robin(d_w, s_w, (N_DIM / MAX_QUANT_ELEM), M_DIM);

	for (int i = 0; i < mm_thr; i++) {
		#pragma HLS UNROLL
		alt_mat_mult_main(out_thread[i], d_w[i], d_wsf[i], d_tok[i], d_tok_sf[i], N_DIM, M_DIM/mm_thr);
	}
	
	rr_merge(s_out, out_thread, M_DIM / SM_FL_ELEM);
	s2mm_output_data(out, s_out,M_DIM / SM_FL_ELEM, 0);
	return;
}