
#include "matmult.h"
#include <cstddef>
#include <cstdint>
#include <hls_math.h>
#include <memory>
// #include "forward.h"
#include "hls_task.h"
#include "mha_forward.h"

constexpr size_t TOK_QUANT_MAX =  (MODEL_HIDDEN_DIM / MAX_QUANT_ELEM);
constexpr size_t TOK_SF_MAX = (MODEL_HIDDEN_DIM / (MODEL_SCALING_FACTOR * SM_FL_ELEM));
constexpr int UF = 1;

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
			amm_k_calc:
			for (size_t k = 0; k < SM_FL_ELEM; k++) {
				//do our calculations
				#pragma HLS PIPELINE II=1
				
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


void GeMV_kernel(fdata_v_t *out, s_fdata_v_t &tok_sf, s_idata_v_t &tok_q, fdata_v_t *w_sf, idata_v_t *w, const int N_DIM, const int M_DIM, const int CURR_LAYER, const int W_Off){

	constexpr int mm_thr = 2;
	// const int num = N_DIM * M_DIM ;
	// const int num_sf = N_DIM * M_DIM / (MODEL_SCALING_FACTOR );
	const int w_count = N_DIM * M_DIM / MAX_QUANT_ELEM;
	const int sf_count = N_DIM * M_DIM / (MODEL_SCALING_FACTOR * SM_FL_ELEM);
	const int sfCount = N_DIM / (MODEL_SCALING_FACTOR * SM_FL_ELEM);
	const int qCount = N_DIM / MAX_QUANT_ELEM;
	
	s_fdata_v_t tokens("tokens");
	#pragma HLS STREAM variable=tokens depth = 16// MODEL_HIDDEN_DIM/MAX_FL_ELEM
	s_fdata_v_t s_wsf("s_wsf");
#pragma HLS BIND_STORAGE variable=s_wsf type=fifo impl=bram
	#pragma HLS STREAM variable=s_wsf type=fifo depth=16
	s_idata_v_t s_w("s_w");
#pragma HLS BIND_STORAGE variable=s_w type=fifo impl=bram
	#pragma HLS STREAM variable=s_w type=fifo depth=16
	s_fdata_v_t dist_wsf[mm_thr];
	s_idata_v_t dist_w[mm_thr];
	
	#pragma HLS STREAM variable=tok_q type=fifo depth=32
  // #pragma HLS BIND_STORAGE variable=tok type=ram_2p impl=uram
	idata_v_t w_arr[TOK_QUANT_MAX];

	hls::stream<my_float_t> wtok;
	hls::stream<my_float_t> wtok_sf;
	hls::stream<my_float_t> out_thread[mm_thr];
	s_fdata_v_t d_tok_sf[mm_thr];
	s_idata_v_t d_tok[mm_thr];
	s_fdata_v_t d_wsf[mm_thr];
	s_idata_v_t d_w[mm_thr];
	#pragma HLS STREAM variable=d_wsf depth = 4096// MODEL_HIDDEN_DIM/MAX_FL_ELEM
	#pragma HLS STREAM variable=d_w depth = 4096// MODEL_HIDDEN_DIM/MAX_FL_ELEM
#pragma HLS BIND_STORAGE variable=d_w type=fifo impl=uram
#pragma HLS BIND_STORAGE variable=d_wsf type=fifo impl=uram
	s_fdata_v_t s_out("s_out");
	

	#pragma HLS DATAFLOW
	// tok_load_input(tokens, fl_tok, N_DIM);
	inf_split_tee(d_tok_sf, tok_sf, (N_DIM / (MODEL_SCALING_FACTOR * SM_FL_ELEM)));
	inf_split_tee(d_tok, tok_q, (N_DIM / MAX_QUANT_ELEM));
	
	mm2s_input_data(s_wsf, w_sf, sf_count, CURR_LAYER);
	mm2s_input_data(s_w, w, w_count, CURR_LAYER);

	inf_round_robin(d_wsf, s_wsf, (N_DIM / (MODEL_SCALING_FACTOR * SM_FL_ELEM)), M_DIM);
	inf_round_robin(d_w, s_w, (N_DIM / MAX_QUANT_ELEM), M_DIM);

	for (int i = 0; i < mm_thr; i++) {
		#pragma HLS UNROLL
		alt_mat_mult_main(out_thread[i], d_w[i], d_wsf[i], d_tok[i], d_tok_sf[i], N_DIM, M_DIM/mm_thr);
	}
	
	rr_merge(s_out, out_thread, M_DIM / SM_FL_ELEM);
	s2mm_output_data(out, s_out, M_DIM / SM_FL_ELEM, (W_Off / SM_FL_ELEM));
	return;
}