#include "matmult.h"
#include "mha_forward.h"
#include "rmsnorm.h"
#include "swiglu.h"
#include "mha.h"
#include "quantizer.h"
#include <cstdint>
#include <exception>
#include <iterator>
#include <sys/types.h>

constexpr int mm_thr = 2;
struct keys {
	int n = 0;
	int CURR_LAYER = 0;
	int AXI_SEL = 0;
	int N_DIM = MODEL_ELEMENTS;
	int M_DIM = MODEL_ELEMENTS;
	int curr_state = 0;
	int next_state = 0;
	int w_sf;
	int w;
	int INIT;
} ;

struct axi_reg{
	int POS;
	int N_DIM;
	int M_DIM; 
	int QKV_W;
	int QKV_sf_W;
	int Out_W;
	int Out_sf_W;
	int FF_w1w3_W;
	int FF_w1w3_sf_W;
	int FF_w2_W;
	int FF_w2_sf_W; 
	int Embed_W;
	int Embed_sf_W; 
	int rms_att_W;
	int rms_ffn_W; 
	int rms_final_W;
};

void cu_selecter(	s_fdata_v_t &s_tokens, fdata_v_t internal_tokens[MODEL_ELEMENTS/SM_FL_ELEM], 
									fdata_v_t *weights, fdata_v_t *diff, adata_v_t *mha_tokens, fdata_v_t *w1w3,
									adata_v_t *key_cache, adata_v_t *value_cache, 
									const int POS, keys &r, axi_reg &tt){
	// add back init
	// remove case 5
	// add diff loader outside of for loop;
	r.INIT =  ((r.next_state == 0) && (r.CURR_LAYER == 0)) ? 1 : 0;
	r.curr_state = r.next_state;
	r.AXI_SEL = 0;
	switch (r.curr_state) {
	case 0 :	rmsnorm_kernel(s_tokens, internal_tokens, diff, weights, r.CURR_LAYER, r.INIT, tt.rms_att_W); 
						r.next_state++;
						//current GeMV dimensions:
						r.N_DIM = MODEL_ELEMENTS; // 768 tokens
						r.M_DIM = MODEL_ELEMENTS * 3; // QKV
						r.w_sf = tt.QKV_sf_W;
						r.w = tt.QKV_W;
						break;
	case 1 :	mha_kernel(s_tokens, mha_tokens, key_cache, value_cache, POS, r.CURR_LAYER); 
						r.next_state++;
						//current GeMV dimensions:
						r.N_DIM = MODEL_ELEMENTS; // 768 tokens
						r.M_DIM = MODEL_ELEMENTS; // Out
						r.w_sf = tt.Out_sf_W;
						r.w = tt.Out_W;
						break;
	case 2 :	rmsnorm_kernel(s_tokens, internal_tokens, diff, weights, r.CURR_LAYER, 0, tt.rms_ffn_W);
						r.next_state++;
						//current GeMV dimensions:
						r.N_DIM = MODEL_ELEMENTS; // 768 tokens
						r.M_DIM = MODEL_HIDDEN_DIM * 2; // gate & up
						r.w_sf = tt.FF_w1w3_sf_W;
						r.w = tt.FF_w1w3_W;
						break;
	case 3 :	swiglu_kernel(s_tokens, w1w3); 
						r.CURR_LAYER++;
						r.next_state = (r.CURR_LAYER == MODEL_NUM_LAYERS) ? 4 : 0;
						//current GeMV dimensions:
						r.N_DIM = MODEL_HIDDEN_DIM; // 2048 tokens
						r.M_DIM = MODEL_ELEMENTS; // down
						r.w_sf = tt.FF_w2_sf_W;
						r.w = tt.FF_w2_W;
						break;
	case 4 :	rmsnorm_kernel(s_tokens, internal_tokens, diff, weights, 0, 0, tt.rms_final_W); 
						
						r.N_DIM = MODEL_ELEMENTS; // 768 tokens
						r.M_DIM = MODEL_TOKENS; // embeddings out
						r.w_sf = tt.Embed_sf_W;
						r.w = tt.Embed_W;
						r.AXI_SEL = 1;
						break;
	// case 5 :	rmsnorm_kernel(s_tokens, internal_tokens, diff, weights, r.CURR_LAYER, 1, tt.rms_att_W); 
	// 					r.next_state++;
	// 					//current GeMV dimensions:
	// 					r.N_DIM = MODEL_ELEMENTS; // 768 tokens
	// 					r.M_DIM = MODEL_ELEMENTS * 3; // QKV
	// 					r.w_sf = tt.QKV_sf_W;
	// 					r.w = tt.QKV_W;
	// 					break;
	}
}

void transformer_cu(	//s_fdata_v_t (&tok_sf)[mm_thr] , s_idata_v_t (&tok_q)[mm_thr],
								fdata_v_t *out_0, fdata_v_t *w_sf_0, idata_v_t *w_0, 
								fdata_v_t *out_1, fdata_v_t *w_sf_1, idata_v_t *w_1, 
								fdata_v_t *tokens, fdata_v_t *weights, fdata_v_t *w1w3, 
								adata_v_t *mha_tokens, adata_v_t *key_cache, adata_v_t *value_cache, 
								const int POS, const int N_DIM, const int M_DIM, 
								const int QKV_W, const int QKV_sf_W,
								const int Out_W, const int Out_sf_W,
								const int FF_w1w3_W, const int FF_w1w3_sf_W,
								const int FF_w2_W, const int FF_w2_sf_W, 
								const int Embed_W, const int Embed_sf_W, 
								const int rms_att_W, const int rms_ffn_W, const int rms_final_W,
								const int faker){
	
	constexpr int RMS_DEPTH = MODEL_ELEMENTS / SM_FL_ELEM;
	constexpr int CACHE_DEPTH = MODEL_ELEMENTS * MODEL_SEQUENCE_LEN * MODEL_NUM_LAYERS / MAX_FL_ELEM;
	constexpr int TOK_DEPTH = MODEL_ELEMENTS / MAX_FL_ELEM;
	constexpr int HD_QUANT_DEPTH = MODEL_HIDDEN_DIM * MODEL_ELEMENTS * MODEL_NUM_LAYERS * 2 / MAX_QUANT_ELEM;
	constexpr int HD_SF_DEPTH = MODEL_HIDDEN_DIM * MODEL_ELEMENTS * MODEL_NUM_LAYERS * 2 / (MODEL_SCALING_FACTOR * SM_FL_ELEM);
	constexpr int TOK_OUT_DEPTH = MODEL_HIDDEN_DIM / SM_FL_ELEM * 2;
	constexpr int MHA_DEPTH = MODEL_ELEMENTS / MID_FL_ELEM * 3;
	

	#pragma HLS INTERFACE mode=m_axi port=out_0 		bundle=D_TOK_W_SF_0 	depth=TOK_OUT_DEPTH 	offset=slave max_write_burst_length=(4096/SM_DW * 8)	
	#pragma HLS INTERFACE mode=m_axi port=w_sf_0 		bundle=D_TOK_W_SF_0 	depth=HD_SF_DEPTH 		offset=slave max_read_burst_length=(4096/SM_DW * 8)	
	#pragma HLS INTERFACE mode=m_axi port=w_0 				bundle=D_W_GEMM_0 		depth=HD_QUANT_DEPTH 	offset=slave max_read_burst_length=(4096/MAX_DW*8) num_read_outstanding=32 
	#pragma HLS INTERFACE mode=m_axi port=out_1 		bundle=D_TOK_W_SF_1 	depth=TOK_OUT_DEPTH 	offset=slave max_write_burst_length=(4096/SM_DW * 8)	
	#pragma HLS INTERFACE mode=m_axi port=w_sf_1 		bundle=D_TOK_W_SF_1 	depth=HD_SF_DEPTH 		offset=slave max_read_burst_length=(4096/SM_DW * 8)	
	#pragma HLS INTERFACE mode=m_axi port=w_1 				bundle=D_W_GEMM_1 		depth=HD_QUANT_DEPTH 	offset=slave max_read_burst_length=(4096/MAX_DW*8) num_read_outstanding=32 
	
	#pragma HLS INTERFACE mode=m_axi port=w1w3					bundle=rms_out_w 	depth=TOK_OUT_DEPTH		offset=slave max_read_burst_length=(4096/SM_DW * 8)
	#pragma HLS INTERFACE mode=m_axi port=mha_tokens			bundle=vc_gemm 		depth=MHA_DEPTH		offset=slave max_read_burst_length=(4096/MID_DW * 8)
	#pragma HLS INTERFACE mode=m_axi port=weights					bundle=rms_out_w 	depth=RMS_DEPTH		offset=slave max_read_burst_length=(4096/SM_DW * 8)
	#pragma HLS INTERFACE mode=m_axi port=tokens					bundle=rms_out_w 	depth=RMS_DEPTH		offset=slave max_read_burst_length=(4096/SM_DW * 8)
	#pragma HLS INTERFACE mode=m_axi port=value_cache			bundle=vc_gemm		depth=CACHE_DEPTH			offset=slave max_read_burst_length=(4096/MID_DW * 8)		max_write_burst_length=(4096/MAX_DW * 8)
	#pragma HLS INTERFACE mode=m_axi port=key_cache				bundle=kc_gemm		depth=CACHE_DEPTH			offset=slave max_read_burst_length=(4096/MID_DW * 8)		max_write_burst_length=(4096/MAX_DW * 8)

	#pragma HLS INTERFACE mode=s_axilite port=out_0 				bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=w_sf_0 				bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=w_0 					bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=out_1 				bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=w_sf_1 				bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=w_1 					bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=weights				bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=tokens				bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=value_cache 	bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=key_cache			bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=w1w3 					bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=mha_tokens		bundle=control
	
	#pragma HLS INTERFACE mode=s_axilite port=POS 					bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=N_DIM 				bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=M_DIM 				bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=faker 				bundle=control
	
	#pragma HLS INTERFACE mode=s_axilite port=QKV_W					bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=QKV_sf_W			bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=Out_W					bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=Out_sf_W			bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=FF_w1w3_W			bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=FF_w1w3_sf_W	bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=FF_w2_W				bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=FF_w2_sf_W 		bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=Embed_W				bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=Embed_sf_W 		bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=rms_att_W			bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=rms_ffn_W			bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=rms_final_W		bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=return				bundle=control

	fdata_v_t internal_tokens[MODEL_ELEMENTS / SM_FL_ELEM];
	// fdata_v_t internal_diff[MODEL_HIDDEN_DIM/SM_FL_ELEM * 2];
	s_fdata_v_t internal_stream[2];
	#pragma HLS BIND_STORAGE variable=internal_tokens type=ram_t2p impl=bram
	#pragma HLS ARRAY_PARTITION variable=internal_tokens dim=1 type=cyclic factor=2
	
	// #pragma HLS BIND_STORAGE variable=internal_diff type=ram_t2p impl=bram
	// #pragma HLS ARRAY_PARTITION variable=internal_diff dim=1 type=block factor=2
	
	keys runner;
	runner.CURR_LAYER = 0;
	axi_reg tt = {
		POS, 
		N_DIM, 
		M_DIM, 
		QKV_W, 
		QKV_sf_W,
		Out_W, 
		Out_sf_W,
		FF_w1w3_W, 
		FF_w1w3_sf_W,
		FF_w2_W, 
		FF_w2_sf_W, 
		Embed_W, 
		Embed_sf_W, 
		rms_att_W, 
		rms_ffn_W, 
		rms_final_W
	};
	// s_fdata_v_t foo;
	// mm2s_input_data(foo, tokens, MODEL_ELEMENTS/SM_FL_ELEM);
	// s2mm_output_data(internal_diff, foo, MODEL_ELEMENTS / SM_FL_ELEM, 0);

	for(int ii = 0; ii < faker; ii++) {
		s_fdata_v_t s_tokens;
		#pragma HLS STREAM variable=s_tokens depth=MODEL_HIDDEN_DIM/SM_FL_ELEM
		cu_selecter(s_tokens, internal_tokens, weights, tokens, mha_tokens, w1w3, key_cache, value_cache, POS, runner, tt);
		
		int rn = runner.N_DIM;
		int rm = runner.M_DIM;
		// int ra = runner.AXI_SEL;
			
		// {
			// #pragma HLS DATAFLOW
			hls::stream<my_float_t> s_tok_sf;
			s_fdata_v_t tok_sf[mm_thr];
			s_idata_v_t s_tok_q, tok_q[mm_thr];

			quantizer_kernel(s_tok_sf, s_tok_q, s_tokens, N_DIM);

			inf_split_tee(tok_sf, s_tok_sf, (N_DIM / (MODEL_SCALING_FACTOR * SM_FL_ELEM)));
			inf_split_tee(tok_q, s_tok_q, (N_DIM / MAX_QUANT_ELEM));

			GeMV_kernel(out_0, tok_sf[0], tok_q[0], w_sf_0, w_0, rn, rm/2, 0, 0);
			GeMV_kernel(out_1, tok_sf[1], tok_q[1], w_sf_1, w_1, rn, rm/2, 1, rm/2);
		// }
		// s2arr_output_data(internal_diff, internal_stream, rm, 0, ra);
	}
	return;
}

