#include "matmult.h"
#include "mha_forward.h"
#include "rmsnorm.h"
#include "swiglu.h"
#include "mha.h"
#include "quantizer.h"
#include <cstdint>

constexpr int mm_thr = 2;

void cu_selecter(s_fdata_v_t &s_tokens, 
									fdata_v_t internal_tokens[MODEL_ELEMENTS/SM_FL_ELEM], fdata_v_t *diff, fdata_v_t *weights, 
									fdata_v_t *w1w3, 
									adata_v_t *tokens, adata_v_t *key_cache, adata_v_t *value_cache, 
									const int POS, const int CURR_LAYER, const int SEL, const int INIT){
	switch (SEL) {
	case 0 : rmsnorm_kernel(s_tokens, internal_tokens, diff, weights, CURR_LAYER, INIT); break;
	case 1 : swiglu_kernel(s_tokens, w1w3); break;
	default: mha_kernel(s_tokens, tokens, key_cache, value_cache, POS, CURR_LAYER); break;
	}
}

void transformer_cu(	//s_fdata_v_t (&tok_sf)[mm_thr] , s_idata_v_t (&tok_q)[mm_thr],
								fdata_v_t *out_0, fdata_v_t *w_sf_0, idata_v_t *w_0, 
								fdata_v_t *out_1, fdata_v_t *w_sf_1, idata_v_t *w_1, 
								fdata_v_t *diff, fdata_v_t *weights, fdata_v_t *w1w3, 
								adata_v_t *tokens, adata_v_t *key_cache, adata_v_t *value_cache, 
								const int POS, const int CURR_LAYER, const int SEL, const int INIT,
								const int N_DIM, const int M_DIM){
	
	constexpr int RMS_DEPTH = MODEL_ELEMENTS / SM_FL_ELEM;
	constexpr int CACHE_DEPTH = MODEL_ELEMENTS * MODEL_SEQUENCE_LEN * MODEL_NUM_LAYERS / MAX_FL_ELEM;
	constexpr int TOK_DEPTH = MODEL_ELEMENTS / MAX_FL_ELEM;
	constexpr int HD_QUANT_DEPTH = MODEL_HIDDEN_DIM * MODEL_ELEMENTS * MODEL_NUM_LAYERS  / MAX_QUANT_ELEM;
	constexpr int HD_SF_DEPTH = MODEL_HIDDEN_DIM * MODEL_ELEMENTS * MODEL_NUM_LAYERS / (MODEL_SCALING_FACTOR * SM_FL_ELEM);
	constexpr int TOK_OUT_DEPTH = MODEL_HIDDEN_DIM / SM_FL_ELEM;
	

	#pragma HLS INTERFACE mode=m_axi port=out_0 		bundle=D_TOK_W_SF_0 	depth=TOK_OUT_DEPTH 	offset=slave max_write_burst_length=(4096/SM_DW * 8)	
	#pragma HLS INTERFACE mode=m_axi port=w_sf_0 		bundle=D_TOK_W_SF_0 	depth=HD_SF_DEPTH 		offset=slave max_read_burst_length=(4096/SM_DW * 8)	
	#pragma HLS INTERFACE mode=m_axi port=w_0 				bundle=D_W_GEMM_0 		depth=HD_QUANT_DEPTH 	offset=slave max_read_burst_length=(4096/MAX_DW*8) num_read_outstanding=32 

	#pragma HLS INTERFACE mode=m_axi port=out_1 		bundle=D_TOK_W_SF_1 	depth=TOK_OUT_DEPTH 	offset=slave max_write_burst_length=(4096/SM_DW * 8)	
	#pragma HLS INTERFACE mode=m_axi port=w_sf_1 		bundle=D_TOK_W_SF_1 	depth=HD_SF_DEPTH 		offset=slave max_read_burst_length=(4096/SM_DW * 8)	
	#pragma HLS INTERFACE mode=m_axi port=w_1 				bundle=D_W_GEMM_1 		depth=HD_QUANT_DEPTH 	offset=slave max_read_burst_length=(4096/MAX_DW*8) num_read_outstanding=32 
	
	#pragma HLS INTERFACE mode=m_axi port=weights					bundle=rms_out_w 	depth=RMS_DEPTH		offset=slave max_read_burst_length=(4096/SM_DW * 8)
	#pragma HLS INTERFACE mode=m_axi port=diff						bundle=rms_out_w 	depth=RMS_DEPTH		offset=slave max_read_burst_length=(4096/SM_DW * 8)
	#pragma HLS INTERFACE mode=m_axi port=w1w3						bundle=sg_in_out		depth=MODEL_HIDDEN_DIM	offset=slave max_read_burst_length=(4096/SM_DW * 8)
	#pragma HLS INTERFACE mode=m_axi port=value_cache			bundle=vc_gemm		depth=CACHE_DEPTH			offset=slave max_read_burst_length=(4096/MID_DW * 8)		max_write_burst_length=(4096/MAX_DW * 8)
	#pragma HLS INTERFACE mode=m_axi port=key_cache				bundle=kc_gemm		depth=CACHE_DEPTH			offset=slave max_read_burst_length=(4096/MID_DW * 8)		max_write_burst_length=(4096/MAX_DW * 8)
	#pragma HLS INTERFACE mode=m_axi port=tokens					bundle=token_gemm	depth=TOK_DEPTH				offset=slave max_read_burst_length=(4096/MAX_DW * 8)


	#pragma HLS INTERFACE mode=s_axilite port=weights				bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=diff				bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=w1w3			bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=tokens				 					bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=value_cache 						bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=key_cache								bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=CURR_LAYER 							bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=POS 										bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=INIT										bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=SEL 										bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=return 				bundle=control

	#pragma HLS INTERFACE mode=s_axilite port=out_0 				bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=w_sf_0 				bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=w_0 					bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=out_1 				bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=w_sf_1 				bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=w_1 					bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=N_DIM 			bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=M_DIM 			bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=return			bundle=control

	static fdata_v_t internal_tokens[MODEL_ELEMENTS / SM_FL_ELEM];
	#pragma HLS BIND_STORAGE variable=internal_tokens type=ram_t2p impl=bram
	#pragma HLS ARRAY_PARTITION variable=internal_tokens dim=1 type=cyclic factor=2

#pragma HLS DATAFLOW
s_fdata_v_t s_tokens, s_tok_sf, tok_sf[mm_thr];
s_idata_v_t s_tok_q, tok_q[mm_thr];

	cu_selecter(s_tokens, internal_tokens, diff, weights, w1w3, tokens, key_cache, value_cache, POS, CURR_LAYER, SEL, INIT);
	quantizer_kernel(s_tok_sf, s_tok_q, s_tokens, N_DIM);

	inf_split_tee(tok_sf, s_tok_sf, (N_DIM / (MODEL_SCALING_FACTOR * SM_FL_ELEM)));
	inf_split_tee(tok_q, s_tok_q, (N_DIM / MAX_QUANT_ELEM));

	GeMV_kernel(out_0, tok_sf[0], tok_q[0], w_sf_0, w_0, N_DIM, M_DIM/2, 0, 0);
	GeMV_kernel(out_1, tok_sf[1], tok_q[1], w_sf_1, w_1, N_DIM, M_DIM/2, 1, M_DIM/2);
	return;
}
