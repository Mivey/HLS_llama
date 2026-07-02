#include "mha_forward.h"
#include "matmult.h"
#include "rmsnorm.h"
#include "swiglu.h"
#include "mha.h"
#include "quantizer.h"
#include "combiner.h"
// #include "sampler.h"
#include <algorithm>
#include <cstdint>
#include <exception>
#include <hls_fence.h>
#include <iterator>
#include <limits>
#include <sys/types.h>

constexpr int mm_thr = 2;
struct keys {
	int n = 0;
	int CURR_LAYER = 0;
	int next_layer = 0;
	int AXI_SEL = 0;
	int N_DIM = MODEL_ELEMENTS;
	int M_DIM = MODEL_ELEMENTS;
	int curr_state = 0;
	int next_state = 0;
	int w_sf;
	int w;
	int INIT;
	int stop;
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
	// bool mha_return;
};

void cu_selecter(	/*s_fdata_v_t &s_tokens,*/ hls::stream<my_float_t> &tok_sf, s_idata_v_t &tok_w,
									fdata_v_t *weights, fdata_v_t *diff, //adata_v_t *mha_tokens, fdata_v_t *w1w3,
									mfdata_v_t *key_cache, mfdata_v_t *value_cache, 
									keys &r, axi_reg &tt){ //todo: remove POS, it's apart of the axi_reg keyring
	// add back init
	// remove case 5
	// add diff loader outside of for loop;
	r.curr_state = r.next_state;
	r.AXI_SEL = 0;
	r.CURR_LAYER = r.next_layer;
	// r.INIT =  ((r.next_state == 0) && (r.CURR_LAYER == 0)) ? 1 : 0;
	switch (r.curr_state) {
	case 0 :	rmsnorm_kernel(tok_w, tok_sf, diff, weights, r.CURR_LAYER, r.INIT, tt.rms_att_W / sizeof(fdata_v_t)); 
						r.next_state++;
						//current GeMV dimensions:
						r.N_DIM = MODEL_ELEMENTS; // 768 tokens
						r.M_DIM = MODEL_ELEMENTS * 3; // QKV
						r.w_sf = tt.QKV_sf_W / sizeof(fdata_v_t);
						r.w = tt.QKV_W / sizeof(idata_v_t);
						r.INIT = 0;
						break;
						
	case 1 :	mha_kernel(tok_sf, tok_w, diff, key_cache, value_cache, tt.POS, r.CURR_LAYER); 
						r.next_state++;
						//current GeMV dimensions:
						r.N_DIM = MODEL_ELEMENTS; // 768 tokens
						r.M_DIM = MODEL_ELEMENTS; // Out
						r.w_sf = tt.Out_sf_W / sizeof(fdata_v_t);
						r.w = tt.Out_W / sizeof(idata_v_t);
						break;
						
	case 2 :	rmsnorm_kernel(tok_w, tok_sf, diff, weights, r.CURR_LAYER, 0, tt.rms_ffn_W / sizeof(fdata_v_t));
						r.next_state++;
						//current GeMV dimensions:
						r.N_DIM = MODEL_ELEMENTS; // 768 tokens
						r.M_DIM = MODEL_HIDDEN_DIM * 2; // gate & up
						r.w_sf = tt.FF_w1w3_sf_W / sizeof(fdata_v_t);
						r.w = tt.FF_w1w3_W / sizeof(idata_v_t);
						break;
						
	case 3 :	swiglu_kernel(tok_sf, tok_w, diff); 
						r.next_layer = r.CURR_LAYER + 1;
						r.next_state = (r.next_layer == MODEL_NUM_LAYERS) ? 4 : 0;
						//current GeMV dimensions:
						r.N_DIM = MODEL_HIDDEN_DIM; // 2048 tokens
						r.M_DIM = MODEL_ELEMENTS; // down
						r.w_sf = tt.FF_w2_sf_W / sizeof(fdata_v_t);
						r.w = tt.FF_w2_W / sizeof(idata_v_t);
						break;
						
	case 4 :	rmsnorm_kernel(tok_w, tok_sf, diff, weights, 0, 0, tt.rms_final_W/ sizeof(fdata_v_t)); 
						r.N_DIM = MODEL_ELEMENTS; // 768 tokens
						r.M_DIM = MODEL_TOKENS; // embeddings out
						r.w_sf = tt.Embed_sf_W / sizeof(fdata_v_t);
						r.w = tt.Embed_W / sizeof(idata_v_t);
						r.AXI_SEL = 1;
						r.stop = 0;
						r.CURR_LAYER = 0;
						break;
	}
}
void df_region(	fdata_v_t *out, ProbIndex *ss_reg, fdata_v_t *w_sf_0, fdata_v_t *w_sf_1, 
								idata_v_t *w_0, idata_v_t *w_1, hls::stream<my_float_t> &s_tok_sf, s_idata_v_t &s_tok_q, 
								const int rn, const int rm, const int sf_reg, const int w_reg, const int layer,  const int boop,
								const float temperature, const float coin, int* pick){
	
	#pragma HLS DATAFLOW
	hls::stream<my_float_t> s_out[mm_thr];
	s_fdata_v_t tok_sf[mm_thr];
	s_idata_v_t tok_q[mm_thr];
	hls::stream<ProbIndex> ss_val;
	// s_fdata_v_t sys_sort;
	// hls::stream<int> sys_val;
	
		#pragma HLS STREAM variable=s_tok_sf depth=MODEL_HIDDEN_DIM/SM_FL_ELEM
		#pragma HLS STREAM variable=tok_sf depth=MODEL_HIDDEN_DIM/SM_FL_ELEM
		#pragma HLS STREAM variable=s_tok_q depth=MODEL_HIDDEN_DIM/MAX_QUANT_ELEM
		#pragma HLS STREAM variable=tok_q depth=MODEL_HIDDEN_DIM/MAX_QUANT_ELEM

	// quantizer_kernel(s_tok_sf, s_tok_q, s_cu_sel_in, rn);

	inf_split_tee(tok_sf, s_tok_sf, (rn / (MODEL_SCALING_FACTOR * SM_FL_ELEM)));
	inf_split_tee(tok_q, s_tok_q, (rn / MAX_QUANT_ELEM));

	GeMV_kernel(s_out[0], tok_sf[0], tok_q[0], w_sf_0, w_0, rn, rm/2, layer * 2 + 0, 0, sf_reg, w_reg);
	GeMV_kernel(s_out[1], tok_sf[1], tok_q[1], w_sf_1, w_1, rn, rm/2, layer * 2 + 1, rm/2, sf_reg, w_reg);
	// GeMV_kernel(s_out[0], tok_sf[0], tok_q[0], w_sf_0, w_0, rn, rm/2, layer * 2 + 0, 0, sf_reg, w_reg);
	// GeMV_kernel(s_out[1], tok_sf[1], tok_q[1], w_sf_1, w_1, rn, rm/2, layer * 2 + 1, rm/2, sf_reg, w_reg);

	// gemv_combo(out, s_out, rm);
	// gemv_split(out, s_out, rm);
	gemv_split(out, ss_val, s_out, rm, boop);
	systolic_sort(ss_val, ss_reg, rm);
}

void transformer_cu(
				fdata_v_t *tokens,
				fdata_v_t *w_sf_0, idata_v_t *w_0, 
				fdata_v_t *w_sf_1, idata_v_t *w_1, 
				fdata_v_t *weights, mfdata_v_t *key_cache, mfdata_v_t *value_cache, 
				const int POS,
				const int QKV_W, const int QKV_sf_W,
				const int Out_W, const int Out_sf_W,
				const int FF_w1w3_W, const int FF_w1w3_sf_W,
				const int FF_w2_W, const int FF_w2_sf_W, 
				const int Embed_W, const int Embed_sf_W, 
				const int rms_att_W, const int rms_ffn_W, const int rms_final_W,
			#ifdef __DEBUG__
				const int faker ,const int INIT, const int CURR_LAYER, const int NEXT_STATE,
			#endif
				const float temperature, const float coin, int* pick
			){
	
	
	constexpr int q_size = (MODEL_ELEMENTS * ((MODEL_ELEMENTS * 4 + MODEL_HIDDEN_DIM * 3 ) * MODEL_NUM_LAYERS + MODEL_TOKENS)) * sizeof(int8_t);
	constexpr int rms_size = (MODEL_ELEMENTS * (MODEL_NUM_LAYERS * 2 + 1)) * sizeof(my_float_t);
	constexpr int sf_size = (q_size * sizeof(my_float_t) / (sizeof(int8_t) * MODEL_SCALING_FACTOR));
	
	constexpr int RMS_DEPTH = MODEL_ELEMENTS * (MODEL_NUM_LAYERS * 2 + 1) / SM_FL_ELEM;
	constexpr int CACHE_DEPTH = MODEL_ELEMENTS * MODEL_SEQUENCE_LEN * MODEL_NUM_LAYERS / MAX_FL_ELEM;
	constexpr int TOK_DEPTH = MODEL_ELEMENTS / MAX_FL_ELEM;
	constexpr int HD_QUANT_DEPTH = q_size / MAX_QUANT_ELEM;//MODEL_HIDDEN_DIM * MODEL_ELEMENTS * MODEL_NUM_LAYERS * 2 / MAX_QUANT_ELEM;
	constexpr int HD_SF_DEPTH = sf_size / SM_FL_ELEM; //MODEL_HIDDEN_DIM * MODEL_ELEMENTS * MODEL_NUM_LAYERS * 2 / (MODEL_SCALING_FACTOR * SM_FL_ELEM);
	constexpr int TOK_OUT_DEPTH = MODEL_TOKENS / SM_FL_ELEM;
	constexpr int MHA_DEPTH = MODEL_ELEMENTS / MAX_FL_ELEM * 3;
	

	#pragma HLS INTERFACE mode=m_axi port=tokens 				bundle=w_n_t_gemm 		depth=TOK_OUT_DEPTH 	offset=slave max_write_burst_length=16 max_read_burst_length=(4096/SM_DW*8)
	#pragma HLS INTERFACE mode=m_axi port=w_sf_0 				bundle=D_TOK_W_SF_0 	depth=HD_SF_DEPTH 		offset=slave max_read_burst_length=(4096/SM_DW * 8)		num_read_outstanding=32
	#pragma HLS INTERFACE mode=m_axi port=w_0 					bundle=D_W_GEMM_0 		depth=HD_QUANT_DEPTH 	offset=slave max_read_burst_length=(4096/MAX_DW * 8) 	num_read_outstanding=32 
	#pragma HLS INTERFACE mode=m_axi port=w_sf_1 				bundle=D_TOK_W_SF_1 	depth=HD_SF_DEPTH 		offset=slave max_read_burst_length=(4096/SM_DW * 8)		num_read_outstanding=32
	#pragma HLS INTERFACE mode=m_axi port=w_1 					bundle=D_W_GEMM_1 		depth=HD_QUANT_DEPTH 	offset=slave max_read_burst_length=(4096/MAX_DW * 8) 	num_read_outstanding=32 
	#pragma HLS INTERFACE mode=m_axi port=weights				bundle=w_n_t_gemm 		depth=RMS_DEPTH				offset=slave max_read_burst_length=(4096/SM_DW * 8)
	#pragma HLS INTERFACE mode=m_axi port=value_cache		bundle=vc_gemm				depth=CACHE_DEPTH			offset=slave max_read_burst_length=(4096/MAX_DW * 8)	max_write_burst_length=(4096/MAX_DW * 8)
	#pragma HLS INTERFACE mode=m_axi port=key_cache			bundle=kc_gemm				depth=CACHE_DEPTH			offset=slave max_read_burst_length=(4096/MAX_DW * 8)	max_write_burst_length=(4096/MAX_DW * 8)

	#pragma HLS INTERFACE mode=s_axilite port=tokens 				bundle=control
	
	#pragma HLS INTERFACE mode=s_axilite port=w_sf_0 				bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=w_0 					bundle=control
	
	#pragma HLS INTERFACE mode=s_axilite port=w_sf_1 				bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=w_1 					bundle=control
	
	#pragma HLS INTERFACE mode=s_axilite port=weights				bundle=control
	
	#pragma HLS INTERFACE mode=s_axilite port=value_cache 	bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=key_cache			bundle=control
	
	#pragma HLS INTERFACE mode=s_axilite port=POS 					bundle=control
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
	
	#ifdef __DEBUG__
			#pragma HLS INTERFACE mode=s_axilite port=faker 				bundle=control
			#pragma HLS INTERFACE mode=s_axilite port=INIT 				bundle=control
			#pragma HLS INTERFACE mode=s_axilite port=CURR_LAYER 				bundle=control
			#pragma HLS INTERFACE mode=s_axilite port=NEXT_STATE 				bundle=control
	#endif
	#pragma HLS INTERFACE mode=s_axilite port=temperature			bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=coin		bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=pick				bundle=control
	// #pragma HLS INTERFACE mode=s_axilite port=mha_return				bundle=control

	// fdata_v_t internal_diff[MODEL_HIDDEN_DIM/SM_FL_ELEM * 2];
	// s_fdata_v_t internal_stream[2];
	fdata_v_t internal_token[MODEL_TOKENS/SM_FL_ELEM] = {};
	ProbIndex ss_reg[REG_SIZE];
	std::fill(internal_token->begin(), internal_token->end(), 0);
	
	#pragma HLS ARRAY_PARTITION variable=internal_token dim=1 factor=2 type=block
	#pragma HLS BIND_STORAGE variable=internal_token type=ram_1p impl=uram
	#pragma HLS ARRAY_PARTITION variable=reg complete dim=1
	
	keys runner;
	// runner.stop = 0;
	runner.INIT = 1;
	runner.stop = 1;
	// bool skip = (mha_return != 0) ? 1 : 0;
	
	runner.next_layer = 0;
	runner.next_layer = 0;
	axi_reg tt = {
		POS, 
		0, 
		0, 
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
		// skip
	};

	#ifndef __DEBUG__
		const int faker = MODEL_NUM_LAYERS * 4 + 1;
	#endif
	#ifdef __DEBUG__
		runner.next_layer = CURR_LAYER;
		runner.next_state = NEXT_STATE;
	#endif

	mm2mm_store(internal_token, tokens, MODEL_ELEMENTS, 2, 0, MODEL_TOKENS);
	mm2mm_store(internal_token, tokens, MODEL_ELEMENTS, 2, 1, MODEL_TOKENS);

	for(int ii = 0; ii < faker; ii++) {
		
		hls::stream<my_float_t> s_tok_sf;
		#pragma HLS STREAM variable=s_tok_sf depth=MODEL_HIDDEN_DIM/MODEL_SCALING_FACTOR
		s_idata_v_t s_tok_w;
		#pragma HLS STREAM variable=s_tok_w depth=MODEL_HIDDEN_DIM/MAX_QUANT_ELEM

		cu_selecter(s_tok_sf, s_tok_w, weights, internal_token, key_cache, value_cache, runner, tt);
		
		df_region(internal_token, ss_reg, w_sf_0, w_sf_1, w_0, w_1, s_tok_sf, s_tok_w, runner.N_DIM, runner.M_DIM, runner.w_sf, runner.w, runner.CURR_LAYER, runner.stop,
		temperature, coin, pick);
	}
	#ifdef __DEBUG__
		mm2mm_store(tokens, internal_token, MODEL_TOKENS);
	#endif
	#ifndef __DEBUG__
		// systolic_sort(internal_token, pick, temperature, coin);
		ss_final(ss_reg, pick, temperature, coin);
	#endif
	return;
}