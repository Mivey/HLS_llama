#include "mha_forward.h"
#include "matmult.h"
#include "rmsnorm.h"
#include "swiglu.h"
#include "mha.h"
#include "quantizer.h"
#include "combiner.h"
#include <algorithm>
#include <cstdint>
#include <exception>
#include <hls_fence.h>
#include <iterator>
#include <sys/types.h>

constexpr int mm_thr = 2;
struct keys {
	bool FINAL_FLAG = 1;
	int N_DIM = MODEL_ELEMENTS;
	int M_DIM = MODEL_ELEMENTS;
	int CURR_LAYER = 0;
	int w_sf;
	int w;
	#ifdef __DEBUG__
	bool INIT;
	int SAVE_ADDR;
	#endif
};

struct fsm_data {
	// bool AXI_SEL = 0;
	int n = 0;
	int next_layer = 0;
	int curr_state = 0;
	int next_state = 0;
	
};

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

void cu_selecter(	s_fdata_v_t &s_tokens,
					fdata_v_t *weights, fdata_v_t *diff,
					adata_v_t *key_cache, adata_v_t *value_cache, fdata_v_t *res_con,
					keys &r, fsm_data &fd, axi_reg &tt, hls::stream<keys> &key_out){ 
	fd.curr_state = fd.next_state;
	// r.AXI_SEL = 0;
	r.CURR_LAYER = fd.next_layer;
	r.FINAL_FLAG = 1;
	
	#pragma HLS BIND_STORAGE variable=res_con type=ram_t2p impl=bram
	#pragma HLS ARRAY_PARTITION variable=res_con dim=1 type=cyclic factor=2

	#ifdef __DEBUG__
		r.SAVE_ADDR += (r.N_DIM / SM_FL_ELEM);
	#endif
	
	switch (fd.curr_state) {
	case 0 :	
		fd.next_state++;
		//current GeMV dimensions:
		r.N_DIM = MODEL_ELEMENTS; // 768 tokens
		r.M_DIM = MODEL_ELEMENTS * 3; // QKV
		r.w_sf = tt.QKV_sf_W / sizeof(mfdata_v_t);
		r.w = tt.QKV_W / sizeof(idata_v_t);
		#ifdef __DEBUG__
			if (r.INIT == true) {
				r.SAVE_ADDR = 0;
				r.INIT = false;
			} 
		#endif
		key_out.write(r);
		rmsnorm_kernel(s_tokens, diff, weights, res_con, r.CURR_LAYER, tt.rms_att_W / sizeof(fdata_v_t)); 
		break;
						
	case 1 :	
		fd.next_state++;
		//current GeMV dimensions:
		r.N_DIM = MODEL_ELEMENTS; // 768 tokens
		r.M_DIM = MODEL_ELEMENTS; // Out
		r.w_sf = tt.Out_sf_W / sizeof(mfdata_v_t);
		r.w = tt.Out_W / sizeof(idata_v_t);
		key_out.write(r);
		mha_kernel(s_tokens, diff, key_cache, value_cache, tt.POS, r.CURR_LAYER); 
		break;
						
	case 2 :	
		fd.next_state++;
		//current GeMV dimensions:
		r.N_DIM = MODEL_ELEMENTS; // 768 tokens
		r.M_DIM = MODEL_HIDDEN_DIM * 2; // gate & up
		r.w_sf = tt.FF_w1w3_sf_W / sizeof(mfdata_v_t);
		r.w = tt.FF_w1w3_W / sizeof(idata_v_t);
		key_out.write(r);
		rmsnorm_kernel(s_tokens, diff, weights, res_con, r.CURR_LAYER, tt.rms_ffn_W / sizeof(fdata_v_t));
		break;
						
	case 3 :	
		fd.next_layer = r.CURR_LAYER + 1;
		fd.next_state = (fd.next_layer == MODEL_NUM_LAYERS) ? 4 : 0;
		//current GeMV dimensions:
		r.N_DIM = MODEL_HIDDEN_DIM; // 2048 tokens
		r.M_DIM = MODEL_ELEMENTS; // down
		r.w_sf = tt.FF_w2_sf_W / sizeof(mfdata_v_t);
		r.w = tt.FF_w2_W / sizeof(idata_v_t);
		key_out.write(r);
		swiglu_kernel(s_tokens, diff); 
		break;
						
	case 4 :	
		r.N_DIM = MODEL_ELEMENTS; // 768 tokens
		r.M_DIM = MODEL_TOKENS; // embeddings out
		r.w_sf = tt.Embed_sf_W / sizeof(mfdata_v_t);
		r.w = tt.Embed_W / sizeof(idata_v_t);
		r.FINAL_FLAG = false;
		r.CURR_LAYER = 0;
		key_out.write(r);
		rmsnorm_kernel(s_tokens, diff, weights, res_con, 0, tt.rms_final_W/ sizeof(fdata_v_t)); 
		break;
	}
}

void df_region(	fdata_v_t *out, mfdata_v_t *w_sf_0, mfdata_v_t *w_sf_1, 
				idata_v_t *w_0, idata_v_t *w_1, s_fdata_v_t &s_cu_sel_in, ProbIndex *ss_reg, 
				hls::stream<keys> &vec_cnt
				#ifdef __DEBUG__
				, fdata_v_t *data_out
				#endif
				){
	const keys r = vec_cnt.read();
	#pragma HLS DATAFLOW // actually can't I then change this to inline if I do use dataflow in the transformer_kernel?
	// #pragma HLS INLINE
	hls::stream<my_float_t> s_tok_sf, s_out[mm_thr];
	hls::stream<fdata_v_t> tok_sf[mm_thr];
	s_idata_v_t s_tok_q, tok_q[mm_thr];
	hls::stream<ProbIndex> sys_sort;
	
		#pragma HLS STREAM variable=s_out depth=MODEL_SCALING_FACTOR
#pragma HLS BIND_STORAGE variable=s_out type=fifo impl=bram
		#pragma HLS STREAM variable=s_tok_sf depth=MODEL_HIDDEN_DIM/SM_FL_ELEM
		#pragma HLS STREAM variable=tok_sf depth=MODEL_HIDDEN_DIM/SM_FL_ELEM
		#pragma HLS STREAM variable=s_tok_q depth=MODEL_HIDDEN_DIM/MAX_QUANT_ELEM
		#pragma HLS STREAM variable=tok_q depth=MODEL_HIDDEN_DIM/MAX_QUANT_ELEM
		
	#ifndef __DEBUG__
	quantizer_kernel(s_tok_sf, s_tok_q, s_cu_sel_in, r.N_DIM);
	#endif
	#ifdef __DEBUG__
		quantizer_kernel(s_tok_sf, s_tok_q, s_cu_sel_in, r.N_DIM, data_out, r.SAVE_ADDR);
	#endif

	inf_split_tee(tok_sf, s_tok_sf, (r.N_DIM / (MODEL_SCALING_FACTOR * SM_FL_ELEM)));
	inf_split_tee(tok_q, s_tok_q, (r.N_DIM / MAX_QUANT_ELEM));

	GeMV_kernel(s_out[0], tok_sf[0], tok_q[0], w_sf_0, w_0, r.N_DIM, r.M_DIM/2, r.CURR_LAYER * 2 + 0, 0, r.w_sf, r.w);
	GeMV_kernel(s_out[1], tok_sf[1], tok_q[1], w_sf_1, w_1, r.N_DIM, r.M_DIM/2, r.CURR_LAYER * 2 + 1, r.M_DIM/2, r.w_sf, r.w);
	
	gemv_split(out, sys_sort, s_out, r.M_DIM, r.FINAL_FLAG);
	// systolic_sort(sys_sort, ss_reg, r.M_DIM);
	insertion_sort(sys_sort, ss_reg, r.M_DIM);
}

void transformer_cu(
				fdata_v_t *tokens,
				mfdata_v_t *w_sf_0, idata_v_t *w_0, 
				mfdata_v_t *w_sf_1, idata_v_t *w_1, 
				fdata_v_t *weights, mfdata_v_t *key_cache, mfdata_v_t *value_cache, 
				const int POS,
				const int QKV_W, const int QKV_sf_W,
				const int Out_W, const int Out_sf_W,
				const int FF_w1w3_W, const int FF_w1w3_sf_W,
				const int FF_w2_W, const int FF_w2_sf_W, 
				const int Embed_W, const int Embed_sf_W, 
				const int rms_att_W, const int rms_ffn_W, const int rms_final_W,
			#ifdef __DEBUG__
				const int faker, const int CURR_LAYER, const int NEXT_STATE, fdata_v_t *data_out,
			#endif
				const float temperature, const float coin//, int* pick
			){
	
	
	constexpr int q_size = (MODEL_ELEMENTS * ((MODEL_ELEMENTS * 4 + MODEL_HIDDEN_DIM * 3 ) * MODEL_NUM_LAYERS + MODEL_TOKENS)) * sizeof(int8_t);
	constexpr int rms_size = (MODEL_ELEMENTS * (MODEL_NUM_LAYERS * 2 + 1)) * sizeof(my_float_t);
	constexpr int sf_size = (q_size * sizeof(my_float_t) / (sizeof(int8_t) * MODEL_SCALING_FACTOR));
	
	constexpr int RMS_DEPTH = MODEL_ELEMENTS * (MODEL_NUM_LAYERS * 2 + 1) / SM_FL_ELEM;
	constexpr int CACHE_DEPTH = MODEL_ELEMENTS * MODEL_SEQUENCE_LEN * MODEL_NUM_LAYERS / MAX_FL_ELEM;
	constexpr int TOK_DEPTH = MODEL_ELEMENTS / MAX_FL_ELEM;
	constexpr int HD_QUANT_DEPTH = q_size / MAX_QUANT_ELEM;
	constexpr int HD_SF_DEPTH = sf_size / MAX_FL_ELEM; 
	constexpr int TOK_OUT_DEPTH = INTERNAL_DATA_SIZE / SM_FL_ELEM;
	constexpr int MHA_DEPTH = MODEL_ELEMENTS / MID_FL_ELEM * 3;
	constexpr int RECORD_DEPTH = ((MODEL_ELEMENTS * 3  + MODEL_HIDDEN_DIM) * 12 + MODEL_ELEMENTS) * 64 / SM_FL_ELEM;

	#pragma HLS INTERFACE mode=m_axi port=tokens 				bundle=w_n_t_gemm 		depth=TOK_OUT_DEPTH 	offset=slave max_write_burst_length=16 max_read_burst_length=(4096/SM_DW*8)
	#ifdef __DEBUG__
		#pragma HLS INTERFACE mode=m_axi port=data_out 				bundle=w_n_t_gemm 		depth=RECORD_DEPTH 	offset=slave max_write_burst_length=16 max_read_burst_length=(4096/SM_DW*8)
	#endif
	#pragma HLS INTERFACE mode=m_axi port=w_sf_0 				bundle=D_TOK_W_SF_0 		depth=HD_SF_DEPTH 		offset=slave max_read_burst_length=(4096/MAX_DW * 8)		num_read_outstanding=16
	#pragma HLS INTERFACE mode=m_axi port=w_0 					bundle=D_W_GEMM_0 		depth=HD_QUANT_DEPTH 	offset=slave max_read_burst_length=(4096/MAX_DW * 8) 		num_read_outstanding=64 
	#pragma HLS INTERFACE mode=m_axi port=w_sf_1 				bundle=D_TOK_W_SF_1		 	depth=HD_SF_DEPTH 		offset=slave max_read_burst_length=(4096/MAX_DW * 8)		num_read_outstanding=16
	#pragma HLS INTERFACE mode=m_axi port=w_1 					bundle=D_W_GEMM_1 		depth=HD_QUANT_DEPTH 	offset=slave max_read_burst_length=(4096/MAX_DW * 8) 		num_read_outstanding=64 
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
		#pragma HLS INTERFACE mode=s_axilite port=faker 						bundle=control
		#pragma HLS INTERFACE mode=s_axilite port=CURR_LAYER 				bundle=control
		#pragma HLS INTERFACE mode=s_axilite port=NEXT_STATE 				bundle=control
		#pragma HLS INTERFACE mode=s_axilite port=data_out 				bundle=control
	#endif
	#pragma HLS INTERFACE mode=s_axilite port=temperature			bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=coin		bundle=control
	
	fdata_v_t internal_token[INTERNAL_DATA_SIZE/SM_FL_ELEM];
	fdata_v_t res_con[MODEL_ELEMENTS / SM_FL_ELEM]{};
	ProbIndex ss_reg[REG_SIZE];
	
	#pragma HLS ARRAY_PARTITION variable=internal_token dim=1 factor=2 type=block
	#pragma HLS BIND_STORAGE variable=internal_token type=ram_1p impl=uram
	#pragma HLS ARRAY_PARTITION variable=ss_reg complete dim=1
	
	keys runner;
	fsm_data fd;
	runner.INIT = 1;
	
	fd.next_layer = 0;
	fd.next_layer = 0;
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
	};

	#ifndef __DEBUG__
		const int faker = MODEL_NUM_LAYERS * 4 + 1;
	#endif
	#ifdef __DEBUG__
		fd.next_layer = CURR_LAYER;
		fd.next_state = NEXT_STATE;
		runner.SAVE_ADDR = 0;
		mm2mm_store(res_con, data_out, MODEL_ELEMENTS);
	#endif
	
	mm2mm_store(internal_token, tokens, MODEL_ELEMENTS, 2, 0, INTERNAL_DATA_SIZE);
	mm2mm_store(internal_token, tokens, MODEL_ELEMENTS, 2, 1, INTERNAL_DATA_SIZE);

	for(int ii = 0; ii < faker; ii++) {
		s_fdata_v_t s_cu_sel_out;
		hls::stream<keys> vec_cnt;
		#pragma HLS STREAM variable=s_cu_sel_out depth=MODEL_HIDDEN_DIM/SM_FL_ELEM
		#pragma HLS DATAFLOW 
		
		cu_selecter(s_cu_sel_out, weights, internal_token, key_cache, value_cache, res_con, runner, fd, tt, vec_cnt);
		df_region(internal_token, w_sf_0, w_sf_1, w_0, w_1, s_cu_sel_out, ss_reg, vec_cnt
								#ifdef __DEBUG__
								, data_out
								#endif
		);
	}
	#ifdef __DEBUG__
		mm2mm_store(tokens, internal_token, INTERNAL_DATA_SIZE);

	#endif
	// #ifndef __DEBUG__
		ss_final(ss_reg, tokens, temperature, 0.9, coin);
	// #endif
	return;
}