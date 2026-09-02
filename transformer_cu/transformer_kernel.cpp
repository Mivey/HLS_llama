#include "mha_forward.h"
#include "matmult.h"
#include "rmsnorm.h"
#include "swiglu.h"
#include "mha.h"
#include "quantizer.h"
#include "combiner.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
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
  int SAVE_ADDR = 0;
  #endif
  #ifdef __ULTRADEBUG__
  int mmSAVE_ADDR = 0;
  #endif
  
  int sfCount() const {
    return N_DIM * M_DIM / (MODEL_SCALING_FACTOR * MAX_FL_ELEM * mm_thr);
  }

  int wCount() const {
    return N_DIM * M_DIM / (mm_thr * MAX_QUANT_ELEM);
  }
  
};

struct fsm_data {
  // bool AXI_SEL = 0;
  // int n = 0;
  int layer = 0;
  int state = 0;
  int next_state = 0;
  keys gemv_config;
  #ifdef __DEBUG__
  int SAVE_ADDR = 0;
  #endif
  #ifdef __ULTRADEBUG__
  int mmSAVE_ADDR = 0;
  #endif
  
};

struct axi_reg{
  int POS;
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

void cu_selecter(  s_fdata_v_t &s_tokens, hls::stream<keys> &key_out,
          fdata_v_t *weights, fdata_v_t *diff,
          adata_v_t *key_cache, adata_v_t *value_cache, fdata_v_t *res_con,
          hls::stream<fsm_data> &curr_state, const axi_reg &tt){ 
  
	// bool go = start.read(); // not using for now...
  fsm_data cs = curr_state.read();
  keys r = cs.gemv_config;
  
  switch (cs.state) {
		case 0 :  
			key_out.write(r);
			rmsnorm_kernel(s_tokens, diff, weights, res_con, r.CURR_LAYER, tt.rms_att_W / sizeof(fdata_v_t)); 
			break;
							
		case 1 :  
			key_out.write(r);
			mha_kernel(s_tokens, diff, key_cache, value_cache, tt.POS, r.CURR_LAYER); 
			break;
							
		case 2 :  
			key_out.write(r);
			rmsnorm_kernel(s_tokens, diff, weights, res_con, r.CURR_LAYER, tt.rms_ffn_W / sizeof(fdata_v_t));
			break;
							
		case 3 :  
			key_out.write(r);
			swiglu_kernel(s_tokens, diff); 
			break;
							
		case 4 :  
			key_out.write(r);
			rmsnorm_kernel(s_tokens, diff, weights, res_con, 0, tt.rms_final_W/ sizeof(fdata_v_t)); 
			break;
  }
}

//=============================================================

fsm_data weight_fsm(const axi_reg &tt, fsm_data &next_state){
  fsm_data curr_state;

  curr_state = next_state;
  // curr_state.gemv_config.CURR_LAYER = next_state.layer;
  next_state.state++;
  
  
  switch (curr_state.state) {
		case 0 :  
			//current GeMV dimensions:
			curr_state.gemv_config.N_DIM = MODEL_ELEMENTS; // 768 tokens
			curr_state.gemv_config.M_DIM = MODEL_ELEMENTS * 3; // QKV
			curr_state.gemv_config.w_sf = tt.QKV_sf_W / sizeof(mfdata_v_t);
			curr_state.gemv_config.w = tt.QKV_W / sizeof(idata_v_t);
			break;
							
		case 1 :  
			//current GeMV dimensions:
			curr_state.gemv_config.N_DIM = MODEL_ELEMENTS; // 768 tokens
			curr_state.gemv_config.M_DIM = MODEL_ELEMENTS; // Out
			curr_state.gemv_config.w_sf = tt.Out_sf_W / sizeof(mfdata_v_t);
			curr_state.gemv_config.w = tt.Out_W / sizeof(idata_v_t);
			break;
							
		case 2 :  
			//current GeMV dimensions:
			curr_state.gemv_config.N_DIM = MODEL_ELEMENTS; // 768 tokens
			curr_state.gemv_config.M_DIM = MODEL_HIDDEN_DIM * 2; // gate & up
			curr_state.gemv_config.w_sf = tt.FF_w1w3_sf_W / sizeof(mfdata_v_t);
			curr_state.gemv_config.w = tt.FF_w1w3_W / sizeof(idata_v_t);
			break;
							
		case 3 :  
			next_state.layer = ++next_state.gemv_config.CURR_LAYER;
			next_state.state = (next_state.layer == MODEL_NUM_LAYERS) ? 4 : 0;
			//current GeMV dimensions:
			curr_state.gemv_config.N_DIM = MODEL_HIDDEN_DIM; // 2048 tokens
			curr_state.gemv_config.M_DIM = MODEL_ELEMENTS; // down
			curr_state.gemv_config.w_sf = tt.FF_w2_sf_W / sizeof(mfdata_v_t);
			curr_state.gemv_config.w = tt.FF_w2_W / sizeof(idata_v_t);
			break;
							
		case 4 :  
			curr_state.gemv_config.N_DIM = MODEL_ELEMENTS; // 768 tokens
			curr_state.gemv_config.M_DIM = MODEL_TOKENS; // embeddings out
			curr_state.gemv_config.w_sf = tt.Embed_sf_W / sizeof(mfdata_v_t);
			curr_state.gemv_config.w = tt.Embed_W / sizeof(idata_v_t);
			curr_state.gemv_config.FINAL_FLAG = false;
			curr_state.gemv_config.CURR_LAYER = 0;
			break;
  }

  #ifdef __DEBUG__
    next_state.gemv_config.SAVE_ADDR += (curr_state.gemv_config.N_DIM / SM_FL_ELEM);
  #endif
  #ifdef __ULTRADEBUG__
    next_state.gemv_config.mmSAVE_ADDR += (curr_state.gemv_config.M_DIM / SM_FL_ELEM);
  #endif
  return curr_state;
}

void weights_df(s_idata_v_t (&s_w)[mm_thr], idata_v_t *w_0, idata_v_t *w_1, keys* r, const int CTRL_CNT){
// void weights_df(s_idata_v_t (&s_w)[mm_thr], idata_v_t *w_0, idata_v_t *w_1, const fsm_data* r, const int CTRL_CNT){
    
  Weight_df:
  for (int i = 0; i < CTRL_CNT; i++) {
    #pragma HLS DATAFLOW
    mm2s_input_data(s_w[0], w_0, r[i].wCount(), r[i].CURR_LAYER * mm_thr + 0, r[i].w);
    mm2s_input_data(s_w[1], w_1, r[i].wCount(), r[i].CURR_LAYER * mm_thr + 1, r[i].w);
  }
  
}


void calc_loop(  fdata_v_t *out, ProbIndex *ss_reg, 
        mfdata_v_t* wsf_0, mfdata_v_t* wsf_1, s_idata_v_t (&s_w)[mm_thr],
        s_fdata_v_t &s_cu_sel_in, hls::stream<keys> &vec_cnt
        #ifdef __DEBUG__
        , fdata_v_t *data_out
        #endif
        #ifdef __ULTRADEBUG__
        , fdata_v_t *GeMV_data_out
        #endif
        ){
	
  const keys r = vec_cnt.read();
  #pragma HLS DATAFLOW 
  // #pragma HLS INLINE
	s_mfdata_v_t s_wsf[mm_thr];
  s_idata_v_t s_tok_q, s_inf_tok_q[mm_thr];
  hls::stream<my_float_t> s_tok_sf, s_out[mm_thr];
  hls::stream<fdata_v_t> s_inf_tok_sf[mm_thr];
  hls::stream<ProbIndex> sys_sort;
	// hls::stream<bool> s_cu_sel_start("cu_selecter run signal");
  
  #pragma HLS STREAM variable=s_out depth=16 //MODEL_SCALING_FACTOR
  #pragma HLS BIND_STORAGE variable=s_out type=fifo //impl=bram
  #pragma HLS STREAM variable=s_tok_sf depth=4 //MODEL_HIDDEN_DIM/SM_FL_ELEM
  #pragma HLS STREAM variable=s_inf_tok_sf depth=4 //MODEL_HIDDEN_DIM/SM_FL_ELEM
  #pragma HLS STREAM variable=s_tok_q depth=4 //MODEL_HIDDEN_DIM/MAX_QUANT_ELEM
  #pragma HLS STREAM variable=s_inf_tok_q depth=8 //MODEL_HIDDEN_DIM/MAX_QUANT_ELEM
  #pragma HLS STREAM variable=sys_sort depth=64
	#pragma HLS STREAM variable=s_wsf depth=1024
	#pragma HLS BIND_STORAGE variable=s_wsf type=fifo impl=bram
	// #pragma HLS STREAM variable=s_cu_sel_start depth=1

	// #pragma HLS STREAM variable=s_w depth=4096*4
	// #pragma HLS BIND_STORAGE variable=s_w type=fifo impl=uram

	mm2s_input_data(s_wsf[0], wsf_0, r.sfCount(), r.CURR_LAYER * mm_thr + 0, r.w_sf);
	mm2s_input_data(s_wsf[1], wsf_1, r.sfCount(), r.CURR_LAYER * mm_thr + 1, r.w_sf);
	
  quantizer_kernel(s_tok_sf, s_tok_q, s_cu_sel_in, r.N_DIM
  #ifdef __DEBUG__
    , data_out, r.SAVE_ADDR
  #endif
  );

  inf_split_tee(s_inf_tok_sf, s_tok_sf, (r.N_DIM / (MODEL_SCALING_FACTOR * SM_FL_ELEM)));
  inf_split_tee(s_inf_tok_q, s_tok_q, (r.N_DIM / MAX_QUANT_ELEM));

  // s_GeMV_kernel(s_out[0], s_inf_tok_sf[0], s_inf_tok_q[0], s_wsf_0, s_w_0, r.N_DIM, r.M_DIM/2, r.CURR_LAYER * 2 + 0, 0, r.w_sf, r.w);
  // s_GeMV_kernel(s_out[1], s_inf_tok_sf[1], s_inf_tok_q[1], s_wsf_1, s_w_1, r.N_DIM, r.M_DIM/2, r.CURR_LAYER * 2 + 1, r.M_DIM/2, r.w_sf, r.w);
	
	for (int k = 0 ; k < mm_thr; k++) {
	#pragma HLS UNROLL
	s_GeMV_kernel(s_out[k], s_inf_tok_sf[k], s_inf_tok_q[k], s_wsf[k], s_w[k], r.N_DIM, r.M_DIM/2); }
  
  gemv_split(out, sys_sort, s_out, r.M_DIM, r.FINAL_FLAG
                #ifdef __ULTRADEBUG__
                  , GeMV_data_out, r.mmSAVE_ADDR
                #endif
                );
  insertion_sort(sys_sort, ss_reg, r.M_DIM);
}

void calc_fsm(fdata_v_t *tokens, fdata_v_t *weights, mfdata_v_t *key_cache, mfdata_v_t *value_cache, 
              mfdata_v_t *wsf_0, mfdata_v_t *wsf_1, s_idata_v_t (&s_w)[mm_thr],
							const int CTRL_CNT, hls::stream<fsm_data> &s_curr_fsm, const axi_reg &tt,
      #ifdef __DEBUG__
        const int CURR_LAYER, const int NEXT_STATE, fdata_v_t *data_out,
      #endif
      #ifdef __ULTRADEBUG__
        fdata_v_t *GeMV_data_out,
      #endif 
      const float_t temperature, int32_t *curr_token, const float_t coin, const bool rms_flag, const bool prefill_flag){
  
  const int RMS_SIZE = MODEL_ELEMENTS * (MODEL_NUM_LAYERS * 2 + 1);
  fdata_v_t internal_token[INTERNAL_DATA_SIZE/SM_FL_ELEM];
  fdata_v_t res_con[MODEL_ELEMENTS / SM_FL_ELEM]{};
  static fdata_v_t internal_rms_weights[RMS_SIZE / SM_FL_ELEM];
  ProbIndex ss_reg[REG_SIZE];
  float_t internal_coin;
  
  #pragma HLS ARRAY_PARTITION variable=internal_token dim=1 factor=2 type=block
  #pragma HLS BIND_STORAGE variable=internal_token type=ram_1p impl=bram
  #pragma HLS BIND_STORAGE variable=internal_rms_weights type=ram_1p impl=uram
  #pragma HLS ARRAY_PARTITION variable=ss_reg complete dim=1
  #pragma HLS BIND_STORAGE variable=res_con type=ram_2p
  
	// ===== IINITIALIZE RMS =====
  if (rms_flag) {
    // load weights into memory
    mm2mm_store(internal_rms_weights, weights, (MODEL_ELEMENTS * (MODEL_NUM_LAYERS * 2 + 1)));
  }
	// int ct = *curr_token;
  // ===== INITIALIZE TOKENS =====
	int ct = curr_token[tt.POS];
  mm2mm_store(internal_token, tokens, MODEL_ELEMENTS, 2, 0, INTERNAL_DATA_SIZE, ct * (int32_t) (MODEL_ELEMENTS / SM_FL_ELEM));
  mm2mm_store(internal_token, tokens, MODEL_ELEMENTS, 2, 1, INTERNAL_DATA_SIZE, ct * (int32_t) (MODEL_ELEMENTS / SM_FL_ELEM));

  for(int ii = 0; ii < CTRL_CNT; ii++) {
    s_fdata_v_t s_cu_sel_out;
    hls::stream<keys> vec_cnt;
    #pragma HLS STREAM variable=s_cu_sel_out depth=MODEL_HIDDEN_DIM/SM_FL_ELEM
    
    cu_selecter(s_cu_sel_out, vec_cnt, internal_rms_weights, internal_token, key_cache, value_cache, res_con, s_curr_fsm, tt);
    calc_loop(internal_token, ss_reg, wsf_0, wsf_1, s_w, s_cu_sel_out, vec_cnt
      #ifdef __DEBUG__
      , data_out
      #endif
      #ifdef __ULTRADEBUG__
      , GeMV_data_out
      #endif
    );
  }
    // ss_final(ss_reg, tokens, temperature, 0.9, coin);
		ss_final(ss_reg, ct, temperature, 0.9, coin);
		if (!prefill_flag) {
			curr_token[tt.POS + 1] = ct;
		}
		
}

// void init(const axi_reg tt, )
/* ================ TRANSFORMER KERNEL ================ TRANSFORMER KERNEL ================ TRANSFORMER KERNEL ================ TRANSFORMER KERNEL ================ */ 

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
        const int rms_att_W, const int rms_ffn_W, const int rms_final_W, int *curr_token,
      #ifdef __DEBUG__
        const int faker, const int CURR_LAYER, const int NEXT_STATE, fdata_v_t *data_out,
      #endif
      #ifdef __ULTRADEBUG__
        fdata_v_t *GeMV_data_out,
      #endif
        const float temperature, const float coin,
        const bool init_rms_flag, const bool prefill_flag){
  
  
  constexpr int q_size = (MODEL_ELEMENTS * ((MODEL_ELEMENTS * 4 + MODEL_HIDDEN_DIM * 3 ) * MODEL_NUM_LAYERS + MODEL_TOKENS)) * sizeof(int8_t);
  constexpr int rms_size = (MODEL_ELEMENTS * (MODEL_NUM_LAYERS * 2 + 1)) * sizeof(my_float_t);
  constexpr int sf_size = (q_size * sizeof(my_float_t) / (sizeof(int8_t) * MODEL_SCALING_FACTOR));
  
  constexpr int RMS_DEPTH = MODEL_ELEMENTS * (MODEL_NUM_LAYERS * 2 + 1) / SM_FL_ELEM;
  constexpr int CACHE_DEPTH = MODEL_ELEMENTS * MODEL_SEQUENCE_LEN * MODEL_NUM_LAYERS / MAX_FL_ELEM;
  constexpr int TOK_DEPTH = MODEL_ELEMENTS / sizeof(idata_v_t);
  constexpr int HD_QUANT_DEPTH = q_size / MAX_QUANT_ELEM;
  constexpr int HD_SF_DEPTH = sf_size / sizeof(mfdata_v_t); 
  constexpr int TOK_OUT_DEPTH = INTERNAL_DATA_SIZE / SM_FL_ELEM;
  constexpr int nTOK_OUT_DEPTH = MODEL_TOKENS * MODEL_ELEMENTS /SM_FL_ELEM;
  constexpr int MHA_DEPTH = MODEL_ELEMENTS / MID_FL_ELEM * 3;
  constexpr int RECORD_DEPTH = ((MODEL_ELEMENTS * 3  + MODEL_HIDDEN_DIM) * 12 + MODEL_ELEMENTS) / SM_FL_ELEM;

  #pragma HLS INTERFACE mode=m_axi port=tokens         bundle=w_n_t_gemm     depth=nTOK_OUT_DEPTH   offset=slave max_write_burst_length=16 max_read_burst_length=(4096/SM_DW*8)
  #pragma HLS INTERFACE mode=m_axi port=w_sf_0         bundle=D_TOK_W_SF_0     depth=HD_SF_DEPTH     offset=slave max_read_burst_length=(1024/MAX_DW * 8)    num_read_outstanding=4
  #pragma HLS INTERFACE mode=m_axi port=w_0           bundle=D_W_GEMM_0     depth=HD_QUANT_DEPTH   offset=slave max_read_burst_length=(4096/MAX_DW * 8)     num_read_outstanding=64 
  #pragma HLS INTERFACE mode=m_axi port=w_sf_1         bundle=D_TOK_W_SF_1       depth=HD_SF_DEPTH     offset=slave max_read_burst_length=(1024/MAX_DW * 8)    num_read_outstanding=4
  #pragma HLS INTERFACE mode=m_axi port=w_1           bundle=D_W_GEMM_1     depth=HD_QUANT_DEPTH   offset=slave max_read_burst_length=(4096/MAX_DW * 8)     num_read_outstanding=64 
  #pragma HLS INTERFACE mode=m_axi port=weights        bundle=w_n_t_gemm     depth=RMS_DEPTH        offset=slave max_read_burst_length=(4096/SM_DW * 8)
  #pragma HLS INTERFACE mode=m_axi port=value_cache    bundle=vc_gemm        depth=CACHE_DEPTH      offset=slave max_read_burst_length=(4096/MAX_DW * 8)  max_write_burst_length=(512/MAX_DW * 8)	num_read_outstanding=64 
  #pragma HLS INTERFACE mode=m_axi port=key_cache      bundle=kc_gemm        depth=CACHE_DEPTH      offset=slave max_read_burst_length=(4096/MAX_DW * 8)  max_write_burst_length=(512/MAX_DW * 8)	num_read_outstanding=64 
	#pragma HLS INTERFACE mode=m_axi port=curr_token 			bundle=token_val depth=MODEL_SEQUENCE_LEN offset=slave max_read_burst_length=1 max_write_burst_length=1 num_read_outstanding=1 num_write_outstanding=1

  #pragma HLS INTERFACE mode=s_axilite port=tokens       bundle=control
  #pragma HLS INTERFACE mode=s_axilite port=w_sf_0       bundle=control
  #pragma HLS INTERFACE mode=s_axilite port=w_0         bundle=control
  #pragma HLS INTERFACE mode=s_axilite port=w_sf_1       bundle=control
  #pragma HLS INTERFACE mode=s_axilite port=w_1         bundle=control
  #pragma HLS INTERFACE mode=s_axilite port=weights      bundle=control
  #pragma HLS INTERFACE mode=s_axilite port=value_cache     bundle=control
  #pragma HLS INTERFACE mode=s_axilite port=key_cache      bundle=control
  #pragma HLS INTERFACE mode=s_axilite port=POS         bundle=control
  #pragma HLS INTERFACE mode=s_axilite port=QKV_W        bundle=control
  #pragma HLS INTERFACE mode=s_axilite port=QKV_sf_W      bundle=control
  #pragma HLS INTERFACE mode=s_axilite port=Out_W        bundle=control
  #pragma HLS INTERFACE mode=s_axilite port=Out_sf_W      bundle=control
  #pragma HLS INTERFACE mode=s_axilite port=FF_w1w3_W      bundle=control
  #pragma HLS INTERFACE mode=s_axilite port=FF_w1w3_sf_W    bundle=control
  #pragma HLS INTERFACE mode=s_axilite port=FF_w2_W      bundle=control
  #pragma HLS INTERFACE mode=s_axilite port=FF_w2_sf_W     bundle=control
  #pragma HLS INTERFACE mode=s_axilite port=Embed_W      bundle=control
  #pragma HLS INTERFACE mode=s_axilite port=Embed_sf_W     bundle=control
  #pragma HLS INTERFACE mode=s_axilite port=rms_att_W      bundle=control
  #pragma HLS INTERFACE mode=s_axilite port=rms_ffn_W      bundle=control
  #pragma HLS INTERFACE mode=s_axilite port=rms_final_W    bundle=control
  #pragma HLS INTERFACE mode=s_axilite port=temperature    bundle=control
  #pragma HLS INTERFACE mode=s_axilite port=coin        bundle=control
  #pragma HLS INTERFACE mode=s_axilite port=init_rms_flag    bundle=control
  #pragma HLS INTERFACE mode=s_axilite port=prefill_flag    bundle=control
  #pragma HLS INTERFACE mode=s_axilite port=curr_token    bundle=control
  #pragma HLS INTERFACE mode=s_axilite port=return        bundle=control
  
  #ifdef __DEBUG__
  #pragma HLS INTERFACE mode=m_axi port=data_out         bundle=w_n_t_gemm     depth=RECORD_DEPTH   offset=slave max_write_burst_length=16 max_read_burst_length=(4096/SM_DW*8)
    #pragma HLS INTERFACE mode=s_axilite port=faker       bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=CURR_LAYER   bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=NEXT_STATE   bundle=control
    #pragma HLS INTERFACE mode=s_axilite port=data_out     bundle=control
  #endif
  #ifdef __ULTRADEBUG__
  #pragma HLS INTERFACE mode=m_axi port=GeMV_data_out         bundle=w_n_t_gemm     depth=RECORD_DEPTH   offset=slave max_write_burst_length=16 max_read_burst_length=(4096/SM_DW*8)
    #pragma HLS INTERFACE mode=s_axilite port=GeMV_data_out    bundle=control
  #endif
  
  /* ========== INITIALIZATION ========== INITIALIZATION ========== INITIALIZATION ========== INITIALIZATION ========== INITIALIZATION */
  
	// ===== AXI REGISTER KEYS ===
	const axi_reg tt = {
    POS, 
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
    rms_final_W };

	// ====== FSM LOOP BUILDER======

  #ifndef __DEBUG__
    const int CTRL_CNT = (prefill_flag) ? (MODEL_NUM_LAYERS * 4 - 2) : (MODEL_NUM_LAYERS * 4 + 1);
  #endif
  #ifdef __DEBUG__
  const int CTRL_CNT = faker;
  #endif
	
	// int ct = *curr_token;

  fsm_data fd;
  // fsm_data curr_state[MODEL_NUM_LAYERS * 4 + 1];
	hls::stream<fsm_data> s_curr_state("current fsm");
	#pragma HLS STREAM variable=s_curr_state depth = (MODEL_NUM_LAYERS * 4 + 1)
	keys curr_keys[(MODEL_NUM_LAYERS * 4 + 1)];
	#pragma HLS BIND_STORAGE variable=curr_keys type=ram_2p impl=bram
	
  WL_fsm:
  for (int i = 0; i < CTRL_CNT; i++) {
    #pragma HLS PIPELINE 
		fsm_data tmp_fsm = weight_fsm(tt, fd);
    s_curr_state.write(tmp_fsm);
		curr_keys[i] = tmp_fsm.gemv_config;
		
  }

	/* ========== INITIALIZATION ========== INITIALIZATION ========== INITIALIZATION ========== INITIALIZATION ========== INITIALIZATION */


	/* ========== DATAFLOW ========== DATAFLOW ========== DATAFLOW ========== DATAFLOW ========== DATAFLOW ========== DATAFLOW ========== DATAFLOW */
  #pragma HLS DATAFLOW
	s_idata_v_t s_w[mm_thr];
	#pragma HLS STREAM variable=s_w	depth=4096*4
	#pragma HLS BIND_STORAGE variable=s_w type=fifo impl=uram
	weights_df(s_w, w_0, w_1, curr_keys, CTRL_CNT);
	calc_fsm(tokens, weights, key_cache, value_cache, w_sf_0, w_sf_1, s_w, CTRL_CNT, s_curr_state, tt,
      #ifdef __DEBUG__
        CURR_LAYER, NEXT_STATE, data_out,
      #endif
      #ifdef __ULTRADEBUG__
        GeMV_data_out,
      #endif 
			 temperature, curr_token, coin, init_rms_flag, prefill_flag);
		return;
}
