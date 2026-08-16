#include "rmsnorm.h"
#include "mha_forward.h"

void rms_mm2s_data(s_fdata_v_t &out, fdata_v_t *in, const int cnt){
  // #pragma HLS PIPELINE off
  mm2s_input_data(out, in, cnt/2, 0, 0 );
  mm2s_input_data(out, in, cnt/2, 0, (INTERNAL_DATA_SIZE / (SM_FL_ELEM * 2)));
}

void rmsnorm_kernel(s_fdata_v_t &s_tokens_out, fdata_v_t *diff, fdata_v_t *weights, fdata_v_t *res_con, const int CURR_LAYER, const int offset){

  #pragma HLS DATAFLOW
  s_fdata_v_t s_weights, s_tokens, s_diff;
  const int ratio = MODEL_ELEMENTS / SM_FL_ELEM;
  const int bram_depth = 64;
  
  #pragma HLS STREAM variable=s_diff depth=bram_depth
  #pragma HLS STREAM variable=s_weights depth=ratio
  
  #pragma HLS STREAM variable=s_tokens depth=ratio
  #pragma HLS STREAM variable=s_tokens_out depth=ratio
  
  #pragma HLS BIND_STORAGE variable=s_tokens type=fifo impl=bram
  #pragma HLS BIND_STORAGE variable=s_diff type=fifo impl=bram
  #pragma HLS BIND_STORAGE variable=s_tokens_out type=fifo impl=bram
  #pragma HLS BIND_STORAGE variable=s_weights type=fifo impl=bram
  
  rms_mm2s_data(s_diff, diff, ratio);
  mm2s_input_data(s_weights, weights, ratio, CURR_LAYER, offset);
  rmsnorm(s_tokens_out, s_diff, res_con, s_weights);
    
  return;
}