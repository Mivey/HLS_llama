#include "../transformer_cu/mha_forward.h"



constexpr int weight_fsm(int n){

  int total = 0;
  for (int i = 0; i < n; i++) {
    int t = (i == (n - 1)) ? 4 : i % 4;
    switch (t) {
      case 0 :  
        total += MODEL_ELEMENTS * 3;
        break;
                
      case 1 :  
        total += MODEL_ELEMENTS;
        break;
                
      case 2 :  
        total += MODEL_HIDDEN_DIM * 2;
        break;
                
      case 3 :  
        total += MODEL_HIDDEN_DIM;
        break;
                
      case 4 :  
        total +=  MODEL_TOKENS;
        break;
    }
  }
  return (total * MODEL_ELEMENTS * 17 / (16 * 64));
}



void data_mover_cu(s_wide_t &s_w_0, s_wide_t &s_w_1, s_wide_t &s_sf_0, s_wide_t &s_sf_1, const wide_t *w_0, const wide_t *w_1, 
                  const bool prefill_flag){
  
  constexpr int q_size = (MODEL_ELEMENTS * ((MODEL_ELEMENTS * 4 + MODEL_HIDDEN_DIM * 3 ) * MODEL_NUM_LAYERS + MODEL_TOKENS)) * sizeof(int8_t);
  constexpr int sf_size = (q_size * sizeof(my_float_t) / (sizeof(int8_t) * MODEL_SCALING_FACTOR));
  constexpr int HD_QUANT_DEPTH = q_size / MAX_QUANT_ELEM;
  constexpr int HD_SF_DEPTH = sf_size / sizeof(mfdata_v_t); 
  constexpr int WIDE_DEPTH = HD_SF_DEPTH + HD_QUANT_DEPTH;

  
  #pragma HLS INTERFACE mode=axis   port=s_w_0    depth=4096            name=Quantized_Data_0
  #pragma HLS INTERFACE mode=axis   port=s_w_1    depth=4096            name=Quantized_Data_1
  #pragma HLS INTERFACE mode=axis   port=s_sf_0   depth=256             name=Scaling_Factor_0
  #pragma HLS INTERFACE mode=axis   port=s_sf_1   depth=256             name=Scaling_Factor_1
  #pragma HLS INTERFACE mode=m_axi  port=w_0      bundle=D_W_GEMM_0     depth=WIDE_DEPTH latency=128 max_read_burst_length=(4096/MAX_DW*8) max_write_burst_length=1 num_read_outstanding=64 num_write_outstanding=1 offset=slave
  #pragma HLS INTERFACE mode=m_axi  port=w_1      bundle=D_W_GEMM_1     depth=WIDE_DEPTH latency=128 max_read_burst_length=(4096/MAX_DW*8) max_write_burst_length=1 num_read_outstanding=64 num_write_outstanding=1 offset=slave
  

  #pragma HLS INTERFACE mode=s_axilite port=w_0         bundle=control
  #pragma HLS INTERFACE mode=s_axilite port=w_1         bundle=control
  // #pragma HLS INTERFACE mode=s_axilite port=w_0_offset      bundle=control
  // #pragma HLS INTERFACE mode=s_axilite port=w_1_offset     bundle=control
  #pragma HLS INTERFACE mode=s_axilite port=prefill_flag    bundle=control
  #pragma HLS INTERFACE mode=s_axilite port=return        bundle=control
  constexpr int PREFILL_CNT = weight_fsm((MODEL_NUM_LAYERS * 4 - 2));
  constexpr int DECODE_CNT = weight_fsm((MODEL_NUM_LAYERS * 4 + 1));
  
  const int CTRL_CNT = (prefill_flag) ? PREFILL_CNT : DECODE_CNT;
  const int N_COUNT = CTRL_CNT / TXFR_FRAME;

  #pragma HLS DATAFLOW
  s_wide_t dout0, dout1;
  #pragma HLS STREAM variable=dout0 depth=4096
  #pragma HLS BIND_STORAGE variable=dout0 type=fifo impl=uram
  #pragma HLS STREAM variable=dout1 depth=4096
  #pragma HLS BIND_STORAGE variable=dout1 type=fifo impl=uram
  mm2s_input_data(dout0, w_0, CTRL_CNT);
  mm2s_input_data(dout1, w_1, CTRL_CNT);
  mm2ds_split_data(s_sf_0, s_w_0, dout0, N_COUNT);
  mm2ds_split_data(s_sf_1, s_w_1, dout1, N_COUNT);
  
}