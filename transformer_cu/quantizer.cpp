
#include "quantizer.h"
#include "mha_forward.h"
#include <sys/types.h>


void debug_abs_intake(s_fdata_v_t &tokens_out, s_fdata_v_t &abs_tokens, s_fdata_v_t &tokens_in, 
                    fdata_v_t *data_out, const int save_addr, const int SF_COUNT){
  
  const size_t TOK_COUNT = MODEL_SCALING_FACTOR / SM_FL_ELEM;
  my_float_t max_val = 0.0f;
  
  group_scaling:
  for (size_t j = 0; j < TOK_COUNT; j++) {
#pragma HLS PIPELINE II=1
    
    fdata_v_t val = tokens_in.read();
    tokens_out.write(val);
    data_out[SF_COUNT * TOK_COUNT + save_addr + j] = val;
    
    fdata_v_t c_val;
    for (int k = 0; k < SM_FL_ELEM; k++) {
      c_val[k] = hls::absf(val[k]);
    }		
    abs_tokens.write(c_val);
  }
  // can probably get rid of abs_intake
}

void abs_intake(s_fdata_v_t &tokens_out, s_fdata_v_t &abs_tokens, s_fdata_v_t &tokens_in){
  
  const size_t TOK_COUNT = MODEL_SCALING_FACTOR / SM_FL_ELEM;
  my_float_t max_val = 0.0f;
  
  group_scaling:
  for (size_t j = 0; j < TOK_COUNT; j++) {
    #pragma HLS PIPELINE II=1
    
    fdata_v_t val = tokens_in.read();
    tokens_out.write(val);
    
    fdata_v_t c_val;
    for (int k = 0; k < SM_FL_ELEM; k++) {
      c_val[k] = hls::absf(val[k]);
    }		
    abs_tokens.write(c_val);
  }
  // can probably get rid of abs_intake
}

void max_finder(hls::stream<my_float_t> &max_val, s_fdata_v_t & abs_tokens){
  
  const my_float_t Q_MAX = 1.0f / 127.0f;
  const int cnt = MODEL_SCALING_FACTOR / SM_FL_ELEM;
  my_float_t c_val[MODEL_SCALING_FACTOR];
  #pragma HLS ARRAY_PARTITION variable=c_val dim=1 type=complete
  //here we store token_out and then assign token_out[i] to c_val[i * MAX_FL_ELEM + k] = hls::absf(token_out[i][k])
  
  
  mf_intake:
  for (int i = 0; i < cnt; i++) {
    #pragma HLS PIPELINE II=1
    fdata_v_t val = abs_tokens.read();
    for (int k = 0; k < SM_FL_ELEM; k++) {
      c_val[i * SM_FL_ELEM + k] = val[k];
    }
  }

  for (int stride = (MODEL_SCALING_FACTOR>>1); stride > 0; stride >>=1) {
    #pragma HLS UNROLL
    for (int i = 0; i < stride; i++) {
      #pragma HLS UNROLL
      c_val[i] = (c_val[i]  > c_val[i + stride] ) ? c_val[i] : c_val[i + stride];
    }
  }	
  max_val.write(c_val[0] * Q_MAX);
}


void quant_out( hls::stream<my_float_t> &tok_sf_out, s_idata_v_t &tok_out, s_fdata_v_t &tokens_in, hls::stream<my_float_t> &max_val){
  
  const size_t TOK_COUNT = MODEL_SCALING_FACTOR / SM_FL_ELEM;
  my_float_t dscale = max_val.read(); 
  my_float_t scale = hls::recipf(dscale);//Q_MAX / max_val;
  idata_v_t quant_tmp; // not an array anymore
  
  create_q_val:
  for (size_t j = 0; j < TOK_COUNT; j++) {
    #pragma HLS PIPELINE
    fdata_v_t proc_tok = tokens_in.read();
    
    create_q_val_ppl:	
    for (size_t k = 0; k < SM_FL_ELEM; k++) {
      #pragma HLS UNROLL
      quant_tmp[j * SM_FL_ELEM + k] = (my_quant_data_t) hls::roundf(proc_tok[k] * scale);
    }
  }
  
  tok_out.write(quant_tmp);
  tok_sf_out.write(dscale);
}

void quantizer_kernel(hls::stream<my_float_t>  &tok_sf_out, s_idata_v_t &tok_out, s_fdata_v_t &tokens, const int N_DIM){
  
  const size_t SF_COUNT = N_DIM / MODEL_SCALING_FACTOR;
  const size_t TOK_COUNT = MODEL_SCALING_FACTOR / SM_FL_ELEM;
  
  for (int i = 0; i < SF_COUNT; i++) {
    #pragma HLS LOOP_TRIPCOUNT max=MODEL_HIDDEN_DIM / MODEL_SCALING_FACTOR min=MODEL_ELEMENTS / MODEL_SCALING_FACTOR
    #pragma HLS DATAFLOW
    hls::stream<my_float_t> max_val;
    s_fdata_v_t tokens_out, abs_tokens;
    #pragma HLS STREAM variable=tokens_out depth=64
    #pragma HLS STREAM variable=max_val depth=4 //TOK_COUNT
    #pragma HLS STREAM variable=abs_tokens depth=64
    
    abs_intake(tokens_out, abs_tokens, tokens);
    max_finder(max_val, abs_tokens);
    quant_out(tok_sf_out, tok_out, tokens_out, max_val);
  }
}

void quantizer_kernel(hls::stream<my_float_t>  &tok_sf_out, s_idata_v_t &tok_out, s_fdata_v_t &tokens, const int N_DIM, 
                      fdata_v_t *data_out, const int SAVE_ADDR){
                        
  const size_t SF_COUNT = N_DIM / MODEL_SCALING_FACTOR;
  const size_t TOK_COUNT = MODEL_SCALING_FACTOR / SM_FL_ELEM;
  // #pragma HLS STREAM variable=tok_out depth=64
  // #pragma HLS STREAM variable=tok_sf_out depth=64
  for (int i = 0; i < SF_COUNT; i++) {
    #pragma HLS LOOP_TRIPCOUNT max=MODEL_HIDDEN_DIM / MODEL_SCALING_FACTOR min=MODEL_ELEMENTS / MODEL_SCALING_FACTOR
    #pragma HLS DATAFLOW
    hls::stream<my_float_t> max_val;
    s_fdata_v_t tokens_out, abs_tokens;
    #pragma HLS STREAM variable=tokens_out depth=64
    #pragma HLS STREAM variable=max_val depth=4 //TOK_COUNT
    #pragma HLS STREAM variable=abs_tokens depth=64
    
    debug_abs_intake(tokens_out, abs_tokens, tokens, data_out, SAVE_ADDR, i);
    max_finder(max_val, abs_tokens);
    quant_out(tok_sf_out, tok_out, tokens_out, max_val);
  }
}


void dequantize_kernel(fdata_v_t* internal_token, idata_v_t* tokq, fdata_v_t* toksf, const int curr_token, const int wcls_offset_q, const int wcls_sf){
	
  const int ct_ratio = MODEL_NUM_HEADS / SM_FL_ELEM;
  const int OFFSET = wcls_sf / sizeof(fdata_v_t) + curr_token * ct_ratio;
  // const int mod_off = curr_token % 4;
  const int QUANT_OFF = curr_token * MODEL_ELEMENTS / MAX_QUANT_ELEM + (wcls_offset_q / sizeof(idata_v_t));
  
  fdata_v_t tmp_sf[ct_ratio]; // JUSTIFY WITH RIGHT NUMBERS LATER
  
  
  for (int i = 0 ; i < (ct_ratio); i++) {
    #pragma HLS PIPELINE II=1
    tmp_sf[i] = toksf[OFFSET + i];
  }
  
  // int internal_offset = 0;
  // int internal_cnt = 0;
  fdata_v_t tmpc;
  for (int i = 0; i < MODEL_NUM_HEADS; i++) {
    int jj = i % SM_FL_ELEM;
    if (jj == 0) { tmpc = tmp_sf[i/SM_FL_ELEM]; }
    
    int internal_offset = (i < 6 ) ? 0 : INTERNAL_DATA_SIZE / (SM_FL_ELEM * 2) - 6 * MAX_QUANT_ELEM / SM_FL_ELEM;
    
    my_float_t ftmp = tmpc[jj];
    idata_v_t itmp = tokq[i + QUANT_OFF];
    int baseaddr = i * (MAX_QUANT_ELEM/SM_FL_ELEM) + internal_offset;
    
    for (int k = 0; k < MAX_QUANT_ELEM / SM_FL_ELEM; k++) {
      #pragma HLS PIPELINE II=1
      fdata_v_t tmpo;
      
      for (int ii = 0; ii < SM_FL_ELEM; ii++) {
        #pragma hls UNROLL
        tmpo[ii] = itmp[ii] * ftmp;
      }
      
      for (int ii = 0; ii < (MAX_QUANT_ELEM - SM_FL_ELEM); ii++) {
        #pragma HLS UNROLL
        itmp[ii] = itmp[ii + SM_FL_ELEM];
      }
      
      internal_token[baseaddr + k] = tmpo;
    }
  }
}