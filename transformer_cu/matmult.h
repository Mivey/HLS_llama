
#ifndef MARK_MM
#define MARK_MM
#include "mha_forward.h"
#include <cmath>
#include <cstddef>
#include <hls_vector.h>
void GeMV_kernel(hls::stream<my_float_t> &out, hls::stream<fdata_v_t>  &tok_sf, s_idata_v_t &tok_q, mfdata_v_t *w_sf, idata_v_t *w, const int N_DIM, const int M_DIM, const int CURR_LAYER, const int W_Off, const int sf_reg, const int w_reg);
void s_GeMV_kernel(hls::stream<my_float_t> &out, s_fdata_v_t &tok_sf, s_idata_v_t &tok_q, s_mfdata_v_t &s_wsf, idata_v_t* w, const int N_DIM, const int M_DIM, const int CURR_LAYER, const int W_Off, const int sf_reg, const int w_reg);

constexpr size_t TOK_QUANT_MAX =  (MODEL_HIDDEN_DIM / MAX_QUANT_ELEM);
constexpr size_t TOK_SF_MAX = (MODEL_HIDDEN_DIM / MODEL_SCALING_FACTOR);


template<typename T, size_t M>
void GeMV_PE(hls::stream<float_t> &qout, hls::stream<hls::vector<T, M>> &w, hls::stream<hls::vector<T, M>> &tok, const int N_DIM, const int M_DIM){
  const int num_loop = N_DIM / M;
  typedef hls::vector<T, M> tmp_t;
  
  tmp_t arr[TOK_QUANT_MAX];
#pragma HLS BIND_STORAGE variable=arr type=fifo impl=srl
  
  amm_tok:
  for (size_t i = 0; i < num_loop; i++){ // vCount here is 1/4 vCount in send_wtok!!
    #pragma HLS PIPELINE II=1
    #pragma HLS LOOP_TRIPCOUNT max = TOK_SF_MAX min=MODEL_ELEMENTS/(MODEL_SCALING_FACTOR * SM_FL_ELEM )  
    arr[i] = tok.read();

  }
  for (size_t k = 0; k < M_DIM; k++) {
    #pragma HLS LOOP_FLATTEN
    
    PE_quant:
    for (size_t j = 0; j < num_loop; j++) {
      #pragma HLS PIPELINE II=1
      tmp_t curr_tok = arr[j];
      tmp_t curr_w = w.read();
      
      int32_t prod{}; 
      for (size_t i = 0; i < M; i++) {
        #pragma HLS UNROLL
        prod += (int32_t) curr_tok[i] * curr_w[i];
      }

      float fprod = (float_t) prod;
      qout.write(fprod);
    }
  }
}

template<size_t N, typename T>
void GeMV_PE_sf(hls::stream<float_t> &sfout, hls::stream<hls::vector<T, N>> &w_sf, hls::stream<hls::vector<T, N>> &tok_sf, const int N_DIM, const int M_DIM){
  
  const int num_loop = N_DIM * M_DIM / (N * MODEL_SCALING_FACTOR);
  typedef hls::vector<T, N>	tmp_t;

  tmp_t arr_sf[TOK_SF_MAX];
#pragma HLS BIND_STORAGE variable=arr_sf type=fifo impl=srl

  amm_tok_sf:
  for (size_t i = 0; i < num_loop; i++){ // vCount here is 1/4 vCount in send_wtok!!
    #pragma HLS PIPELINE II=1
    #pragma HLS LOOP_TRIPCOUNT max = TOK_SF_MAX min=MODEL_ELEMENTS/(MODEL_SCALING_FACTOR * SM_FL_ELEM )  
    arr_sf[i] = tok_sf.read();
  }
  
  for (size_t k = 0; k < M_DIM; k++) {
    
    for (size_t j = 0; j < num_loop; j++) {
      #pragma HLS LOOP_FLATTEN
      tmp_t curr_tok = arr_sf[j] * w_sf.read();
      PE_sf:
      for (size_t i = 0; i < N; i++) {
        #pragma HLS PIPELINE II=1
        sfout.write(curr_tok[i]);
      }
    }
  }
}
template<size_t N, typename T>
void GeMV_PE_sf(hls::stream<float_t> &sfout, hls::stream<hls::vector<T, N>> &w_sf, hls::stream<float_t> &tok_sf, const int N_DIM, const int M_DIM){
  
  const int num_loop = N_DIM / (N * MODEL_SCALING_FACTOR);
  typedef hls::vector<T, N>	tmp_t;

  float_t arr_sf[TOK_SF_MAX];
#pragma HLS BIND_STORAGE variable=arr_sf type=fifo impl=srl

  amm_tok_sf:
  for (size_t i = 0; i < (num_loop * N); i++){ // vCount here is 1/4 vCount in send_wtok!!
    #pragma HLS PIPELINE II=1
    #pragma HLS LOOP_TRIPCOUNT max = TOK_SF_MAX min=MODEL_ELEMENTS/(MODEL_SCALING_FACTOR )  
    arr_sf[i] = tok_sf.read();
  }
  // tmp_t curr_w = w_sf.read();
  
  for (size_t k = 0; k < M_DIM; k++) {
    
    for (size_t j = 0; j < num_loop; j++) {
      #pragma HLS LOOP_FLATTEN
      tmp_t curr_tok = w_sf.read();
      PE_sf:
      for (size_t i = 0; i < N; i++) {
        #pragma HLS PIPELINE
        float_t tmpa = arr_sf[j * N + i] * curr_tok[i];
        sfout.write(tmpa);
      }
    }
  }
}

#endif
