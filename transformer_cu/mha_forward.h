
#ifndef MARK_FORWARD
#define MARK_FORWARD

#include <cmath>
// #define __DEBUG__
// #define __ULTRADEBUG__

#include <cstddef>
#include <cstdint>
#include <hls_stream.h>
#include <hls_math.h>
#include <hls_vector.h>
#include "hls_fence.h"
#include <ap_int.h>
#include <utils/x_hls_defines.h>

#define MODEL_ELEMENTS 768
#define MODEL_HIDDEN_DIM 2048
#define MODEL_NUM_HEADS 12
#define MODEL_NUM_LAYERS 12
#define MODEL_TOKENS 32000
#define MODEL_SEQUENCE_LEN 1024
#define MODEL_SCALING_FACTOR 64
// const int MODEL_RMS_SIZE =;
#define bytes_in(n) sizeof(n)
#define runs(n) SCALING_FACTOR/sizeof(n)

/* ************************************* */
struct fast_bf16{
  ap_uint<16> bits;
  
  fast_bf16() {}

  // convert float to bf16
  fast_bf16(float f) {
    #pragma HLS INLINE
    ap_uint<32> float_bits = reinterpret_cast<ap_uint<32>&>(f);
    bits = float_bits(31, 16);
  }

  operator float() const {
    #pragma HLS INLINE
    ap_uint<32> float_bits = (bits, ap_uint<16>(0));
    return reinterpret_cast<float&>(float_bits);
  }
};
// typedef ap_float<32, 8> my_float_t;
typedef float my_float_t;
// typedef fast_bf16 my_float_t;
typedef int8_t my_quant_data_t;
/* ************************************* */

constexpr size_t MAX_DW = 512;
constexpr size_t MID_DW = 512;
constexpr size_t QUANT_MODIFIER = 1;//(MAX_DW == 512) ? 2 : 1;
constexpr size_t SM_DW = 128;
constexpr size_t MAX_FL_ELEM = (MAX_DW / (sizeof(my_float_t) * 8));
constexpr size_t MID_FL_ELEM = (MID_DW / (sizeof(my_float_t) * 8));
constexpr size_t MAX_QUANT_ELEM = ((MAX_DW / QUANT_MODIFIER) / (sizeof(my_quant_data_t) * 8));
constexpr size_t SM_FL_ELEM = (SM_DW / (sizeof(my_float_t) * 8));
constexpr size_t SM_QUANT_ELEM = (SM_DW / (sizeof(my_quant_data_t) * 8));

constexpr int MODEL_HEAD_SIZE = MODEL_ELEMENTS / MODEL_NUM_HEADS;
// #define MAX_W 512
constexpr int MAX_W_Q = MAX_DW/(sizeof(my_quant_data_t) * 8);
constexpr int MAX_W_F = MAX_DW/(sizeof(my_float_t) * 8);
constexpr int MAX_SF_W_F = MAX_DW/(sizeof(my_float_t) * 4 * 8);
constexpr int TOK_CHUNKSIZE = 256;
constexpr int MM_CHUNKSIZE = 256;
constexpr int MHA_CHUNKSIZE = 64;
constexpr int INTERNAL_DATA_SIZE = MODEL_HIDDEN_DIM * 2;
constexpr int RMS_SIZE = (MODEL_ELEMENTS * (MODEL_NUM_LAYERS * 2 + 1));

const int SQUARE_TOK = MODEL_ELEMENTS * MODEL_ELEMENTS;
const int SQUARE_SF = SQUARE_TOK / MODEL_SCALING_FACTOR;
const int RECT_TOK = MODEL_ELEMENTS * MODEL_HIDDEN_DIM;
const int RECT_SF = RECT_TOK / MODEL_SCALING_FACTOR;

/* ==================================================================================== */

typedef hls::vector<my_quant_data_t, MAX_QUANT_ELEM> idata_v_t;
typedef hls::vector<my_float_t, SM_FL_ELEM>  fdata_v_t;
typedef hls::vector<my_float_t, MAX_FL_ELEM>  mfdata_v_t;
typedef hls::vector<my_float_t, MID_FL_ELEM>  adata_v_t;

typedef hls::stream<idata_v_t> s_idata_v_t;
typedef hls::stream<fdata_v_t> s_fdata_v_t; 
typedef hls::stream<mfdata_v_t> s_mfdata_v_t;
typedef hls::stream<adata_v_t> s_adata_v_t;

template<typename T, int N>
void inf_split_tee(hls::stream<T> (&out)[N], hls::stream<T> &in, const int vCount){
  
  for (int i = 0; i < vCount; i++) {
    #pragma HLS LOOP_TRIPCOUNT max= (MODEL_HIDDEN_DIM / MAX_QUANT_ELEM)
    #pragma HLS PIPELINE II=1
    T data = in.read();
    for (int j = 0; j < N; j++) {
      #pragma HLS UNROLL
      out[j].write(data);
    }
  }
}

template<typename T, size_t N, size_t M>
void inf_split_tee(hls::stream<hls::vector<T, M>> (&out)[N], hls::stream<T> &in, const int vCount){
  
  hls::vector<T, M> tmp;
  for (int i = 0; i < vCount; i++) {
    #pragma HLS LOOP_TRIPCOUNT max= (MODEL_HIDDEN_DIM / MAX_QUANT_ELEM)
    #pragma HLS PIPELINE II=1
    // T data = in.read();

    for (int i = 0 ; i < M; i++) {
      tmp[i] = in.read();
    }
    for (int j = 0; j < N; j++) {
      #pragma HLS UNROLL
      out[j].write(tmp);
    }
  }
}

template<typename T, int N>
void inf_round_robin(hls::stream<T> (&out)[N], hls::stream<T> &in, const int vElem, const int vCount){
  
  const int vSize = vCount / N;
  
  inf_rr_loop:
  for (int i = 0; i < vSize; i++) {
    #pragma HLS LOOP_TRIPCOUNT min=(MODEL_ELEMENTS/N) max=(MODEL_TOKENS / N)
    
    elem_dist_loop:
    for (int j = 0; j < N; j++) {
      #pragma HLS LOOP_TRIPCOUNT max=N
      elem_per_stream_loop:
      for (int k = 0; k < vElem; k++) {
      #pragma HLS PIPELINE II=1 //style=flp
        #pragma HLS LOOP_TRIPCOUNT min=(MODEL_ELEMENTS / (MODEL_SCALING_FACTOR * SM_FL_ELEM)) max=(MODEL_TOKENS/MAX_QUANT_ELEM)
        T data = in.read();
        out[j].write(data);
      }
    }
  }
}

template<typename T, int N>
void rr_merge(hls::stream<T> &out, hls::stream<T> (&in)[N], const int M_DIM){
  
  tot_num_data:
  for (int i = 0; i < M_DIM / N; i++) {
    #pragma HLS LOOP_TRIPCOUNT max=MODEL_TOKENS / MAX_FL_ELEM
    #pragma HLS PIPELINE
      
      for (int j = 0; j < N; j++) {
        #pragma HLS UNROLL
        out.write(in[j].read());
      }
  }
}

template<typename T, size_t N, size_t M>
void vec_down_converter(hls::stream<hls::vector<T, N>> &out, hls::stream<hls::vector<T, M>> &in, const int N_cnt){
  
  static_assert(M % N == 0, "M must be divisible by N");
  static_assert(N > 0,      "N must be positive");
  // const int num_heads = vSize / MODEL_HEAD_SIZE;
  const size_t ratio = M/N;
  
  typedef hls::vector<T, M> M_t;
  typedef hls::vector<T, N> N_t;

  M_t shift_reg;
  
  mha_WAR_store: 
  for (int i = 0; i < N_cnt; i++) {
    #pragma HLS PIPELINE II=1
    
    if (i % ratio == 0) { shift_reg = in.read(); }
    
    N_t tmp;
    for (int k = 0; k < N; k++) {
        #pragma HLS UNROLL
        tmp[k] = shift_reg[k]; 
    }
    out.write(tmp);
    
    for (int s = 0; s < M - N; s++) {
        #pragma HLS UNROLL
        shift_reg[s] = shift_reg[s + N];
    }
  }
}

template<typename T, size_t N, size_t M>
void mm2s_vec_up(hls::stream<hls::vector<T, N>> &in, hls::vector<T, M> *out, const int OFFSET, const int N_COUNT){
	
  static_assert(N % M == 0, "M must be divisible by N");
  static_assert(M > 0,      "M must be positive");
	constexpr size_t ratio = N/M;
	
  typedef hls::vector<T, M> M_t;
  typedef hls::vector<T, N> N_t;

	N_t shift_reg;
	
	for (int i = 0; i < (ratio * N_COUNT); i++) {
		#pragma HLS PIPELINE II=1
		
		M_t tmp = out[i + OFFSET];
		for (int j = 0; j < (N - M); j++) {
			#pragma HLS UNROLL
			shift_reg[j] = shift_reg[j + M];
		}

		for (int j = 0; j < M; j++) {
			#pragma HLS UNROLL
			shift_reg[j + N - M] = tmp[j];
		}
		if (i % ratio == (ratio - 1)) {
			in.write(shift_reg);
		}
	}
}
template<typename T, size_t N, size_t M>
void mm2s_vec_up(hls::stream<hls::vector<T, N>> &in, hls::vector<T, M> *out, const int OFFSET_1, const int OFFSET_2, const int N_COUNT){
	
mm2s_vec_up(in, out, OFFSET_1, N_COUNT/2);
mm2s_vec_up(in, out, OFFSET_2, N_COUNT/2);
}

template<typename T>//
void mm2s_input_data(hls::stream<T> &out, T *in, const int COUNT, const int CURR_LAYER, const int offset){
  
  const int tot_off = CURR_LAYER * COUNT + offset; //line 286
	#pragma HLS BIND_OP variable=tot_off op=mul impl=dsp latency=2 // WTF
	T* base_ptr = in + tot_off;
  AXI4_to_STREAM:
  for (int i = 0; i < COUNT; i++) {
    #pragma HLS PIPELINE II=1
    #pragma HLS LOOP_TRIPCOUNT min=(MODEL_ELEMENTS * MODEL_ELEMENTS / (2 * SM_FL_ELEM * MODEL_SCALING_FACTOR)) max=(MODEL_TOKENS * MODEL_ELEMENTS / (2 * MAX_QUANT_ELEM))
    out.write(base_ptr[i]);
  }
}

template<typename T, size_t N>
void mm2mm_store(hls::vector<T, N> *mm_out, hls::vector<T,N> *mm_in, const int count){
  
  const int vCount = count/ N;
  
  mm2mm_writer:
  for (int i = 0; i < vCount; i++) {
    #pragma HLS PIPELINE II=1
    hls::vector<T, N> tmp = mm_in[i];
    mm_out[i] = tmp; //mm_in[i];
  }
}

template<typename T, size_t N>//
void mm2mm_store(hls::vector<T, N> *mm_out, hls::vector<T,N> *mm_in, const int count, const int ts, const int cur_spl, const int OUT_OFF, const int IN_OFF){
  
  const int vCount = count/ (N * ts);
  const int wos = OUT_OFF * cur_spl / (N * ts);
  const int ros = IN_OFF + cur_spl * vCount;
  
  mm2mm_writer:
  for (int i = 0; i < vCount; i++) {
    #pragma HLS PIPELINE II=1
    mm_out[i + wos] = mm_in[i + ros];
  }
}

template<typename T, size_t N>
void mha_WAR_store_load(hls::vector<T, N> *cache, hls::stream<hls::vector<T, N>> &output, hls::stream<hls::vector<T, N>> &input, const int CURR_LAYER, const int POS){
  const int vec_per_head = MODEL_HEAD_SIZE / N;

  const int layer_offset = CURR_LAYER * MODEL_NUM_HEADS * MODEL_SEQUENCE_LEN * vec_per_head;
  const int head_offset = MODEL_SEQUENCE_LEN * vec_per_head;
  const int pos_offset = POS * vec_per_head;
	typedef hls::vector<T, N> N_t;
  
  const int vec_to_read = vec_per_head * (POS); // remove the + 1 from here.
  for (int idx = 0; idx < MODEL_NUM_HEADS; idx++) {
		int baseaddr = layer_offset + (idx * head_offset);
		int baseaddrw = layer_offset + (idx * head_offset) + pos_offset;
	  fw_mha_pos:
    for (int j = 0; j < vec_to_read; j++) {
      #pragma HLS PIPELINE II=1
      #pragma HLS LOOP_TRIPCOUNT max=MODEL_HEAD_SIZE * (MODEL_SEQUENCE_LEN + 1) / MAX_FL_ELEM
      int addr = baseaddr + j;
      N_t tmp = cache[addr];
      output.write(tmp);
    } // second for loop that will read 4 elements from array
    fw_mha_new:
    for (int j = 0; j < vec_per_head; j++) {
      #pragma HLS PIPELINE II=1
			int addr =  baseaddrw + j;
			N_t tmpa = input.read();
      output.write(tmpa);
			// cache_array[j] = tmpa;
			cache[addr] = tmpa;
    }
	}
}

/* *************************** GeMV FUNCTION *************************************/


void s_GeMV_kernel(hls::stream<my_float_t> &out, s_fdata_v_t &tok_sf, s_idata_v_t &tok_q, s_mfdata_v_t &s_wsf, s_idata_v_t &s_w, const int N_DIM, const int M_DIM);

constexpr size_t TOK_QUANT_MAX =  (MODEL_HIDDEN_DIM / MAX_QUANT_ELEM);
constexpr size_t TOK_SF_MAX = (MODEL_HIDDEN_DIM / MODEL_SCALING_FACTOR);



/* *************************** QUANTIZER FUNCTION *************************************/

void quantizer_kernel(hls::stream<my_float_t>  &tok_sf_out, s_idata_v_t &tok_out, s_fdata_v_t &tokens, const int N_DIM);
void quantizer_kernel(hls::stream<my_float_t>  &tok_sf_out, s_idata_v_t &tok_out, s_fdata_v_t &tokens, const int N_DIM, fdata_v_t *data_out, const int SAVE_ADDR);

/* *************************** RoPE FUNCTION *************************************/

template<int HEAD>
void init_freq_arr(float arr[HEAD]){
  for (int i = 0; i < HEAD; i++) {
  arr[i] = 1.0f / hls::powf(10000.0f, ((i) / (float) MODEL_HEAD_SIZE));
  }
}

template<typename T, size_t N, int N_DIM = MODEL_ELEMENTS>
void rope_kernel (hls::stream<hls::vector<T, N>> &o, hls::stream<hls::vector<T, N>> &in, const int POS){
  float arr[MODEL_HEAD_SIZE];
  init_freq_arr<MODEL_HEAD_SIZE>(arr);
  ROPE_MAIN:
  for (int i = 0; i < (N_DIM / N); i++) {
    #pragma HLS loop_flatten 
 // increment by number of element in fdata_v_t
  
  int k = i * N;
    hls::vector<T, N> tmp = in.read();
    hls::vector<T, N> tmp_o;
    head_dim_unroll_loop:
    for (int j = 0 ; j < (N / 2); j++) {
      #pragma HLS PIPELINE
      #pragma HLS UNROLL factor = 2
      int head_dim = (k + j * 2) % MODEL_HEAD_SIZE;
      float freq =  arr[head_dim]; /*1.0f / hls::powf(10000.0f, (float)head_dim/HEAD_SIZE);*/ 
      float val = POS * freq;
      float fcr;
      float fci;
      hls::sincosf(val, &fci, &fcr);
      float v0 = tmp[j * 2 + 0];
      float v1 = tmp[j * 2 + 1];
      tmp_o[j * 2 + 0] = v0 * fcr - v1 * fci;
      tmp_o[j * 2 + 1] = v0 * fci + v1 * fcr;
    }
    o.write(tmp_o);
  }
}

/* *************************** MULTIHEAD ATTENTION FUNCTION *************************************/
void mha_kernel(s_fdata_v_t &output, fdata_v_t *tokens, adata_v_t *key_cache,  adata_v_t *value_cache,  const int POS, const int CURR_LAYER);


/* *************************** SWIGLU FUNCTION *************************************/

template<size_t M = 4, typename T, size_t N>//
void swiglu(hls::stream<hls::vector<T, N>> &hb_out, hls::stream<hls::vector<T, N>> &hb_in, hls::stream<hls::vector<T, N>> &hb2_in){
  // int elem = sizeof(T)/ sizeof(my_float_t);
  typedef hls::vector<T, N> tmp_t;
  const int HD_N_RATIO = MODEL_HIDDEN_DIM / N;
  for (int i = 0 ; i < HD_N_RATIO; i++) {
  #pragma HLS pipeline II=1
    tmp_t val = hb_in.read();
    tmp_t tmp_hb2 = hb2_in.read();
    tmp_t eval;
    for (int j = 0; j < N; j++) {
      #pragma HLS UNROLL factor = M
      // #pragma HLS PIPELINE II=2
      float_t tmpa = 1.0f + hls::expf(- val[j]);
      float_t tmpb = hls::recipf(tmpa);
      float_t tmpc = tmp_hb2[j] * val[j];
      // eval[j] = val[j] / ( 1.0f + hls::expf(-1 * val[j])) * tmp_hb2[j];
      eval[j] = tmpc * tmpb;
    }
    hb_out.write(eval);
  }
}

void swiglu_kernel(s_fdata_v_t &output, fdata_v_t *w1w3);

/* *************************** TOP FUNCTION *************************************/
/* *************************** TRANSFORMER KERNEL *************************************/

void transformer_cu(  
    fdata_v_t *tokens, 
    mfdata_v_t *w_sf_0, idata_v_t *w_0, mfdata_v_t *w_sf_1, idata_v_t *w_1, 
    fdata_v_t *weights, mfdata_v_t *key_cache, mfdata_v_t *value_cache, 
    const int POS, int QKV_W, const int QKV_sf_W,
    const int Out_W, const int Out_sf_W, const int FF_w1w3_W, const int FF_w1w3_sf_W,
    const int FF_w2_W, const int FF_w2_sf_W, const int Embed_W, const int Embed_sf_W, 
    const int rms_att_W, const int rms_ffn_W, const int rms_final_W, int *curr_token,
  #ifdef __DEBUG__
      const int faker, const int CURR_LAYER, const int NEXT_STATE, fdata_v_t *data_out,
  #endif
  #ifdef __ULTRADEBUG__
    fdata_v_t *GeMV_data_out,
  #endif
    const float temperature, const float coin, const bool init_rms_flag, const bool prefill_flag );
        
#endif

