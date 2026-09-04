
#ifndef MARK_FORWARD
#define MARK_FORWARD

// #include "fast_common.h"
#include <cstddef>
#include <cstdint>
#include <hls_stream.h>
#include <hls_math.h>
#include <hls_vector.h>
#include "hls_fence.h"
// #include "mha_cu/mha.h"
// #include <ap_float.h>

#define DATAWIDTH 32
#define MODEL_ELEMENTS 768
#define MODEL_HIDDEN_DIM 2048
#define QUANT 8 // bits in the word... either 4 or 8
#define MODEL_NUM_HEADS 12
#define MODEL_NUM_LAYERS 12
#define MODEL_TOKENS 32000
#define MODEL_SEQUENCE_LEN 1024
#define MODEL_SCALING_FACTOR 64
#define bytes_in(n) sizeof(n)
#define runs(n) SCALING_FACTOR/sizeof(n)
constexpr float Q_FACTOR = ((QUANT%4)==0) ? \
                 static_cast<float>((1<<(QUANT - 1)) - 1) : 127;

/* ************************************* */
// typedef ap_float<32, 8> my_float_t;
typedef float my_float_t;
typedef int8_t my_quant_data_t;
/* ************************************* */

constexpr size_t MAX_DW = 256;
constexpr size_t QUANT_MODIFIER = 1;//(MAX_DW == 512) ? 2 : 1;
constexpr size_t SM_DW = 128;
constexpr size_t MAX_FL_ELEM = (MAX_DW / (sizeof(my_float_t) * 8));
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

const int SQUARE_TOK = MODEL_ELEMENTS * MODEL_ELEMENTS;
const int SQUARE_SF = SQUARE_TOK / MODEL_SCALING_FACTOR;
const int RECT_TOK = MODEL_ELEMENTS * MODEL_HIDDEN_DIM;
const int RECT_SF = RECT_TOK / MODEL_SCALING_FACTOR;

/* ==================================================================================== */

typedef hls::vector<my_quant_data_t, MAX_QUANT_ELEM> idata_v_t;
typedef hls::vector<my_float_t, SM_FL_ELEM>	fdata_v_t;
typedef hls::vector<my_float_t, MAX_FL_ELEM>	mfdata_v_t;

typedef hls::stream<idata_v_t> s_idata_v_t;
typedef hls::stream<fdata_v_t> s_fdata_v_t; 
typedef hls::stream<mfdata_v_t> s_mfdata_v_t;

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

template<typename T, int N>
void inf_round_robin(hls::stream<T> (&out)[N], hls::stream<T> &in, const int vElem, const int vCount){
	
	const int vSize = vCount / N;
	
  inf_rr_loop:
	for (int i = 0; i < vSize; i++) {
		
		elem_dist_loop:
		for (int j = 0; j < N; j++) {
			
			elem_per_stream_loop:
			for (int k = 0; k < vElem; k++) {
				#pragma HLS PIPELINE II=1
				T data = in.read();
				out[j].write(data);
			}
		}
	}
}


template<typename T, typename S, int N>
void rr_merge(hls::stream<S> &out, hls::stream<T> (&in)[N], const int vCount){
	S data;
	const int ratio = sizeof(S) / (sizeof(T) * N);
	tot_num_data_loop:
	for (int i = 0; i < vCount; i++) {
		#pragma HLS LOOP_TRIPCOUNT max=MODEL_TOKENS / MAX_FL_ELEM
		#pragma HLS PIPELINE
		ratio_loop:
		for (int j = 0; j < ratio; j++) {
			
			elem_merge_loop:
			for (int k = 0; k < N; k++) {
				int offset = j * N + k;
				data[offset] = in[k].read();
			}
		}
		out.write(data);
	}
}

template<typename T>
void mm2s_input_data(hls::stream<T> &out, T *in, const size_t COUNT){
	
	AXI4_to_STREAM:
	for (int i = 0; i < COUNT; i++) {
		#pragma HLS PIPELINE II=1
		out.write(in[i]);
	}
}

template<typename T>
void mm2s_input_data(hls::stream<T> &out, T *in, const size_t COUNT, const size_t CURR_LAYER){
	
	const int offset = CURR_LAYER * COUNT;
	AXI4_to_STREAM:
	for (int i = 0; i < COUNT; i++) {
		#pragma HLS PIPELINE II=1
		out.write(in[i + offset]);
	}
}

template<typename T>
void s2mm_output_data(T *out, hls::stream<T> &in,const size_t COUNT, const size_t W_Off){
	//remember to calculate W_Off before passing it here. T could be any size, lterally. 
	S2MM_output:
	for (int i = 0; i < COUNT; i++) {
		#pragma HLS LOOP_TRIPCOUNT max=MODEL_TOKENS min=MODEL_ELEMENTS
		#pragma HLS PIPELINE II=1
		out[i + W_Off] = in.read();
	}
}


template<typename T, int N>
void s2mm_output_data(hls::vector<T, N> *out, hls::stream<T> &in ,const size_t COUNT, const size_t W_Off){
	//remember to calculate W_Off before passing it here. T could be any size, lterally. 

	S2MM_output:
	for (int i = 0; i < COUNT / N; i++) {
		#pragma HLS LOOP_TRIPCOUNT max=MODEL_TOKENS / N min=MODEL_ELEMENTS / N
		
		hls::vector<T, N> tmp;
		for (int j = 0 ; j < N; j++) {
			#pragma HLS PIPELINE II=1
			tmp[j] = in.read();
		}
		
		out[i + W_Off] = tmp;
	}
}

template<typename T, int M>
void s2mm_output_data(T *out, hls::stream<T> (&in)[M] ,const size_t COUNT, const size_t W_Off){
	//remember to calculate W_Off before passing it here. T could be any size, lterally. 

	// T arr[M-1][COUNT];
	for (int j = 0; j < M; j++) {
	
		S2MM_output:
		for (int i = 0; i < COUNT; i++) {
			#pragma HLS LOOP_TRIPCOUNT max=MODEL_TOKENS min=MODEL_ELEMENTS
			#pragma HLS PIPELINE II=1
			out[COUNT * j + i + W_Off] = in[j].read();
		}
	}
}

template<typename T>
void store_output(T *out, hls::stream<T> &in , const int vSize){

int elem = sizeof(T) / sizeof(float);
const int NUM = vSize / elem;
// int offset = CURR_LAYER * 
	store_to_m_axi_loop: 
	for (int i = 0; i < NUM; i++) {
		#pragma HLS PIPELINE II=1
		out[i] = in.read();
	}
}

template<typename T>
void store_bytes_output(T *out, hls::stream<T> &in , const int vCount){


// int offset = CURR_LAYER * 
	store_to_m_axi_loop: 
	for (int i = 0; i < vCount; i++) {
		#pragma HLS PIPELINE II=1
		out[i] = in.read();
	}
}

template<typename T>
void store_output(T *out, hls::stream<my_float_t> &in , const int vSize){

int elem = sizeof(T) / sizeof(float);
const int NUM = vSize / elem;
// int offset = CURR_LAYER * 
	store_to_m_axi_loop: 
	for (int i = 0; i < NUM; i++) {
		T tmp;
		for (int j = 0; j < elem; j++) {
		#pragma HLS PIPELINE II=1
			tmp[j] = in.read();
		}
		out[i] = tmp;
	}
}

template<typename T>
void rms_load_input(hls::stream<T> &out, T *in, const int CURR_LAYER){
  // load_input<(MODEL_ELEMENTS / MAX_W_F)>(out, in);
	int elem = sizeof(T) / sizeof(float);
	const int offset = CURR_LAYER * MODEL_ELEMENTS / elem;
	fw_rms_load_loop:
	for (int i = 0; i < (MODEL_ELEMENTS/elem); i++) {
		#pragma HLS PIPELINE II=1
		T data = in[i + offset];
		out.write(data);
	}
}

template<typename T>
void tok_load_input(hls::stream<T> &out, T *in){
	int elem = sizeof(T) / sizeof(float);
	fw_token_load_loop:
	for (int i = 0; i < (MODEL_ELEMENTS/elem); i++) {
		#pragma HLS PIPELINE II=1
		out.write(in[i]);
	}
}

template<typename T>
void tok_load_input(hls::stream<T> &out, T *in, const int N_DIM){
	int elem = sizeof(T) / sizeof(float);
	fw_token_load_loop:
	for (int i = 0; i < (N_DIM/elem); i++) {
		#pragma HLS PIPELINE II=1
		out.write(in[i]);
	}
}

/* *************************** SWIGLU FUNCTION *************************************/
template<typename T>
void swiglu(hls::stream<T> &hb_out, hls::stream<T> &hb_in, hls::stream<T> &hb2_in){
	int elem = sizeof(T)/ sizeof(my_float_t);
	for (int i = 0 ; i < MODEL_HIDDEN_DIM / elem; i++) {
	#pragma HLS pipeline II=4
		T val =hb_in.read();
		T eval;
		for (int j = 0; j < elem; j++) {
			#pragma HLS UNROLL
			eval[j] = val[j] / ( 1.0f + hls::expf(-1 * (float)val[j]));
		}
		hb_out.write(eval * hb2_in.read());
	}
}

/* =================================== RESIDUAL CONNECTION ===================================== */

template<typename T>
void resid_conn(hls::stream<T> &tokens_out, hls::stream<T> &tokens_in, hls::stream<T> &xb){
	int elem = sizeof(T)/ sizeof(my_float_t);
	for (int i = 0; i < MODEL_ELEMENTS / elem; i++) {
		#pragma HLS PIPELINE II=1
		T tmp, tmpa, tmpb;
		tmpa =tokens_in.read();
		tmpb = xb.read();
		tmp = tmpa + tmpb;// tokens_in.read() + xb.read();
		tokens_out.write(tmp);

	}
}
void mha_WAR_store_load(mfdata_v_t *cache, s_mfdata_v_t &output, s_mfdata_v_t &input, const int CURR_LAYER, const int POS);
void mm_tok_load_input(s_idata_v_t &out, idata_v_t *in, const int vCount, const int CURR_LAYER);
void mm_load_input(s_fdata_v_t &out, fdata_v_t *in, const int vCount, const int CURR_LAYER);

#endif

