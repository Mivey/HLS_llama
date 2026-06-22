
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

constexpr size_t MAX_DW = 512;
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


template<typename T, size_t N, size_t M>
void mha_input_data(hls::stream<hls::vector<T, M>> &out, hls::vector<T, N> *in, const size_t CURR_LAYER, const int offset){
	
	const size_t COUNT = MODEL_ELEMENTS / N;
	const int tot_off = CURR_LAYER * COUNT + (offset / N);
	AXI4_to_STREAM:
	for (int i = 0; i < (MODEL_HEAD_SIZE / M); i++) {
		size_t idx = (M/N);
		hls::vector<T, M> temp_m;
		
		for (int j = 0; j < idx; j++) {
			#pragma HLS PIPELINE II=1
			
			size_t jdx = idx * i + j;
			hls::vector<T, N> temp_n = in[jdx + tot_off];
			
			for (int k = 0; k < N; k++) {
				size_t kdx = N * j + k;
				temp_m[kdx] = temp_n[k];
			}
		}
		out.write(temp_m);
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


/* =============================== NEW MHA WRITE AFTER READ ======================================= */
template<typename T, size_t N>
void mha_WAR_store_load(hls::vector<T, N> *cache, hls::stream<hls::vector<T, N>> &output, hls::stream<hls::vector<T, N>> &input, const int CURR_LAYER, const int POS){
	// const int num_heads = vSize / MODEL_HEAD_SIZE;
	const int vec_per_head = MODEL_HEAD_SIZE / N;
	const int cache_arr_size = vec_per_head * MODEL_NUM_HEADS;

	const int layer_offset = CURR_LAYER * MODEL_NUM_HEADS * MODEL_SEQUENCE_LEN * vec_per_head;
	const int head_offset = MODEL_SEQUENCE_LEN * vec_per_head;
	const int pos_offset = POS * vec_per_head;
	
	hls::vector<T, N> cache_array[cache_arr_size];
	mha_WAR_store_loop:
	for (int i = 0;  i < cache_arr_size; i++) {
		#pragma hls PIPELINE II=1
		cache_array[i] = input.read();
	}
	const int vec_to_read = vec_per_head * (POS); // remove the + 1 from here.
		
	mha_num_head_loop:
	for (int i = 0; i < MODEL_NUM_HEADS; i++) {
		#pragma HLS LOOP_FLATTEN
		fw_mha_pos_loop:
		for (int j = 0; j < vec_to_read; j++) {
			#pragma HLS PIPELINE II=1
			#pragma HLS LOOP_TRIPCOUNT max=MODEL_HEAD_SIZE * (MODEL_SEQUENCE_LEN + 1) / MAX_FL_ELEM
			int addr = layer_offset + (i * head_offset) + j;
			hls::vector<T, N> tmp = cache[addr];
			output.write(tmp);
		} // second for loop that will read 4 elements from array
		fw_mha_new_loop:
		for (int j = 0; j < vec_per_head; j++) {
			#pragma HLS PIPELINE II=1
			int t = j + i * vec_per_head;
			output.write(cache_array[t]);
		}
	}
	
	hls::fence(output, input);
	
	#pragma HLS STREAM variable=input depth=48
	#pragma HLS STREAM variable=output depth=4
	store_to_m_axi_loop: 
	for (int i = 0; i < MODEL_NUM_HEADS; i++) {
		for (int j = 0; j < vec_per_head; j++) {
			#pragma HLS PIPELINE II=1
			int addr = layer_offset + (i * head_offset) + pos_offset + j;
			cache[addr] = cache_array[j + vec_per_head * i]; // this happens AFTER we're done reading from RAM
		}
	}
}


template<typename T, size_t N>
void mha_WAR_store_load(hls::vector<T, N> *cache, hls::stream<hls::vector<T, N>> &output, hls::stream<hls::vector<T, N>> &input, const int CURR_LAYER, const int POS, const int idx){
	// const int num_heads = vSize / MODEL_HEAD_SIZE;
	const int vec_per_head = MODEL_HEAD_SIZE / N;
	// const int cache_arr_size = vec_per_head * MODEL_NUM_HEADS;

	const int layer_offset = CURR_LAYER * MODEL_NUM_HEADS * MODEL_SEQUENCE_LEN * vec_per_head;
	const int head_offset = MODEL_SEQUENCE_LEN * vec_per_head;
	const int pos_offset = POS * vec_per_head;
	
	hls::vector<T, N> cache_array[vec_per_head];
	mha_WAR_store_loop:
	for (int i = 0;  i < vec_per_head; i++) {
		#pragma hls PIPELINE II=1
		cache_array[i] = input.read();
	}
	const int vec_to_read = vec_per_head * (POS); // remove the + 1 from here.
		
	// mha_num_head_loop:
	// for (int i = 0; i < MODEL_NUM_HEADS; i++) {
		// #pragma HLS LOOP_FLATTEN
		fw_mha_pos_loop:
		for (int j = 0; j < vec_to_read; j++) {
			#pragma HLS PIPELINE II=1
			#pragma HLS LOOP_TRIPCOUNT max=MODEL_HEAD_SIZE * (MODEL_SEQUENCE_LEN + 1) / MAX_FL_ELEM
			int addr = layer_offset + (idx * head_offset) + j;
			hls::vector<T, N> tmp = cache[addr];
			output.write(tmp);
		} // second for loop that will read 4 elements from array
		fw_mha_new_loop:
		for (int j = 0; j < vec_per_head; j++) {
			#pragma HLS PIPELINE II=1
			int t = j + idx * vec_per_head;
			output.write(cache_array[t]);
		}
	// }
	
	hls::fence(output, input);
	
	#pragma HLS STREAM variable=input depth=48
	#pragma HLS STREAM variable=output depth=4
	store_to_m_axi_loop: 
	// for (int i = 0; i < MODEL_NUM_HEADS; i++) {
		for (int j = 0; j < vec_per_head; j++) {
			#pragma HLS PIPELINE II=1
			int addr = layer_offset + (idx * head_offset) + pos_offset + j;
			cache[addr] = cache_array[j + vec_per_head * idx]; // this happens AFTER we're done reading from RAM
		}
	// }
}


template<typename T, size_t N>
void abs_intake(hls::stream<hls::vector<T, N>> &tokens_out, hls::stream<hls::vector<T, N>> &abs_tokens, hls::stream<hls::vector<T, N>> &tokens_in){
	
	const size_t TOK_COUNT = MODEL_SCALING_FACTOR / N;
	T max_val = 0.0f;
	
	group_scaling:
	for (size_t j = 0; j < TOK_COUNT; j++) {
		#pragma HLS PIPELINE II=1
		
		hls::vector<T, N> val = tokens_in.read();
		tokens_out.write(val);
		
		hls::vector<T, N> c_val;
		for (int k = 0; k < N; k++) {
			c_val[k] = hls::absf(val[k]);
		}		
		abs_tokens.write(c_val);
	}
	// can probably get rid of abs_intake
}

template<typename T, size_t N>
void max_finder(hls::stream<T> &max_val, hls::stream<hls::vector<T, N>> &tokens_out, hls::stream<hls::vector<T, N>> &abs_tokens, hls::stream<hls::vector<T, N>> &tokens_in){
	
	const T Q_MAX = 1.0f / 127.0f;
	const int cnt = MODEL_SCALING_FACTOR / N;
	T c_val[MODEL_SCALING_FACTOR];
	#pragma HLS ARRAY_PARTITION variable=c_val dim=1 type=complete
	//here we store token_out and then assign token_out[i] to c_val[i * MAX_FL_ELEM + k] = hls::absf(token_out[i][k])
	
	
	mf_intake:
	for (int i = 0; i < cnt; i++) {
		#pragma HLS PIPELINE II=1
		hls::vector<T, N> val = abs_tokens.read();
		tokens_out.write(tokens_in.read());
		for (int k = 0; k < N; k++) {
			c_val[i * N + k] = val[k];
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

template<typename T, size_t N>
void quant_out( hls::stream<T> &tok_sf_out, s_idata_v_t &tok_out, hls::stream<hls::vector<T, N>> &tokens_in, hls::stream<my_float_t> &max_val){
	
	const size_t TOK_COUNT = MODEL_SCALING_FACTOR / N;
	T dscale = max_val.read(); 
	T scale = hls::recipf(dscale);//Q_MAX / max_val;
	idata_v_t quant_tmp; // not an array anymore
	// fdata_v_t tok_arr[TOK_COUNT];
	
	create_q_val:
	for (size_t j = 0; j < TOK_COUNT; j++) {
		// #pragma HLS PIPELINE
		hls::vector<T, N> proc_tok = tokens_in.read();
		
		create_q_val_ppl:	
		for (size_t k = 0; k < N; k++) {
			#pragma HLS PIPELINE
			// #pragma HLS UNROLL factor = 4
			quant_tmp[j * N + k] = (my_quant_data_t) hls::roundf(proc_tok[k] * scale);
		}
	}
	
	tok_out.write(quant_tmp);
	tok_sf_out.write(dscale);
}

void mha_WAR_store_load(mfdata_v_t *cache, s_mfdata_v_t &output, s_mfdata_v_t &input, const int CURR_LAYER, const int POS);
void mm_tok_load_input(s_idata_v_t &out, idata_v_t *in, const int vCount, const int CURR_LAYER);
void mm_load_input(s_fdata_v_t &out, fdata_v_t *in, const int vCount, const int CURR_LAYER);

#endif

