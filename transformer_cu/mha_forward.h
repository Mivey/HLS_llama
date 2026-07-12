
#ifndef MARK_FORWARD
#define MARK_FORWARD

// #define __DEBUG__

#include <cstddef>
#include <cstdint>
#include <hls_stream.h>
#include <hls_math.h>
#include <hls_vector.h>
#include "hls_fence.h"

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

const int SQUARE_TOK = MODEL_ELEMENTS * MODEL_ELEMENTS;
const int SQUARE_SF = SQUARE_TOK / MODEL_SCALING_FACTOR;
const int RECT_TOK = MODEL_ELEMENTS * MODEL_HIDDEN_DIM;
const int RECT_SF = RECT_TOK / MODEL_SCALING_FACTOR;

/* ==================================================================================== */

typedef hls::vector<my_quant_data_t, MAX_QUANT_ELEM> idata_v_t;
typedef hls::vector<my_float_t, SM_FL_ELEM>	fdata_v_t;
typedef hls::vector<my_float_t, MAX_FL_ELEM>	mfdata_v_t;
typedef hls::vector<my_float_t, MID_FL_ELEM>	adata_v_t;

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
void mm2s_input_data(hls::stream<T> &out, T *in, const size_t COUNT, const size_t CURR_LAYER, const int offset){
	
	const int tot_off = CURR_LAYER * COUNT + offset;
	AXI4_to_STREAM:
	for (int i = 0; i < COUNT; i++) {
		#pragma HLS PIPELINE II=1
		out.write(in[i + tot_off]);
	}
}

template<typename T, size_t N, size_t M>
void mha_input_data(hls::stream<hls::vector<T, M>> &out, hls::vector<T, N> *in, const int offset, const bool BOUNDRY){
	
	const size_t COUNT = MODEL_ELEMENTS / N;
	const size_t factor = offset / MODEL_HEAD_SIZE - MODEL_NUM_HEADS;
	const size_t m_offset = (size_t) ((INTERNAL_DATA_SIZE / (2 * MODEL_HEAD_SIZE)) + factor - (MODEL_NUM_HEADS * 1 / 2)) * MODEL_HEAD_SIZE;
	int tot_off;
	if ((BOUNDRY == 1) && (factor >= (MODEL_NUM_HEADS / 2))) {
		tot_off =(m_offset / N);
	} else {
		tot_off = (offset / N);
	}
	 
	AXI4_to_STREAM:
	for (size_t i = 0; i < (MODEL_HEAD_SIZE / M); i++) {
		size_t idx = (M/N);
		hls::vector<T, M> temp_m;
		
		for (size_t j = 0; j < idx; j++) {
			#pragma HLS PIPELINE II=1
			
			size_t jdx = idx * i + j;
			hls::vector<T, N> temp_n = in[jdx + tot_off];
			
			for (size_t k = 0; k < N; k++) {
				// #pragma HLS PIPELINE II=1
				#pragma HLS UNROLL
				size_t kdx = N * j + k;
				temp_m[kdx] = temp_n[k];
			}
		}
		out.write(temp_m);
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

template<typename T, int M, size_t N>
void s2arr_output_data(hls::vector<T, N> *out, hls::stream<hls::vector<T, N> > (&in)[M] ,const size_t COUNT, const size_t W_Off, const int AXI_SEL){
	//remember to calculate W_Off before passing it here. T could be any size, lterally. 
	const int tot_cnt = COUNT / (N * M);
	if (AXI_SEL == 0) {
	
		S2MM_output:
		for (int i = 0; i < tot_cnt; i++) {
			#pragma HLS LOOP_TRIPCOUNT max=MODEL_TOKENS min=MODEL_ELEMENTS
			#pragma HLS PIPELINE II=1
			for (int j = 0; j < M; j++) {
				out[tot_cnt * j + i + W_Off] = in[j].read();
			}
		}
	}
}

template<typename T, size_t N>
void s_mm_output_sel(hls::vector<T, N> *mm_out, hls::stream<hls::vector<T, N>> &s_out, hls::stream<hls::vector<T, N>> &s_in, const int COUNT, const size_t W_Off, const int AXI_SEL){
	
	if (AXI_SEL == 1) {
		for (int i = 0; i < COUNT; i++) {
			#pragma HLS LOOP_TRIPCOUNT max=MODEL_TOKENS min=MODEL_ELEMENTS
			#pragma HLS PIPELINE II=1
			mm_out[i + W_Off] = s_in.read();
		}
	}else {
		for (int i = 0; i < COUNT; i++) {
			#pragma HLS LOOP_TRIPCOUNT max=MODEL_TOKENS min=MODEL_ELEMENTS
			#pragma HLS PIPELINE II=1
			s_out.write(s_in.read());
		}
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

template<typename T, size_t N>
void mm2mm_store(hls::vector<T, N> *mm_out, hls::vector<T,N> *mm_in, const int count, const int ts, const int cur_spl, const int offset){
	
	const int vCount = count/ (N * ts);
	const int wos = offset * cur_spl / (N * ts);
	
	mm2mm_writer:
	for (int i = 0; i < vCount; i++) {
		#pragma HLS PIPELINE II=1
		mm_out[i + wos] = mm_in[i + cur_spl * vCount];
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


/* *************************** SWIGLU FUNCTION *************************************/
template<typename T>
void swiglu(hls::stream<T> &hb_out, hls::stream<T> &hb_in, hls::stream<T> &hb2_in){
	int elem = sizeof(T)/ sizeof(my_float_t);
	for (int i = 0 ; i < MODEL_HIDDEN_DIM / elem; i++) {
	#pragma HLS pipeline II=4
		T val =hb_in.read();
		T tmp_hb2 = hb2_in.read();
		T eval;
		for (int j = 0; j < elem; j++) {
			#pragma HLS UNROLL
			eval[j] = val[j] / ( 1.0f + hls::expf(-1 * val[j])) * tmp_hb2[j];
		}
		hb_out.write(eval);
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
// void mha_WAR_store_load(adata_v_t *cache, s_adata_v_t &output, s_adata_v_t &input, const int CURR_LAYER, const int POS);
// void transformer_cu(	//s_fdata_v_t (&tok_sf)[mm_thr] , s_idata_v_t (&tok_q)[mm_thr],
// 								fdata_v_t *out_0, fdata_v_t *w_sf_0, idata_v_t *w_0, 
// 								fdata_v_t *out_1, fdata_v_t *w_sf_1, idata_v_t *w_1, 
// 								fdata_v_t *tokens, fdata_v_t *weights, fdata_v_t *w1w3, 
// 								adata_v_t *mha_tokens, adata_v_t *key_cache, adata_v_t *value_cache, 
// 								const int POS, const int N_DIM, const int M_DIM, 
// 								const int QKV_W, const int QKV_sf_W,
// 								const int Out_W, const int Out_sf_W,
// 								const int FF_w1w3_W, const int FF_w1w3_sf_W,
// 								const int FF_w2_W, const int FF_w2_sf_W, 
// 								const int Embed_W, const int Embed_sf_W, 
// 								const int rms_att_W, const int rms_ffn_W, const int rms_final_W,
// 								const int faker);

template<typename T, size_t N>
void mha_WAR_store_load(hls::vector<T, N> *cache, hls::stream<hls::vector<T, N>> &output, hls::stream<hls::vector<T, N>> &input, const int CURR_LAYER, const int POS, const int idx){
	const int vec_per_head = MODEL_HEAD_SIZE / N;

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
		
		fw_mha_pos:
		for (int j = 0; j < vec_to_read; j++) {
			#pragma HLS PIPELINE II=1
			#pragma HLS LOOP_TRIPCOUNT max=MODEL_HEAD_SIZE * (MODEL_SEQUENCE_LEN + 1) / MAX_FL_ELEM
			int addr = layer_offset + (idx * head_offset) + j;
			hls::vector<T, N> tmp = cache[addr];
			output.write(tmp);
		} // second for loop that will read 4 elements from array
		fw_mha_new:
		for (int j = 0; j < vec_per_head; j++) {
			#pragma HLS PIPELINE II=1
			// int t = j + idx * vec_per_head;
			// output.write(cache_array[t]);
			output.write(cache_array[j]);
		}
	
	hls::fence(output, input);
	
	#pragma HLS STREAM variable=input depth=48
	#pragma HLS STREAM variable=output depth=4
	store_to_m_axi: 
		for (int j = 0; j < vec_per_head; j++) {
			#pragma HLS PIPELINE II=1
			int addr = layer_offset + (idx * head_offset) + pos_offset + j;
			// cache[addr] = cache_array[j + vec_per_head * idx]; // this happens AFTER we're done reading from RAM
			cache[addr] = cache_array[j]; // this happens AFTER we're done reading from RAM
		}
}


void transformer_cu(	//s_fdata_v_t (&tok_sf)[mm_thr] , s_idata_v_t (&tok_q)[mm_thr],
								fdata_v_t *tokens, //fdata_v_t *bokens, 
								mfdata_v_t *w_sf_0, idata_v_t *w_0, 
								mfdata_v_t *w_sf_1, idata_v_t *w_1, 
								fdata_v_t *weights, mfdata_v_t *key_cache, mfdata_v_t *value_cache, 
								const int POS, //const int N_DIM, const int M_DIM, 
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
				);
	void systolic_sort(fdata_v_t *logit, int* pick, const float temperature, const float coin);
#endif

