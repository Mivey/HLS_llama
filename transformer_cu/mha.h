
#ifndef MARK_MHA_H__
#define MARK_MHA_H__

#include "mha_forward.h"
#include "hls_fence.h"
#include <cstddef>
#include <hls_vector.h>

// void mha_kernel(mfdata_v_t *tokens, mfdata_v_t *key_cache, mfdata_v_t *value_cache, 
// 	mfdata_v_t *key_cache_in, mfdata_v_t *value_cache_in, const int POS, const int CURR_LAYER);


// void mha_kernel(s_fdata_v_t &output,
// 								adata_v_t *tokens, //6 mha_kernel
//                 adata_v_t *key_cache, 
//                 adata_v_t *value_cache, 
//                 const int POS, const int CURR_LAYER);

void mha_kernel(s_fdata_v_t &output,
								fdata_v_t *tokens, //6 mha_kernel
                mfdata_v_t *key_cache, 
                mfdata_v_t *value_cache, 
                const int POS, const int CURR_LAYER);


void mha_kernel(hls::stream<my_float_t> &sf,
								s_idata_v_t &w,
								fdata_v_t *tokens, //6 mha_kernel
                mfdata_v_t *key_cache, 
                mfdata_v_t *value_cache, 
                const int POS, const int CURR_LAYER);


template<typename T, size_t N>
void mha_writeback(hls::vector<T, N> *cache, hls::stream<hls::vector<T, N>> &input, const int CURR_LAYER, const int POS){

	const int vec_per_head = MODEL_HEAD_SIZE / N;
	const int layer_offset = CURR_LAYER * MODEL_NUM_HEADS * MODEL_SEQUENCE_LEN * vec_per_head;
	const int head_offset = MODEL_SEQUENCE_LEN * vec_per_head;
	const int pos_offset = POS * vec_per_head;
	store_to_m_axi: 
	for (int i = 0; i < MODEL_NUM_HEADS; i++) {
		int saddr = layer_offset + (i * head_offset) + pos_offset;
		for (int j = 0; j < vec_per_head; j++) {
			#pragma HLS PIPELINE II=1
			// int addr = layer_offset + (i * head_offset) + pos_offset + j;
			cache[saddr + j] = input.read(); // cache_array[j + vec_per_head * i]; // this happens AFTER we're done reading from RAM
		}
	}
}

template<typename T, size_t N, size_t M>
void mha_WAR_data_mover_uk(hls::vector<T, M> *cache, hls::stream<hls::vector<T, M>> &output, hls::stream<hls::vector<T, N>> &input, const int CURR_LAYER, const int POS){

	static_assert(M % N == 0, "M must be divisible by N");
	static_assert(N > 0,      "N must be positive");
	// const int num_heads = vSize / MODEL_HEAD_SIZE;
	const int vec_per_head = MODEL_HEAD_SIZE / M;
	const int cache_arr_size = vec_per_head * MODEL_NUM_HEADS;

	const int layer_offset = CURR_LAYER * MODEL_NUM_HEADS * MODEL_SEQUENCE_LEN * vec_per_head;
	const int head_offset = MODEL_SEQUENCE_LEN * vec_per_head;
	const int pos_offset = POS * vec_per_head;
	const size_t ratio = M/N;
	
	typedef hls::vector<T, M> M_t;
	typedef hls::vector<T, N> N_t;
	
	M_t cache_array[cache_arr_size];
	
	mha_WAR_store: // convert N_t to M_t (ie fdata_v_t to mfdata_v_t)
	for (int i = 0;  i < cache_arr_size; i++) {
		// #pragma hls PIPELINE II=1
		M_t mtmp;
		for (int j = 0; j < ratio; j++) {
			#pragma HLS PIPELINE II=1
			N_t tmp = input.read();
			for (int k = 0; k < N; k++) {
				mtmp[j * N + k] = tmp[k];
			}
		}
		cache_array[i] = mtmp;
	}

	
	const int vec_to_read = vec_per_head * (POS); // remove the + 1 from here.
		
	mha_num_head_loop:
	for (int i = 0; i < MODEL_NUM_HEADS; i++) {
			#pragma HLS LOOP_FLATTEN
		int addr = layer_offset + (i * head_offset);
		fw_mha_pos_loop:
		for (int j = 0; j < vec_to_read; j++) {
			#pragma HLS PIPELINE II=1
			#pragma HLS LOOP_TRIPCOUNT max=MODEL_HEAD_SIZE * (MODEL_SEQUENCE_LEN + 1) / MAX_FL_ELEM
			// int addr = layer_offset + (i * head_offset) + j;
			// mfdata_v_t tmp = cache[addr + j];
			output.write(cache[addr + j]);
		} // second for loop that will read 4 elements from array
		fw_mha_new:
		for (int j = 0; j < vec_per_head; j++) {
			#pragma HLS PIPELINE II=1
			// int t = j + i * vec_per_head;
			output.write(cache_array[j + i * vec_per_head]);
		}
	}
	
	hls::fence(output, input);
	
	// #pragma HLS STREAM variable=input depth=48
	// #pragma HLS STREAM variable=output depth=4
	store_to_m_axi_loop: 
	for (int i = 0; i < MODEL_NUM_HEADS; i++) {
		int saddr = layer_offset + (i * head_offset) + pos_offset;
		for (int j = 0; j < vec_per_head; j++) {
			#pragma HLS PIPELINE II=1
			// int addr = layer_offset + (i * head_offset) + pos_offset + j;
			cache[saddr + j] = cache_array[j + vec_per_head * i]; // this happens AFTER we're done reading from RAM
		}
	}
}


template<size_t N, size_t M>
void vec_up_converter(hls::stream<hls::vector<my_float_t, M>> &out, hls::stream<hls::vector<my_float_t, N>> &in, const int M_cnt){
	
	static_assert(M % N == 0, "M must be divisible by N");
	static_assert(N > 0,      "N must be positive");
	// const int num_heads = vSize / MODEL_HEAD_SIZE;
	const size_t ratio = M/N;
	
	typedef hls::vector<my_float_t, M> M_t;
	typedef hls::vector<my_float_t, N> N_t;
	
	// M_t cache_array[cache_arr_size];
	
	mha_WAR_store: // convert N_t to M_t (ie fdata_v_t to mfdata_v_t)
	for (int i = 0;  i < M_cnt; i++) {
		// #pragma hls PIPELINE II=1
		M_t mtmp;
		for (int j = 0; j < ratio; j++) {
			#pragma HLS PIPELINE II=1
			N_t tmp = in.read();
			for (int k = 0; k < N; k++) {
				mtmp[j * N + k] = tmp[k];
			}
		}
		out.write(mtmp);
	}
}

template<size_t N, size_t M>
void vec_down_converter(hls::stream<hls::vector<my_float_t, N>> &out, hls::stream<hls::vector<my_float_t, M>> &in, const int N_cnt){
	
	static_assert(M % N == 0, "M must be divisible by N");
	static_assert(N > 0,      "N must be positive");
	// const int num_heads = vSize / MODEL_HEAD_SIZE;
	const size_t ratio = M/N;
	
	typedef hls::vector<my_float_t, M> M_t;
	typedef hls::vector<my_float_t, N> N_t;
	
	mha_WAR_store: // convert N_t to M_t (ie fdata_v_t to mfdata_v_t)
	for (int i = 0;  i < N_cnt / ratio; i++) {
		// #pragma hls PIPELINE II=1
		M_t mtmp = in.read();
		for (int j = 0; j < ratio; j++) {
			#pragma HLS PIPELINE II=1
			N_t tmp;
			for (int k = 0; k < N; k++) {
				tmp[k] = mtmp[j * N + k];
			}
			out.write(tmp);
		}
	}

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
		// #pragma HLS PIPELINE II=1
		hls::vector<T, N> val = abs_tokens.read();
		tokens_out.write(tokens_in.read());
		for (int k = 0; k < N; k++) {
			#pragma HLS PIPELINE II=1
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
			quant_tmp[j * N + k] = (my_quant_data_t) hls::roundf(proc_tok[k] * scale);
		}
	}
	
	tok_out.write(quant_tmp);
	tok_sf_out.write(dscale);
}
#endif
