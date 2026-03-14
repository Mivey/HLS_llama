#include "forward.h"
#include "hls_fence.h"
#include <hls_vector.h>

void mm_tok_load_input(s_idata_v_t &out, idata_v_t *in, const int vCount, const int CURR_LAYER){
  // const int offset = CURR_LAYER * vCount;

  // const int loop = vCount / MAX_QUANT_ELEM;
	const int tot_size = vCount / MAX_QUANT_ELEM;
	const int offset = CURR_LAYER * tot_size;
  fw_load_mm_quant_loop:
  for (int i = 0; i < tot_size; i++) {
		#pragma HLS LOOP_TRIPCOUNT max=MODEL_TOKENS * MODEL_ELEMENTS / MAX_QUANT_ELEM min= MODEL_ELEMENTS * MODEL_ELEMENTS / MAX_QUANT_ELEM
    #pragma HLS PIPELINE II=1
		out.write(in[i + offset]);
  }
}

void mm_load_input(s_fdata_v_t &out, fdata_v_t *in, const int vCount, const int CURR_LAYER){
  // const int offset = CURR_LAYER * vCount;
  // const int loop = vCount / SM_FL_ELEM;
	const int tot_size = vCount / SM_FL_ELEM;
	const int offset = CURR_LAYER * tot_size;
  fw_load_mm_sf_loop:
  for (int i = 0; i < tot_size; i++) {
		#pragma HLS LOOP_TRIPCOUNT max=MODEL_TOKENS * MODEL_ELEMENTS / (SM_FL_ELEM * MODEL_SCALING_FACTOR) min= MODEL_ELEMENTS * MODEL_ELEMENTS / (SM_FL_ELEM * MODEL_SCALING_FACTOR) 
		#pragma HLS PIPELINE II=1
		out.write(in[i + offset]);
  }
}

void tok_load_input(s_mfdata_v_t &out, mfdata_v_t *in){
	fw_token_load_loop:
	for (int i = 0; i < (MODEL_ELEMENTS/MAX_FL_ELEM); i++) {
		#pragma HLS PIPELINE II=1
		out.write(in[i]);
	}
}

void mha_WAR_store_load(mfdata_v_t *cache, s_mfdata_v_t &output, s_mfdata_v_t &input, const int CURR_LAYER, const int POS){
	// const int num_heads = vSize / MODEL_HEAD_SIZE;
	const int vec_per_head = MODEL_HEAD_SIZE / MAX_FL_ELEM;
	const int cache_arr_size = vec_per_head * MODEL_NUM_HEADS;

	const int layer_offset = CURR_LAYER * MODEL_NUM_HEADS * MODEL_SEQUENCE_LEN * vec_per_head;
	const int head_offset = MODEL_SEQUENCE_LEN * vec_per_head;
	const int pos_offset = POS * vec_per_head;
	
	mfdata_v_t cache_array[cache_arr_size];
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
			mfdata_v_t tmp = cache[addr];
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
void mha_writeback(hls::vector<T, N> *output, hls::stream<hls::vector<T, N>> &input, const int CURR_LAYER, const int POS){

	const int vec_per_head = MODEL_HEAD_SIZE / N;
	const int layer_offset = CURR_LAYER * MODEL_NUM_HEADS * MODEL_SEQUENCE_LEN * vec_per_head;
	const int head_offset = MODEL_SEQUENCE_LEN * vec_per_head;
	const int pos_offset = POS * vec_per_head;
	store_to_m_axi: 
	for (int i = 0; i < MODEL_NUM_HEADS; i++) {
		for (int j = 0; j < vec_per_head; j++) {
			#pragma HLS PIPELINE II=1
			int addr = layer_offset + (i * head_offset) + pos_offset + j;
			cache[addr] = cache_array[j + vec_per_head * i]; // this happens AFTER we're done reading from RAM
		}
	}
}

void store_output(mfdata_v_t *out, s_mfdata_v_t &in , const int vSize, const int CURR_LAYER){
const int NUM = vSize / MAX_FL_ELEM;
// int offset = CURR_LAYER * 
	store_to_m_axi_loop: 
	for (int i = 0; i < NUM; i++) {
		#pragma HLS PIPELINE II=1
		out[i] = in.read();
	}
}

