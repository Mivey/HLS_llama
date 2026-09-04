#include "mha_forward.h"
#include "hls_fence.h"


void mm_tok_load_input(s_idata_v_t &out, idata_v_t *in, const int vCount, const int CURR_LAYER){
	
	const int tot_size = vCount / MAX_QUANT_ELEM;
	const int offset = CURR_LAYER * tot_size;
  fw_load_mm_quant_loop:
  for (int i = 0; i < tot_size; i++) {
    #pragma HLS PIPELINE
		out.write(in[i + offset]);
  }
}

void mm_load_input(s_fdata_v_t &out, fdata_v_t *in, const int vCount, const int CURR_LAYER){
	
	const int tot_size = vCount / SM_FL_ELEM;
	const int offset = CURR_LAYER * tot_size;
  fw_load_mm_sf_loop:
  for (int i = 0; i < tot_size; i++) {
		#pragma HLS PIPELINE II=1
		out.write(in[i + offset]);
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
		// mfdata_v_t tmp;
		// for (int j = 0; j < (MAX_FL_ELEM/SM_FL_ELEM); j++) {
		// 	#pragma hls PIPELINE II=1
		// 	fdata_v_t stmp = input.read();
		// 	for (int k = 0; k < SM_FL_ELEM; k++) {
		// 		#pragma HLS UNROLL
		// 		tmp[j * SM_FL_ELEM + k] = stmp[k];
		// 	}
		// }
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
