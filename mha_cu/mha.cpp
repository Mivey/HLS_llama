#include "mha.h"
// #include "forward.h"
#include "rope.h"
#include <algorithm>
#include <fenv.h>
#include <hls_math.h>
#include <limits>
#include <utils/x_hls_defines.h>



/*
In the orignial design, Karpathy used a 'Token-Major' memory layout. 
This means we read one head (64 elements) from each position (1 to max seq. length).
The major advantage of this approach is
- Write a 'page' easily. But KV output is relatively small (768 * 4 bytes) vs Read (768 * 256 * 4)
- 'Fast' on first few tokens
This is not good for a few reasons:
- If I want to do dataflow (do one head at a time) I must jump around, up to 1024 times
- If I want to enable burst reads, I can not move to softmax until I have all the calulations in mha_iterate done.
	- may be able to do 12 softmax at the same time, but SM is not the issue
	- means one long key cache read, then one long value cache read


	
The better way is to do a 'Head-Major' memory layout. 
Here we append a new sentence to each page. While this means writes become strided,
our reads are now linear bursts.
Advantages:
- Burst read: 
	- Each 'sentence' on the 'page' is for the given head.
- I can now dataflow (have iterate, softmax and ws all working) and overlap the MHA calculations
- Enables burst read from both Key and Value cache
Disadvantge:
- Write to caches are strided, but constant O(1)


	Book analogy:
    ===============================                ===============================       
        TOKEN-MAJOR MEMORY LAYOUT                      HEAD-MAJOR MEMORY LAYOUT         
    ===============================                ===============================       
                                                                                     
    |----HEAD SIZE --------|\                      |----HEAD SIZE ---------|\             
    |                      | \                     |                       | \             
    |  'Paperback book'    |  \                    |    'News paper'       |  \           
    |    approach          |   \                   |    Approach           |   \           
    H                      |    \                  P                       |    \         
    E    Page 1 of         |     \                 O    Page 1 of 12       |     \         
    A    Max Sequence      |      \                S    (Hidden dim)       |      \       
    D    Length (256)      |       \               |                       |       \       
    |                      |        |              |    256 sentences      |        |     
    |    12 sentences      |        |              |    per 'page          |        |     
    |    per 'page'        |        |              |                       |        |     
    -----------------------|        |              |-----------------------|        |     
    \                       \       |               \                       \       |     
     \                       \      |                \                       \      |     
      \                       \     |                 \                       \     |     
       \      POS              \    |                  \      HEAD             \    |     
        \                       \   |                   \                       \   |     
         \                       \  |                    \                       \  |     
          \                       \ |                     \                       \ |     
           \-----------------------\|                      \-----------------------\|    

(volumee 1 of 12) where each volume is a hidden layer
TOKEN MAJOR:	I read a sentence, (head size), the I turn the page (position)
HEAD MAJOR:		I read all the sentences (head size) on the page (position) before turning to the next page (HEAD)
    
*/

void wide_mha_iterate(hls::stream<my_float_t> &out, s_mfdata_v_t & query, s_mfdata_v_t &key_cache, const int POS){
	
	const size_t array_size = MODEL_HEAD_SIZE / MAX_FL_ELEM;
	const my_float_t score_scalar = 1.0f / sqrtf((float) MODEL_HEAD_SIZE);
	std::array<mfdata_v_t, (array_size)> query_arr;
	my_float_t att = 0.0f;
	// std::array<my_float_t, (array_size)> score;
	mfdata_v_t kc_arr[array_size];
	#pragma HLS ARRAY_PARTITION variable=kc_arr complete
	// #pragma HLS ARRAY_PARTITION variable=score complete
	#pragma HLS ARRAY_PARTITION variable=query_arr complete
	
	//get 64 elements of query
	query_loop:
	for (size_t j = 0; j < array_size; j++){
		#pragma HLS PIPELINE II=1
		query_arr[j] = query.read();
	}
// other way that may be faster is my_float_t tatt[4] 
// then tatt[i] =+ query_arr[i][n] * tmpb[n];
//then att += tatt[i] (hls unroll)
	pos_loop:
	for (size_t k = 0; k < POS; k++){
	#pragma HLS LOOP_TRIPCOUNT max=MODEL_SEQUENCE_LEN min=1
		//att_array adder tree
		#pragma HLS PIPELINE
		for (int j = 0; j < array_size; j++) {
			kc_arr[j] = key_cache.read();
		}
		
		att_loop:// no name b/c we unroll 
		for (size_t j = 0; j < array_size; j++){
			mfdata_v_t tmpa = query_arr[j];
			mfdata_v_t tmpb = kc_arr[j];
			for (int n = 0; n < MAX_FL_ELEM; n++) {
				att += tmpa[n] * tmpb[n];
			}
		}
		out.write(att * score_scalar);
		att = 0.0f;
	}
}

void old_wide_mha_iterate(hls::stream<my_float_t> &out, s_mfdata_v_t & query, s_mfdata_v_t &key_cache, const int POS){
	
	const size_t array_size = MODEL_HEAD_SIZE / MAX_FL_ELEM;
	const my_float_t score_scalar = 1.0f / sqrtf((float) MODEL_HEAD_SIZE);
	std::array<mfdata_v_t, (array_size)> query_arr;
	my_float_t att = 0.0f;
	my_float_t tatt[array_size];
	// mfdata_v_t matt[array_size];
	std::array<my_float_t, (array_size)> score;
	// #pragma HLS ARRAY_PARTITION variable=score complete
	// #pragma HLS ARRAY_PARTITION variable=query_arr complete
	
	//get 64 elements of query
	query_loop:
	for (size_t j = 0; j < array_size; j++){
		#pragma HLS PIPELINE II=1
		query_arr[j] = query.read();
	}

	pos_loop:
	for (size_t k = 0; k < POS; k++){
	#pragma HLS LOOP_TRIPCOUNT max=MODEL_SEQUENCE_LEN min=1
		//att_array adder tree
		att_loop:
		for (size_t j = 0; j < array_size; j++){
			
			#pragma HLS PIPELINE
			mfdata_v_t temp = query_arr[j] * key_cache.read();
			tatt[j] = temp.reduce_add();
			// for (int n = 0; n < MAX_FL_ELEM; n++) {
			// 	att += temp[n];
			// }
		}
		for (int j = 0; j < array_size; j++) {
			#pragma HLS UNROLL
			att += tatt[j];
		}
		out.write(att * score_scalar);
		att = 0.0f;
	}
}


void wide_mha_softmax(hls::stream<my_float_t> &att_out, hls::stream<my_float_t> &att_in, const int POS){
	
	const int tPOS = (POS / 4 + 1) * 4;

	my_float_t att_arr[MODEL_SEQUENCE_LEN + 4] = {std::numeric_limits<float>::lowest()};
	#pragma HLS ARRAY_PARTITION variable=att_arr cyclic factor=4

	my_float_t max_val = std::numeric_limits<float>::lowest();

	sm_new_intake_loop:
	for (int i = 0; i < POS; i++) {
	#pragma HLS LOOP_TRIPCOUNT max=MODEL_SEQUENCE_LEN min=1
		#pragma HLS PIPELINE
		my_float_t val = att_in.read();
		if (max_val < val) {
			max_val = val;
		}
		att_arr[i] = val;
	}
	my_float_t final_soft_sum = 0.0f;
	
	softmax_exp_loop:
	for (int i = 0; i < tPOS; i++) {
	#pragma HLS LOOP_TRIPCOUNT max=MODEL_SEQUENCE_LEN min=1
		#pragma HLS PIPELINE
		#pragma HLS UNROLL factor=4
		my_float_t calc = hls::expf((att_arr[i] - max_val));
		final_soft_sum += calc;
		att_arr[i] = calc;
	}
	my_float_t inv_soft_sum = 1.0f/final_soft_sum;

	softmax_normalize_loop:
	for (int i = 0; i < POS; i++) {
		// #pragma HLS LOOP_TRIPCOUNT max=(SEQ_LEN + 1) min=1
		#pragma HLS LOOP_TRIPCOUNT max=MODEL_SEQUENCE_LEN
		#pragma HLS PIPELINE
		my_float_t tempa = att_arr[i] * inv_soft_sum;
		att_out.write(tempa) ;
	}
}


void wide_mha_weighted_sum(s_mfdata_v_t &xb, hls::stream<my_float_t>  &att_in, s_mfdata_v_t &value_cache, const int POS){

	constexpr int ARR_SIZE = MODEL_HEAD_SIZE / MAX_FL_ELEM;
	mfdata_v_t xb_arr[ARR_SIZE] = {0.0f};
	my_float_t att_arr[MODEL_SEQUENCE_LEN] = {0.0f};
	mfdata_v_t vc_arr[ARR_SIZE];
	#pragma HLS ARRAY_PARTITION variable=xb_arr complete
	#pragma HLS ARRAY_PARTITION variable=vc_arr complete

	mha_pos_loop:
	for (size_t t = 0; t < POS; t++){
		#pragma HLS PIPELINE
		#pragma HLS LOOP_TRIPCOUNT max=(MODEL_SEQUENCE_LEN + 1) min=1
		read_vc_loop:
		for (int i = 0; i < ARR_SIZE; i++) {
			vc_arr[i] = value_cache.read();
		}
		// #pragma HLS PIPELINE II=8
		my_float_t val = att_in.read();
		for (size_t i = 0; i < ARR_SIZE; i++){
			// #pragma HLS PIPELINE
			#pragma HLS UNROLL
			xb_arr[i] += /*att_arr[t]*/ val * vc_arr[i];
		}
	}
	mha_ws_stream_out_xb: // set all values to zero
	for (int i = 0 ; i < ARR_SIZE; i++) {
		#pragma HLS PIPELINE II=1
		xb.write(xb_arr[i]);
	}
}

void old_wide_mha_weighted_sum(s_mfdata_v_t &xb, hls::stream<my_float_t>  &att_in, s_mfdata_v_t &value_cache, const int POS){

	constexpr int ARR_SIZE = MODEL_HEAD_SIZE / MAX_FL_ELEM;
	mfdata_v_t xb_arr[ARR_SIZE] = {0.0f};
	my_float_t att_arr[MODEL_SEQUENCE_LEN] = {0.0f};
	#pragma HLS ARRAY_PARTITION variable=xb_arr complete

	mha_pos_loop:
	for (size_t t = 0; t < POS; t++){
		#pragma HLS LOOP_TRIPCOUNT max=(MODEL_SEQUENCE_LEN + 1) min=1
		// #pragma HLS PIPELINE II=8
		my_float_t val = att_in.read();
		for (size_t i = 0; i < ARR_SIZE; i++){
			#pragma HLS PIPELINE
			xb_arr[i] += /*att_arr[t]*/ val * value_cache.read();
		}
	}
	mha_ws_stream_out_xb: // set all values to zero
	for (int i = 0 ; i < ARR_SIZE; i++) {
		#pragma HLS PIPELINE II=1
		xb.write(xb_arr[i]);
	}
}

void wide_mha_kernel(s_mfdata_v_t &xb, 
								s_mfdata_v_t &key_cache,
								s_mfdata_v_t &value_cache,
								s_mfdata_v_t &query,
								const int POS){

	
	mha_num_head_loop:
	for (size_t i = 0; i < MODEL_NUM_HEADS; i++) {
		#pragma HLS DATAFLOW
		
		hls::stream<my_float_t> mha_it_sm, att_sm_ws;
		#pragma HLS STREAM variable=mha_it_sm depth=8

		wide_mha_iterate(mha_it_sm, query, key_cache, POS);
		wide_mha_softmax(att_sm_ws, mha_it_sm, POS);
		wide_mha_weighted_sum(xb, att_sm_ws, value_cache, POS);
	}
}
void old_wide_mha_kernel(s_mfdata_v_t &xb, 
								s_mfdata_v_t &key_cache,
								s_mfdata_v_t &value_cache,
								s_mfdata_v_t &query,
								const int POS){
	mha_num_head_loop:
	for (size_t i = 0; i < MODEL_NUM_HEADS; i++) {
		#pragma HLS DATAFLOW
		
		hls::stream<my_float_t> mha_it_sm, att_sm_ws;
		#pragma HLS STREAM variable=mha_it_sm depth=8

		wide_mha_iterate(mha_it_sm, query, key_cache, POS);
		wide_mha_softmax(att_sm_ws, mha_it_sm, POS);
		old_wide_mha_weighted_sum(xb, att_sm_ws, value_cache, POS);
	}
}

void mha_kernel(mfdata_v_t *tokens, //6 mha_kernel
                mfdata_v_t *key_cache, 
                mfdata_v_t *value_cache, 
								mfdata_v_t *key_cache_in, mfdata_v_t *value_cache_in,
                const int POS, const int CURR_LAYER){
  

	constexpr int QUANT_DEPTH = MODEL_ELEMENTS * MODEL_ELEMENTS * MODEL_NUM_LAYERS / MAX_QUANT_ELEM;
	constexpr int SF_DEPTH = MODEL_ELEMENTS * MODEL_ELEMENTS * MODEL_NUM_LAYERS / (MODEL_SCALING_FACTOR * SM_FL_ELEM);
	constexpr int HD_QUANT_DEPTH = MODEL_HIDDEN_DIM * MODEL_ELEMENTS * MODEL_NUM_LAYERS  / MAX_QUANT_ELEM;
	constexpr int HD_SF_DEPTH = MODEL_HIDDEN_DIM * MODEL_ELEMENTS * MODEL_NUM_LAYERS / (MODEL_SCALING_FACTOR * SM_FL_ELEM);
	constexpr int CACHE_DEPTH = MODEL_ELEMENTS * MODEL_SEQUENCE_LEN * MODEL_NUM_LAYERS / MAX_FL_ELEM;
	constexpr int TOK_DEPTH = MODEL_ELEMENTS / MAX_FL_ELEM;
	constexpr int RMS_DEPTH = MODEL_ELEMENTS * MODEL_NUM_LAYERS / MAX_FL_ELEM;
	constexpr	int LOGITS_QUANT_DEPTH = MODEL_ELEMENTS * MODEL_TOKENS / MAX_QUANT_ELEM;
	constexpr int LOGITS_SF_DEPTH =  MODEL_ELEMENTS * MODEL_TOKENS / (MODEL_SCALING_FACTOR * SM_FL_ELEM);
	constexpr int LOGITS_DEPTH = MODEL_TOKENS / MAX_FL_ELEM;

	#pragma HLS INTERFACE mode=m_axi port=value_cache			bundle=vc_gemm		depth=CACHE_DEPTH			offset=slave max_read_burst_length=(4096/MAX_DW * 8)		max_write_burst_length=(4096/MAX_DW * 8)
	#pragma HLS INTERFACE mode=m_axi port=key_cache				bundle=kc_gemm		depth=CACHE_DEPTH			offset=slave max_read_burst_length=(4096/MAX_DW * 8)		max_write_burst_length=(4096/MAX_DW * 8)
	#pragma HLS INTERFACE mode=m_axi port=value_cache_in	bundle=token_gemm		depth=TOK_DEPTH			offset=slave max_read_burst_length=(4096/MAX_DW * 8)
	#pragma HLS INTERFACE mode=m_axi port=key_cache_in		bundle=token_gemm		depth=TOK_DEPTH			offset=slave max_read_burst_length=(4096/MAX_DW * 8)
	#pragma HLS INTERFACE mode=m_axi port=tokens					bundle=token_gemm	depth=TOK_DEPTH				offset=slave max_write_burst_length=(4096/MAX_DW * 8)  max_read_burst_length=(4096/MAX_DW * 8)
	/* **********************************************************************************/
	#pragma HLS INTERFACE mode=s_axilite port=tokens				 					bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=value_cache 						bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=key_cache								bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=value_cache_in 					bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=key_cache_in						bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=CURR_LAYER 							bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=POS 										bundle=control
	#pragma HLS INTERFACE mode=s_axilite port=return 									bundle=control
	
	s_mfdata_v_t xb_ws_q("WS to Quantizer for XB Stream");
	s_mfdata_v_t s_key_cache_to_kernel("From DDR to kernel key cache");
	s_mfdata_v_t s_value_cache_to_kernel("From DDR to kernel value cache");
	s_mfdata_v_t s_key_cache_in, s_value_cache_in, s_query, s_query_r, s_key_cache_in_r;

  #pragma HLS STABLE variable=POS
  #pragma HLS STABLE variable=CURR_LAYER
  #pragma HLS STREAM variable=xb_ws_q depth=MODEL_ELEMENTS / MAX_FL_ELEM
  #pragma HLS STREAM variable=tokens depth=MODEL_ELEMENTS / MAX_FL_ELEM

	#pragma HLS STREAM variable=s_key_cache_in depth=MODEL_ELEMENTS / MAX_FL_ELEM
	#pragma HLS STREAM variable=s_key_cache_in_r depth=MODEL_ELEMENTS / MAX_FL_ELEM
	#pragma HLS STREAM variable=s_value_cache_in depth=MODEL_ELEMENTS / MAX_FL_ELEM
	#pragma HLS STREAM variable=s_query depth=MODEL_ELEMENTS / MAX_FL_ELEM
	#pragma HLS STREAM variable=s_query_r depth=MODEL_ELEMENTS / MAX_FL_ELEM
	#pragma HLS STREAM variable=xb_ws_q depth=MODEL_ELEMENTS / MAX_FL_ELEM
	
	#pragma HLS BIND_STORAGE variable=s_key_cache_in type=fifo impl=bram
	#pragma HLS BIND_STORAGE variable=s_key_cache_in_r type=fifo impl=bram
	#pragma HLS BIND_STORAGE variable=s_value_cache_in type=fifo impl=bram
	#pragma HLS BIND_STORAGE variable=s_query type=fifo impl=bram
	#pragma HLS BIND_STORAGE variable=s_query_r type=fifo impl=bram
	#pragma HLS BIND_STORAGE variable=xb_ws_q type=fifo impl=bram

	#pragma HLS DATAFLOW
	tok_load_input(s_query_r, tokens);
	tok_load_input(s_key_cache_in_r, key_cache_in);
	tok_load_input(s_value_cache_in, value_cache_in);
	
	rope_kernel<MODEL_ELEMENTS>(s_query, s_query_r, POS);
	rope_kernel<MODEL_ELEMENTS>(s_key_cache_in, s_key_cache_in_r, POS);

	mha_WAR_store_load(key_cache, s_key_cache_to_kernel, s_key_cache_in, CURR_LAYER, POS);
	mha_WAR_store_load(value_cache, s_value_cache_to_kernel, s_value_cache_in, CURR_LAYER, POS);
	
	wide_mha_kernel(xb_ws_q, s_key_cache_to_kernel, s_value_cache_to_kernel, s_query, POS + 1);
	
	store_output(tokens, xb_ws_q, MODEL_ELEMENTS);

	return;
}

