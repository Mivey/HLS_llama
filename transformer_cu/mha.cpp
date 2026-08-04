#include "mha.h"
// #include "forward.h"
#include "mha_forward.h"
// #include "quantizer.h"
#include "rope.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
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

/*
NEEDS REWRITE:
	1. should be compatible with llama3
		- look at how the small version of tinystories works
	2. crete second softmax function that streams inputs and outputs from aie function
	3. should have only ONE dataflow pragma
	4. incoporate quantizer
	5. break up load and store?

	Looks like this:
	for (int i = 0; i < LOOP_CNT; i++):
		#pragma hls dataflow
		 Get 64 values from query
		 ** do to: learn how KV work here and if I can grab 16 values**
		 rope q, k
		 ** speculation aie  start **
		 leave WAR main idea alone, slight modification to send 12/32/whatever
		 	needs to read/write bf16 for additional size savings and speed
		 implement mha iterate, softmax and ws in aie
		 	inputs : q (from rope) k, v (from WAR), POS
			output : xb (n sized bf16 vector)
		** speculation aie end **
		quantizer
			bf16 to float?
*/

void wide_mha_iterate(hls::stream<float_t> &out, s_mfdata_v_t & query, s_mfdata_v_t &key_cache, const int POS){
	
	const size_t array_size = MODEL_HEAD_SIZE / MAX_FL_ELEM;
	const float_t score_scalar = 1.0f / sqrtf((float) MODEL_HEAD_SIZE);
	std::array<mfdata_v_t, (array_size)> query_arr;
	float_t att = 0.0f;
	
	mfdata_v_t kc_arr[array_size];
	#pragma HLS ARRAY_PARTITION variable=kc_arr complete
	#pragma HLS ARRAY_PARTITION variable=query_arr complete
	
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
		#pragma HLS PIPELINE
		for (int j = 0; j < array_size; j++) {
			mfdata_v_t var = key_cache.read();
			kc_arr[j] = var;
		}
		
		att_loop:// no name b/c we unroll 
		for (size_t j = 0; j < array_size; j++){
			mfdata_v_t tmpa = query_arr[j];
			mfdata_v_t tmpb = kc_arr[j];
			for (int n = 0; n < MAX_FL_ELEM; n++) {
				att += tmpa[n] * tmpb[n];
			}
		}
		float_t write_out = att * score_scalar;
		out.write(write_out);
		att = 0.0f;
	}
}

void wide_mha_softmax(hls::stream<float_t> &att_out, hls::stream<float_t> &att_in, const int POS){
	
	const int tPOS = (POS / 4 + 1) * 4;
	int nPOS = POS;
	const int LATENCY = 4;//MAX_FL_ELEM * 2;
	float_t att_arr[MODEL_SEQUENCE_LEN] = {std::numeric_limits<float>::lowest()};
	float_t fatt_arr[MODEL_SEQUENCE_LEN]{};
	#pragma HLS ARRAY_PARTITION variable=att_arr cyclic factor=4
	#pragma HLS ARRAY_PARTITION variable=fatt_arr cyclic factor=4
	float_t max_val = std::numeric_limits<float>::lowest();

	int elements = 0;
	int in_off = 0;
	float_t tmp_arr[32];
	#pragma HLS ARRAY_PARTITION variable=tmp_arr complete
	multi_read:
	while (nPOS > 0) {
	#pragma HLS LOOP_TRIPCOUNT max=MODEL_SEQUENCE_LEN/32 min=1
		if (nPOS > 32) {
			elements = 32;
		} else {
			elements = nPOS;
		}
		nPOS = nPOS - elements;
		
		read_n_comp:
		for (int i = 0; i < 32; i++) {
		#pragma HLS LOOP_TRIPCOUNT max=32 min=1
			#pragma HLS PIPELINE II=1

			if (i < elements) {
				tmp_arr[i] = att_in.read();
				att_arr[in_off + i] = tmp_arr[i];	
			}else {
				tmp_arr[i] = std::numeric_limits<float>::lowest();
			}
		}
		
		for (int stride = 16; stride > 0; stride >>=1){
			#pragma HLS UNROLL
			for (int j = 0; j < stride; j++) {
				#pragma HLS UNROLL
			tmp_arr[j] = (tmp_arr[j] < tmp_arr[j + stride]) ? tmp_arr[j + stride] : tmp_arr[j];
			}
		}

		max_val = (max_val < tmp_arr[0]) ? tmp_arr[0] : max_val;		
		in_off +=32;
	}
	float_t final_soft_sum = 0.0f;
	
	softmax_exp_loop:
	for (int i = 0; i < tPOS; i++) {
	#pragma HLS LOOP_TRIPCOUNT max=MODEL_SEQUENCE_LEN min=1
		#pragma HLS PIPELINE
		#pragma HLS UNROLL factor=4
		float_t calc = hls::expf((att_arr[i] - max_val));
		final_soft_sum += calc;
		fatt_arr[i] = calc;
	}
	float_t inv_soft_sum = 1.0f/final_soft_sum;

	softmax_normalize_loop:
	for (int i = 0; i < POS; i++) {
		// #pragma HLS LOOP_TRIPCOUNT max=(SEQ_LEN + 1) min=1
		#pragma HLS LOOP_TRIPCOUNT max=MODEL_SEQUENCE_LEN
		#pragma HLS PIPELINE
		float_t tempa = fatt_arr[i] * inv_soft_sum;
		att_out.write(tempa) ;
	}
}

void wide_mha_weighted_sum(s_fdata_v_t &xb, hls::stream<float_t>  &att_in, s_mfdata_v_t &value_cache, const int POS){

	constexpr int ARR_SIZE = MODEL_HEAD_SIZE / MAX_FL_ELEM;
	mfdata_v_t xb_arr[ARR_SIZE]{};
	mfdata_v_t vc_arr[ARR_SIZE];
	#pragma HLS ARRAY_PARTITION variable=xb_arr complete
	#pragma HLS ARRAY_PARTITION variable=vc_arr complete

	mha_pos:
	for (size_t t = 0; t < POS; t++){
		#pragma HLS PIPELINE
		#pragma HLS LOOP_TRIPCOUNT max=(MODEL_SEQUENCE_LEN + 1) min=1
		float_t val = att_in.read();
		for (size_t ii = 0; ii < ARR_SIZE; ii++){
			#pragma HLS UNROLL
			mfdata_v_t vc_dat = value_cache.read();
			mfdata_v_t tmp_xb;
			mfdata_v_t xb = xb_arr[ii];
			for (size_t jj = 0; jj < MAX_FL_ELEM; jj++) {
				#pragma HLS UNROLL
				float_t foo = val * vc_dat[jj] + xb[jj];
				my_float_t bar = foo;
				tmp_xb[jj] = bar;
			}
			xb_arr[ii] = tmp_xb;//xb_arr[ii] + val * value_cache.read();// vc_arr[i];
		}
	}
	mha_ws_stream_out_xb: // set all values to zero
	for (int jj = 0 ; jj < ARR_SIZE; jj++) {
		mfdata_v_t tmp = xb_arr[jj];
		for (int k = 0; k < (MAX_FL_ELEM / SM_FL_ELEM); k++) {
		#pragma HLS PIPELINE II=1
			fdata_v_t sm_tmp;
			for (int l = 0; l < SM_FL_ELEM; l++) {
				#pragma HLS UNROLL
				sm_tmp[l] = tmp[SM_FL_ELEM * k + l];
			}
			xb.write(sm_tmp);
		}
	}
}

void mha_kernel(s_fdata_v_t &output,
								fdata_v_t *tokens, //6 mha_kernel
                mfdata_v_t *key_cache, 
                mfdata_v_t *value_cache, 
                const int POS, const int CURR_LAYER){

	s_fdata_v_t xb_ws_q("WS to Quantizer for XB Stream");
	s_fdata_v_t max_tok_out;
	hls::stream<my_float_t> q_max_val;
	#pragma HLS STREAM variable=q_max_val				depth=4
	#pragma HLS STREAM variable=max_tok_out			depth=32
	s_mfdata_v_t s_key_cache_to_kernel("From DDR to kernel key cache");
	s_mfdata_v_t s_value_cache_to_kernel("From DDR to kernel value cache");
	s_mfdata_v_t s_key_cache_in, s_query, s_value_cache_in, s_query_r, s_key_cache_in_r;
	const size_t VAL_START = (INTERNAL_DATA_SIZE / 2) / MODEL_HEAD_SIZE + MODEL_NUM_HEADS / 2;
	const size_t KEY_START = MODEL_NUM_HEADS;

  #pragma HLS STABLE variable=POS
  #pragma HLS STABLE variable=CURR_LAYER

	#pragma HLS STREAM variable=s_key_cache_in depth=MODEL_HEAD_SIZE / MAX_FL_ELEM  //good
	#pragma HLS STREAM variable=s_key_cache_in_r depth=MODEL_HEAD_SIZE / MAX_FL_ELEM  //good
	#pragma HLS STREAM variable=output depth=MODEL_ELEMENTS / SM_FL_ELEM
	#pragma HLS STREAM variable=s_value_cache_in depth=MODEL_HEAD_SIZE / MAX_FL_ELEM  //good
	#pragma HLS STREAM variable=s_query depth=MODEL_HEAD_SIZE / MAX_FL_ELEM //good
	#pragma HLS STREAM variable=s_query_r depth=MODEL_HEAD_SIZE / MAX_FL_ELEM //good
	#pragma HLS STREAM variable=xb_ws_q depth=8 //good
	#pragma HLS STREAM variable=s_key_cache_to_kernel depth=1024 //good
	#pragma HLS STREAM variable=s_value_cache_to_kernel depth=1024 //good
	
	#pragma HLS BIND_STORAGE variable=s_key_cache_in_r type=fifo impl=bram
	#pragma HLS BIND_STORAGE variable=s_query type=fifo impl=bram
	#pragma HLS BIND_STORAGE variable=s_query_r type=fifo impl=bram
	#pragma HLS BIND_STORAGE variable=s_key_cache_to_kernel type=fifo impl=uram
	#pragma HLS BIND_STORAGE variable=s_value_cache_to_kernel type=fifo impl=uram

	mha_num_head:
	for (size_t i = 0; i < MODEL_NUM_HEADS; i++) {
		#pragma HLS DATAFLOW
		
		hls::stream<float_t> mha_it_sm, att_sm_ws;
		#pragma HLS STREAM variable=mha_it_sm depth=256
		#pragma HLS BIND_STORAGE variable=mha_it_sm type=fifo impl=bram
		#pragma HLS STREAM variable=att_sm_ws depth=256
		#pragma HLS BIND_STORAGE variable=att_sm_ws type=fifo impl=bram

		mha_input_data(s_query_r, tokens, i * MODEL_HEAD_SIZE, 0); //read query first
		mha_input_data(s_key_cache_in_r, tokens, (i + KEY_START) * MODEL_HEAD_SIZE, 1); //key
		mha_input_data(s_value_cache_in, tokens, (i + VAL_START) * MODEL_HEAD_SIZE, 0); // value
		rope_kernel<my_float_t, MAX_FL_ELEM, MODEL_HEAD_SIZE>(s_query, s_query_r, POS);
		rope_kernel<my_float_t, MAX_FL_ELEM, MODEL_HEAD_SIZE>(s_key_cache_in, s_key_cache_in_r, POS);
		mha_WAR_store_load(key_cache, s_key_cache_to_kernel, s_key_cache_in, CURR_LAYER, POS, i);
		mha_WAR_store_load(value_cache, s_value_cache_to_kernel, s_value_cache_in, CURR_LAYER, POS, i);
		wide_mha_iterate(mha_it_sm, s_query, s_key_cache_to_kernel, POS + 1);
		wide_mha_softmax(att_sm_ws, mha_it_sm, POS + 1);
		wide_mha_weighted_sum(output, att_sm_ws, s_value_cache_to_kernel, POS + 1);
	}
	return;
}
