
#include "mha_forward.h"

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
TOKEN MAJOR:  I read a sentence, (head size), the I turn the page (position)
HEAD MAJOR:    I read all the sentences (head size) on the page (position) before turning to the next page (HEAD)
    
*/


void mha_iterate(hls::stream<float_t> &out, hls::stream<float_t> &s_max, s_mfdata_v_t &query, s_mfdata_v_t &key_cache, const int POS){
  
  const size_t array_size = MODEL_HEAD_SIZE / MAX_FL_ELEM;
  const float_t score_scalar = 1.0f / sqrtf((float_t) MODEL_HEAD_SIZE);
  
	for (int i = 0; i < MODEL_NUM_HEADS; i++) {
		std::array<mfdata_v_t, (array_size)> query_arr;
		float_t att = 0.0f;
		float_t max = std::numeric_limits<float_t>::lowest();
		float_t patt[array_size]{};
		#pragma HLS ARRAY_PARTITION variable=patt dim=1 type=complete
		#pragma HLS ARRAY_PARTITION variable=query_arr dim=1 type=complete
	
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
			
			att_loop:// no name b/c we unroll 
			for (size_t j = 0; j < array_size; j++){
				mfdata_v_t tmpa = query_arr[j];
				mfdata_v_t tmpb = key_cache.read();
				for (int n = 0; n < MAX_FL_ELEM; n++) {
					patt[j] += tmpa[n] * tmpb[n];
				}
			}
			for (int j = 0; j < array_size; j++) {
				#pragma hls UNROLL
				att += patt[j];
				patt[j] = 0.0f;
			}
			float_t tmp_comp = att * score_scalar;
			out.write(tmp_comp);
			max = (max < tmp_comp) ? tmp_comp : max;
			att = 0.0f;
		}
		s_max.write(max);
	}
}

void mha_softmax(hls::stream<my_float_t> &att_out, hls::stream<float_t> &iss, hls::stream<float_t> &s_max, hls::stream<my_float_t> &att_in, const int POS){
  
  int nPOS = POS;
	for (int ii = 0; ii < MODEL_NUM_HEADS; ii++) {
		
		float_t part_soft_sum[8]{};// = 0.0f;
		#pragma hls ARRAY_PARTITION variable=part_soft_sum dim=1 type=complete
		float_t max_val = s_max.read();
		
		softmax_exp_loop:
		for (int i = 0; i < POS; i++) {
		#pragma HLS LOOP_TRIPCOUNT max=MODEL_SEQUENCE_LEN min=1
			#pragma HLS PIPELINE II=1
			my_float_t calc = hls::expf((att_in.read() - max_val));
			int32_t bar = i % 8;
			
			part_soft_sum[bar] += calc;
			att_out.write(calc);
		}
		
		float_t final_soft_sum = 0.0f;
		for (int i = 0; i < 8; i++) {
			#pragma HLS UNROLL
			final_soft_sum += part_soft_sum[i];
		}
		my_float_t inv_soft_sum = 1.0f/final_soft_sum;
		iss.write(inv_soft_sum);
	}
}
void mha_weighted_sum(s_mfdata_v_t &xb, hls::stream<my_float_t>  &att_in, hls::stream<my_float_t>  &inv_soft_sum, s_mfdata_v_t &value_cache, const int POS){

  constexpr int ARR_SIZE = MODEL_HEAD_SIZE / MAX_FL_ELEM;

	for (int i = 0; i < MODEL_NUM_HEADS; i++) {
		mfdata_v_t xb_arr[ARR_SIZE] = {0.0f};
		// mfdata_v_t vc_arr[ARR_SIZE];
		#pragma HLS ARRAY_PARTITION variable=xb_arr complete
		// #pragma HLS ARRAY_PARTITION variable=vc_arr complete
		mha_pos:
		for (size_t t = 0; t < POS; t++){
			#pragma HLS PIPELINE
			#pragma HLS LOOP_TRIPCOUNT max=(MODEL_SEQUENCE_LEN + 1) min=1
			my_float_t val = att_in.read();
			for (size_t ii = 0; ii < ARR_SIZE; ii++){
				#pragma HLS UNROLL
				xb_arr[ii] += /*att_arr[t]*/ val * value_cache.read();// vc_arr[i];
			}
		}
		
		float_t iss = inv_soft_sum.read();
		mha_ws_stream_out_xb: // set all values to zero
		for (int jj = 0 ; jj < ARR_SIZE; jj++) {
			#pragma HLS PIPELINE II=1
			xb.write(xb_arr[jj] * iss);
		}
	}
}

void mha_init(s_mfdata_v_t &s_query, s_mfdata_v_t &s_key_cache_in, s_mfdata_v_t &s_value_cache_in, fdata_v_t *tokens, const int POS ){
	
	s_mfdata_v_t s_key_cache_in_r, s_query_r;
	#pragma HLS STREAM variable=s_query_r depth=MODEL_ELEMENTS / MAX_FL_ELEM //good
	#pragma HLS BIND_STORAGE variable=s_query_r type=fifo
	#pragma HLS STREAM variable=s_key_cache_in_r depth=MODEL_ELEMENTS / MAX_FL_ELEM  //good
	#pragma HLS BIND_STORAGE variable=s_key_cache_in_r type=fifo

	mm2s_vec_up(s_query_r, tokens, 0, (MODEL_ELEMENTS / MAX_FL_ELEM));
	rope_kernel(s_query, s_query_r, POS);
	mm2s_vec_up(s_query_r, tokens, (MODEL_ELEMENTS / SM_FL_ELEM), (INTERNAL_DATA_SIZE/(SM_FL_ELEM * 2)), (MODEL_ELEMENTS / MAX_FL_ELEM));
	rope_kernel(s_key_cache_in, s_query_r, POS); // check on and then update this
	mm2s_vec_up(s_value_cache_in, tokens, ((INTERNAL_DATA_SIZE / 2 + MODEL_ELEMENTS / 2) / SM_FL_ELEM), (MODEL_ELEMENTS / MAX_FL_ELEM) );
}

void mha_kernel(s_fdata_v_t &output,//
                fdata_v_t *tokens, //6 mha_kernel
                mfdata_v_t *key_cache, 
                mfdata_v_t *value_cache, 
                const int POS, const int CURR_LAYER){

  const size_t VAL_START = (INTERNAL_DATA_SIZE / 2) / MODEL_HEAD_SIZE + MODEL_NUM_HEADS / 2;
  const size_t KEY_START = MODEL_NUM_HEADS;

	
	s_mfdata_v_t xb_ws_q("WS to Quantizer for XB Stream");
	s_fdata_v_t max_tok_out;
	hls::stream<my_float_t> s_max_val;
	hls::stream<my_float_t> s_iss_val;
	#pragma HLS STREAM variable=s_max_val        depth=4
	#pragma HLS STREAM variable=s_iss_val        depth=4
	#pragma HLS STREAM variable=max_tok_out      depth=32
	s_mfdata_v_t s_key_cache_to_kernel("From DDR to kernel key cache");
	s_mfdata_v_t s_value_cache_to_kernel("From DDR to kernel value cache");
	s_mfdata_v_t s_key_cache_in, s_query, s_value_cache_in, s_key_cache_sl, s_value_cache_sl;
	s_mfdata_v_t s_xb_output;

	#pragma HLS STABLE variable=POS
	#pragma HLS STABLE variable=CURR_LAYER

	#pragma HLS STREAM variable=s_key_cache_in depth=MODEL_ELEMENTS / MAX_FL_ELEM  //good
	#pragma HLS STREAM variable=s_value_cache_sl depth=64  //good
	#pragma HLS STREAM variable=s_key_cache_sl depth=64 //good
	// #pragma HLS STREAM variable=output depth=MODEL_ELEMENTS / SM_FL_ELEM
	#pragma HLS STREAM variable=s_value_cache_in depth=MODEL_ELEMENTS / MAX_FL_ELEM  //good
	#pragma HLS STREAM variable=s_query depth=MODEL_ELEMENTS / MAX_FL_ELEM //good
	#pragma HLS STREAM variable=xb_ws_q depth=32 //good
	#pragma HLS STREAM variable=s_key_cache_to_kernel depth=4096 //good
	#pragma HLS STREAM variable=s_value_cache_to_kernel depth=4096 //good

	#pragma HLS BIND_STORAGE variable=s_query type=fifo
	#pragma HLS BIND_STORAGE variable=s_key_cache_to_kernel type=fifo impl=uram
	#pragma HLS BIND_STORAGE variable=s_value_cache_to_kernel type=fifo impl=uram

	
	hls::stream<my_float_t> mha_it_sm, att_sm_ws;
	#pragma HLS STREAM variable=mha_it_sm depth=1536
	#pragma HLS BIND_STORAGE variable=mha_it_sm type=fifo impl=bram
	#pragma HLS STREAM variable=att_sm_ws depth=1536
	#pragma HLS BIND_STORAGE variable=att_sm_ws type=fifo impl=bram


	
	#pragma HLS DATAFLOW

	mha_init(s_query, s_key_cache_in, s_value_cache_in, tokens, POS);
	
	mha_WAR_cache_read(s_key_cache_sl, key_cache, CURR_LAYER, POS);
	mha_WAR_store_load(key_cache, s_key_cache_to_kernel, s_key_cache_in, s_key_cache_sl, CURR_LAYER, POS);
	
	mha_WAR_cache_read(s_value_cache_sl, value_cache, CURR_LAYER, POS);
	mha_WAR_store_load(value_cache, s_value_cache_to_kernel, s_value_cache_in, s_value_cache_sl, CURR_LAYER, POS);
	
	// mha_WAR_store_load(key_cache, s_key_cache_to_kernel, s_key_cache_in, CURR_LAYER, POS); 
	// mha_WAR_store_load(value_cache, s_value_cache_to_kernel, s_value_cache_in, CURR_LAYER, POS);
	mha_iterate(mha_it_sm, s_max_val, s_query, s_key_cache_to_kernel, POS + 1);
	mha_softmax(att_sm_ws, s_iss_val, s_max_val, mha_it_sm, POS + 1);
	mha_weighted_sum(xb_ws_q, att_sm_ws, s_iss_val, s_value_cache_to_kernel, POS + 1);
	vec_down_converter(output, xb_ws_q, (MODEL_ELEMENTS / SM_FL_ELEM));
  return;
}
// https://claude.ai/share/a3b84fa5-b5b4-49dd-843d-f39adf10cff6
// Your instinct is right, but the report points at different functions than you'd expect. Here's where `mha_kernel`'s 198 DSP and 51,182 cycles actually go:

// | Process | DSP | LUT | cycles | per head |
// |---|---|---|---|---|
// | `rope_kernel` ×2 | **68** | **74,794** | 234 | — |
// | `mha_weighted_sum` | **92** | 38,518 | 49,537 | 4,128 |
// | `mha_iterate` | 25 | 27,893 | 49,585 | 4,132 |
// | `mha_WAR_store_load` ×2 | 0 | 51,729 | 51,181 | 4,265 |
// | `mha_softmax` | 13 | 2,077 | 12,817 | 1,068 |

// Worth noting up front: DSP is now 421/464 (90%) but LUT is 326,352/150,272 — **217%** — and `mha_kernel` alone is 197,439 of that. So LUT is the binding problem, and the same three fixes help both.

// ## The pause between heads is in `mha_WAR_store_load`

// `VITIS_LOOP_372_1` is 12 iterations with an iteration latency of 4,265 and pipelining marked "no". Inside it, `fw_mha_pos` (the burst read) is 4,103 and `fw_mha_new` (the write) is 7. That leaves ~155 cycles per head of non-overlapped address setup, burst issue, and a read→write→read turnaround on the same m_axi port. At a realistic POS of ~32 the burst itself is only ~131 cycles, so the fixed overhead exceeds the useful work — which is exactly the stutter you're describing.

// This is where your idea pays off, just applied to a different function. Split it into two dataflow processes: a reader that issues all 12 head bursts back to back, and a writer that appends the new K/V position afterwards. Right now they alternate per head on one port, which is also why the burst report flags `Stride is incompatible` on `VITIS_LOOP_372_1` — the write sitting between reads blocks coalescing. Bumping `num_read_outstanding` on `kc_gemm`/`vc_gemm` lets head *i+1*'s burst be in flight while head *i* drains.

// ## `rope_kernel` is the DSP and LUT hog

// 34 DSP and 37,397 LUT per instance, two instances, for 234 cycles of work out of 51,182. Each evaluates two `generic_sincos` per element across all 192 elements.

// But the rotation angles depend only on POS and the head-dimension index. There are only `MODEL_HEAD_SIZE/2` = 32 distinct (sin, cos) pairs for a given POS, and POS is `#pragma HLS STABLE` for the whole call. You're computing 32 values 192 times, twice over, in two separate instances that need identical angles.

// Hoist them: build a 32-entry table once at the top of `mha_kernel` and feed both rope instances from it. That's ~68 DSP and ~70k LUT back, which alone takes you to roughly 355/464 DSP and removes a third of the LUT overflow.

// ## `mha_weighted_sum` runs 92 DSP at 25% duty

// `mha_pos` reports `II=4, Resource Limitation`. The `UNROLL` over `ARR_SIZE=4` with 16-wide vectors creates 64 parallel fp32 lanes, but four reads from one `value_cache` stream can't issue in a single cycle, so II is pinned at 4 no matter how many multipliers you throw at it. Three cycles in four, 48 of them idle.

// Flatten the head-size unroll into the position loop:

// ```cpp
// mfdata_v_t xb_acc[2][ARR_SIZE] = {0.0f};
// #pragma HLS ARRAY_PARTITION variable=xb_acc complete
// my_float_t val;
// mha_pos:
// for (int t = 0; t < POS * ARR_SIZE; t++) {
//   #pragma HLS PIPELINE II=1
//   const int ii = t & (ARR_SIZE - 1);
//   const int bank = (t >> 2) & 1;            // distance-8 accumulator chain
//   if (ii == 0) val = att_in.read();
//   xb_acc[bank][ii] += val * value_cache.read();
// }
// ```

// Same cycle count, 16 MACs instead of 64. The two-deep banking gives the accumulator a dependency distance of 8, which clears fp32 add latency at 250 MHz — distance 4 would be marginal. Sum the banks in the existing `mha_ws_stream_out_xb` loop. Expect ~92 → ~28 DSP.

// `mha_iterate` is already frugal at 25 DSP, because the serial `patt[j] +=` chain forced HLS to time-multiplex rather than replicate. But it's also II=4 with an iteration latency of 28, and the same flattening drops that to around 10 — which is where its per-head pipeline drain cost lives.

// ## The larger restructure

// `mha_iterate` has to finish an entire head before it writes `s_max`, and `mha_softmax` blocks on that read. That's why `mha_it_sm` and `att_sm_ws` need to be 1,536 deep in BRAM. A one-pass online softmax — running max with rescale — removes the sync point, both FIFOs, and lets iterate, softmax and weighted-sum fuse into a single pass over the KV stream. That's the direction you were already heading with the streaming softmax idea, and it's the right end state. I'd do it after the three fixes above, though, since those are local, low-risk, and together should get you back under the LUT ceiling.