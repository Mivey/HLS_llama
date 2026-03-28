
#include <cstddef>
#include <hls_vector.h>
#include "mha_forward.h"

struct embedding_t {
	float prob;
	int id;
};

template<size_t N>
void sampler_kernel(hls::stream<embedding_t> &max_tok, hls::vector<my_float_t, N> *mm_in){
	
	const int vCount = MODEL_TOKENS / N;
	const int stride = 125;
	const int arr_idx = vCount / stride;
	embedding_t val[arr_idx];
	
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < stride; j++) {
			
			get_arr_idx_elem:
			for (int k = 0; k < arr_idx; k++) {
				#pragma HLS PIPELINE II=1
				int mm_arr_idx = j + stride * k;
				val[k].prob = mm_in[mm_arr_idx][i];
				val[k].id = i + mm_arr_idx * 4;
			}
			
			for (int k = (arr_idx >>1); k > 0; k++) {
				#pragma HLS UNROLL
				for (int l = 0; l < k; l++) {
					#pragma HLS UNROLL
					val[l] = (val[l].prob > val[l + k].prob) ? val[l] : val[l + k];
				}
			}
			max_tok.write(val[0]);
		}
	}
	
	// read each value into 64 bit array, partition = complete
	
	// find max
}


void wide_mha_softmax(hls::stream<my_float_t> &att_out, hls::stream<my_float_t> &att_in, const int POS){
	
	const int tPOS = (POS / 4 + 1) * 4;
	int nPOS = POS;
	const int LATENCY = 4;//MAX_FL_ELEM * 2;
	my_float_t att_arr[MODEL_SEQUENCE_LEN] = {std::numeric_limits<float>::lowest()};
	my_float_t fatt_arr[MODEL_SEQUENCE_LEN] = {0.0f};
	#pragma HLS ARRAY_PARTITION variable=att_arr cyclic factor=4
	#pragma HLS ARRAY_PARTITION variable=fatt_arr cyclic factor=4
	my_float_t max_val = std::numeric_limits<float>::lowest();

	int elements = 0;
	int in_off = 0;
	my_float_t tmp_arr[32];
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
				// #pragma HLS PIPELINE II=1
				#pragma HLS UNROLL
			tmp_arr[j] = (tmp_arr[j] < tmp_arr[j + stride]) ? tmp_arr[j + stride] : tmp_arr[j];
			}
		}

		max_val = (max_val < tmp_arr[0]) ? tmp_arr[0] : max_val;		
		in_off +=32;
	}
	my_float_t final_soft_sum = 0.0f;
	
	softmax_exp_loop:
	for (int i = 0; i < tPOS; i++) {
	#pragma HLS LOOP_TRIPCOUNT max=MODEL_SEQUENCE_LEN min=1
		#pragma HLS PIPELINE
		#pragma HLS UNROLL factor=4
		my_float_t calc = hls::expf((att_arr[i] - max_val));
		final_soft_sum += calc;
		fatt_arr[i] = calc;
	}
	my_float_t inv_soft_sum = 1.0f/final_soft_sum;

	softmax_normalize_loop:
	for (int i = 0; i < POS; i++) {
		// #pragma HLS LOOP_TRIPCOUNT max=(SEQ_LEN + 1) min=1
		#pragma HLS LOOP_TRIPCOUNT max=MODEL_SEQUENCE_LEN
		#pragma HLS PIPELINE
		my_float_t tempa = fatt_arr[i] * inv_soft_sum;
		att_out.write(tempa) ;
	}
}
