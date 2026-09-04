#ifndef MARK_ROPE
#define MARK_ROPE
#include "../forward.h"
#include <cstddef>

template<int HEAD>
void init_freq_arr(float arr[HEAD]){
  for (int i = 0; i < HEAD; i++) {
  arr[i] = 1.0f / hls::powf(10000.0f, ((i) / (float) MODEL_HEAD_SIZE));
  }
}

// template<int N_DIM>
// void rope_kernel (s_mfdata_v_t &o, s_mfdata_v_t &in, const int POS){
// 	float arr[MODEL_HEAD_SIZE];
// 	init_freq_arr<MODEL_HEAD_SIZE>(arr);
// 	ROPE_MAIN:
// 	for (int i = 0; i < (N_DIM / MAX_FL_ELEM); i++) {
// #pragma HLS loop_flatten 
//  // increment by number of element in fdata_v_t
	
// 	int k = i * MAX_FL_ELEM;
// 		mfdata_v_t tmp = in.read();
// 		mfdata_v_t tmp_o;
// 		head_dim_unroll_loop:
// 		for (int j = 0 ; j < (MAX_FL_ELEM / 2); j++) {
// 			#pragma HLS PIPELINE
// 			int head_dim = (k + j * 2) % MODEL_HEAD_SIZE;
// 			float freq =  arr[head_dim]; /*1.0f / hls::powf(10000.0f, (float)head_dim/HEAD_SIZE);*/ 
// 			float val = POS * freq;
// 			float fcr;
// 			float fci;
// 			hls::sincosf(val, &fci, &fcr);
// 			float v0 = tmp[j * 2 + 0];
// 			float v1 = tmp[j * 2 + 1];
// 			tmp_o[j * 2 + 0] = v0 * fcr - v1 * fci;
// 			tmp_o[j * 2 + 1] = v0 * fci + v1 * fcr;
// 		}
// 		o.write(tmp_o);
// 	}
// }

template<typename T, size_t N, int N_DIM = MODEL_ELEMENTS>
void rope_kernel (hls::stream<hls::vector<T, N>> &o, hls::stream<hls::vector<T, N>> &in, const int POS){
	float arr[MODEL_HEAD_SIZE];
	init_freq_arr<MODEL_HEAD_SIZE>(arr);
	ROPE_MAIN:
	for (int i = 0; i < (N_DIM / N); i++) {
		#pragma HLS loop_flatten 
 // increment by number of element in fdata_v_t
	
	int k = i * N;
		hls::vector<T, N> tmp = in.read();
		hls::vector<T, N> tmp_o;
		head_dim_unroll_loop:
		for (int j = 0 ; j < (N / 2); j++) {
			#pragma HLS PIPELINE
			int head_dim = (k + j * 2) % MODEL_HEAD_SIZE;
			float freq =  arr[head_dim]; /*1.0f / hls::powf(10000.0f, (float)head_dim/HEAD_SIZE);*/ 
			float val = POS * freq;
			float fcr;
			float fci;
			hls::sincosf(val, &fci, &fcr);
			float v0 = tmp[j * 2 + 0];
			float v1 = tmp[j * 2 + 1];
			tmp_o[j * 2 + 0] = v0 * fcr - v1 * fci;
			tmp_o[j * 2 + 1] = v0 * fci + v1 * fcr;
		}
		o.write(tmp_o);
	}
}


template<int N_DIM>
void narrow_rope_kernel(s_mfdata_v_t &out, s_mfdata_v_t &in, const int POS){

	float arr[MODEL_HEAD_SIZE];
	init_freq_arr<MODEL_HEAD_SIZE>(arr);
	my_float_t vec[MODEL_HEAD_SIZE];
	#pragma HLS ARRAY_PARTITION variable=vec type=complete //dim=1 factor=2 
	
	// mfdata_v_t temp;
my_float_t fcr[MODEL_HEAD_SIZE];
my_float_t fci[MODEL_HEAD_SIZE];
	for (int i = 0; i < MODEL_HEAD_SIZE; i++) {
		
			my_float_t val = arr[i] * POS;
			float tfcr, tfci;
			hls::sincosf(val, &tfci, &tfcr);
			fcr[i] = tfcr;
			fci[i] = tfci;
	}
	
	all_the_head_loop:
	for (size_t i = 0; i < MODEL_NUM_HEADS; i++) {
		
		mfdata_v_t temp = in.read();
		value_in_head_loop:
		for (size_t j = 0; j < MODEL_HEAD_SIZE; j+=2) {
			my_float_t v0 = temp[j + 0];
			my_float_t v1 = temp[j + 1];

			vec[j + 0] = v0 * fcr[j] - v1 * fci[j];
			vec[j + 1] = v0 * fci[j] + v1 * fcr[j];
		}

		for (size_t j = 0; j < (MODEL_HEAD_SIZE / MAX_FL_ELEM); j++) {
			
			mfdata_v_t temp_out;
			for (size_t k = 0; k < MAX_FL_ELEM; k++) {
				#pragma HLS PIPELINE II=1
				temp_out[k + 0] = vec[k + 0];
				// temp_out[k + 1] = vec[k + 1];
			}
			out.write(temp_out);
		}
		
	}
}
#endif