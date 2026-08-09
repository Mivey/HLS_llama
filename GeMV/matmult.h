
#ifndef MARK_MM
#define MARK_MM
#include "../forward.h"
#include <cmath>
void GeMV_kernel(fdata_v_t *out, fdata_v_t *fl_tok, fdata_v_t *w_sf, idata_v_t *w, const int N_DIM, const int M_DIM, const int CURR_LAYER, const int W_Off);
void GeMV_PE_kernel(hls::stream<my_float_t> &out, hls::stream<my_float_t>  &tok_sf, s_idata_v_t &tok_q, mfdata_v_t *w_sf, idata_v_t *w, const int N_DIM, const int M_DIM, const int CURR_LAYER, const int W_Off, const int sf_reg, const int w_reg);
void quantizer_kernel(hls::stream<my_float_t>  &tok_sf_out, s_idata_v_t &tok_out, s_fdata_v_t &tokens, const int N_DIM);
constexpr size_t TOK_QUANT_MAX =  (MODEL_HIDDEN_DIM / MAX_QUANT_ELEM);
constexpr size_t TOK_SF_MAX = (MODEL_HIDDEN_DIM / (MODEL_SCALING_FACTOR));


template<typename T, size_t M>
void GeMV_PE(hls::stream<float_t> &qout, hls::stream<hls::vector<T, M>> &w, hls::vector<T, M> *tok, const int N_DIM, const int M_DIM){
	const int num_loop = N_DIM / M;
	typedef hls::vector<T, M> tmp_t;
	
// 	tmp_t arr[TOK_QUANT_MAX];
// #pragma HLS BIND_STORAGE variable=arr type=ram_2p impl=bram
  // #pragma HLS BIND_STORAGE variable=arr type=ram_2p impl=bram
	// #pragma hls ARRAY_PARTITION variable=arr dim=1 type=complete
	
	// amm_tok:
	// for (size_t i = 0; i < num_loop; i++){ // vCount here is 1/4 vCount in send_wtok!!
	// 	#pragma HLS PIPELINE II=1
  // 	#pragma HLS LOOP_TRIPCOUNT max = TOK_SF_MAX min=MODEL_ELEMENTS/(MODEL_SCALING_FACTOR * SM_FL_ELEM )  
	// 	arr[i] = tok.read();

	// }
	for (size_t k = 0; k < M_DIM; k++) {
		#pragma HLS LOOP_FLATTEN OFF
		
		PE_quant:
		for (size_t j = 0; j < num_loop; j++) {
			#pragma HLS PIPELINE II=1
			tmp_t curr_tok = tok[j];
			tmp_t curr_w = w.read();
			
			int32_t prod{}; 
			for (size_t i = 0; i < M; i++) {
				#pragma HLS UNROLL
				prod += (int32_t) curr_tok[i] * curr_w[i];
			}

			float fprod = (float_t) prod;
			qout.write(fprod);
		}
	}
}

template<size_t N, typename T>
void GeMV_PE_sf(hls::stream<float_t> &sfout, hls::stream<hls::vector<T, N>> &w_sf, float_t *tok_sf, const int N_DIM, const int M_DIM){
	
	const int num_loop = N_DIM / (N * MODEL_SCALING_FACTOR);
	typedef hls::vector<T, N>	tmp_t;

// 	float_t arr_sf[TOK_SF_MAX];
// #pragma HLS BIND_STORAGE variable=arr_sf type=ram_2p impl=bram

// 	amm_tok_sf:
// 	for (size_t i = 0; i < (num_loop * N); i++){ // vCount here is 1/4 vCount in send_wtok!!
// 		#pragma HLS PIPELINE II=1
//   	#pragma HLS LOOP_TRIPCOUNT max = TOK_SF_MAX min=MODEL_ELEMENTS/(MODEL_SCALING_FACTOR )  
// 		arr_sf[i] = tok_sf.read();
// 	}
	// tmp_t curr_w = w_sf.read();
	
	for (size_t k = 0; k < M_DIM; k++) {
		
		for (size_t j = 0; j < num_loop; j++) {
		#pragma HLS LOOP_FLATTEN OFF
			tmp_t curr_tok = w_sf.read();
			#pragma HLS ARRAY_PARTITION variable=curr_tok dim=1 type=complete
			#pragma HLS PIPELINE II=1
			
			PE_sf:
			for (size_t i = 0; i < N; i++) {
				#pragma HLS UNROLL
				float_t tmpa = tok_sf[j * N + i] * curr_tok[i];
				sfout.write(tmpa);
			}
		}
	}
}


template<size_t N, typename T>
void GeMV_PE_sf(hls::stream<float_t> &sfout, hls::stream<hls::vector<T, N>> &w_sf, hls::stream<hls::vector<T, N>> &tok_sf, const int N_DIM, const int M_DIM){
	
	const int num_loop = N_DIM * M_DIM / (N * MODEL_SCALING_FACTOR);
	typedef hls::vector<T, N>	tmp_t;

	fdata_v_t arr_sf[TOK_SF_MAX];
  #pragma HLS BIND_STORAGE variable=arr_sf type=ram_2p impl=bram
	// #pragma hls ARRAY_PARTITION variable=arr_sf dim=1 type=complete
  // #pragma HLS BIND_STORAGE variable=arr impl=srl

	amm_tok_sf:
	for (size_t i = 0; i < num_loop; i++){ // vCount here is 1/4 vCount in send_wtok!!
		#pragma HLS PIPELINE II=1
  	#pragma HLS LOOP_TRIPCOUNT max = TOK_SF_MAX min=MODEL_ELEMENTS/(MODEL_SCALING_FACTOR * SM_FL_ELEM )  
		arr_sf[i] = tok_sf.read();
	}
	// tmp_t curr_w = w_sf.read();
	
	for (size_t k = 0; k < M_DIM; k++) {
		#pragma HLS LOOP_FLATTEN OFF
		
		for (size_t j = 0; j < num_loop; j++) {
			
			tmp_t curr_tok = arr_sf[j] * w_sf.read();
			PE_sf:
			for (size_t i = 0; i < N; i++) {
				#pragma HLS PIPELINE II=1
				sfout.write(curr_tok[i]);
			}
		}
	}
}


// inline void GeMV_PE_sum(hls::stream<float_t> &out, hls::stream<float_t> &qin, hls::stream<float_t> &sfin, const int N_DIM, const int M_DIM){
	
// 	const int num_loop = N_DIM / MODEL_SCALING_FACTOR;
// 	for (size_t j = 0; j < M_DIM; j++) {
// 		#pragma HLS LOOP_FLATTEN OFF
		
// 		float_t part_sum[4]{};
// 		#pragma HLS ARRAY_PARTITION variable=part_sum dim=1 type=complete
// 		PE_sum:
// 		for (size_t i = 0; i < num_loop; i+=4) {
// 			#pragma HLS PIPELINE II=4
// 			// #pragma HLS UNROLL factor=4
// 			// float_t tmpa = sfin.read() * qin.read();
// 			part_sum[0] += sfin.read() * qin.read();
// 			part_sum[1] += sfin.read() * qin.read();
// 			part_sum[2] += sfin.read() * qin.read();
// 			part_sum[3] += sfin.read() * qin.read();
// 		}
// 		float_t tsum{};
// 		for (int i = 0; i < 4; i++) {
// 			#pragma HLS UNROLL
// 			tsum += part_sum[i];
// 		}
// 		out.write(tsum);
// 	}
// }



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
		#pragma HLS LOOP_TRIPCOUNT min=(MODEL_ELEMENTS * MODEL_ELEMENTS / (2 * MODEL_SCALING_FACTOR * SM_FL_ELEM) / ratio) max=(MODEL_TOKENS * MODEL_ELEMENTS / (2 * MODEL_SCALING_FACTOR * SM_FL_ELEM))
		// #pragma hls PIPELINE II=1
		M_t mtmp = in.read();
		for (int j = 0; j < ratio; j++) {
			#pragma HLS PIPELINE II=1
			#pragma HLS LOOP_TRIPCOUNT max=ratio
			N_t tmp;
			for (int k = 0; k < N; k++) {
				tmp[k] = mtmp[j * N + k];
			}
			out.write(tmp);
		}
	}
}

template<typename T>
void mm2s_input_data(hls::stream<T> &out, T *in, const int COUNT, const int CURR_LAYER, const int offset){
	
	// const int tot_off = CURR_LAYER * COUNT + offset;
	uint32_t layer_offset = CURR_LAYER * COUNT;
  #pragma HLS BIND_OP variable=layer_offset op=mul impl=dsp latency=3
  
  const uint32_t tot_off = layer_offset + offset;
	AXI4_to_STREAM:
	for (int i = 0; i < COUNT; i++) {
		#pragma HLS PIPELINE II=1
		#pragma HLS LOOP_TRIPCOUNT min=(MODEL_ELEMENTS * MODEL_ELEMENTS / (2 * SM_FL_ELEM * MODEL_SCALING_FACTOR)) max=(MODEL_TOKENS * MODEL_ELEMENTS / (2 * MAX_QUANT_ELEM))
		out.write(in[i + tot_off]);
	}
}
template<typename T>
void mm2s_input_data(hls::stream<T> &out, T *in, const int COUNT, const int CURR_LAYER, const int offset, const int slice, const int sliceCnt){
	
	// const int tot_off = CURR_LAYER * COUNT + offset;
	uint32_t layer_offset = CURR_LAYER * COUNT;
  #pragma HLS BIND_OP variable=layer_offset op=mul impl=dsp latency=3
	uint32_t slice_offset = slice * sliceCnt;
  #pragma HLS BIND_OP variable=slice_offset op=mul impl=dsp latency=3
  
  const uint32_t tot_off = layer_offset + offset + slice_offset;
	AXI4_to_STREAM:
	for (int i = 0; i < sliceCnt; i++) {
		#pragma HLS PIPELINE II=1
		#pragma HLS LOOP_TRIPCOUNT min=(MODEL_ELEMENTS * MODEL_ELEMENTS / (2 * SM_FL_ELEM * MODEL_SCALING_FACTOR)) max=(MODEL_TOKENS * MODEL_ELEMENTS / (2 * MAX_QUANT_ELEM))
		out.write(in[i + tot_off]);
	}
}
#endif
