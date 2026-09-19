
// #include "matmult.h"
// #include <memory>
#include "mha_forward.h"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <hls_vector.h>



template<typename T, size_t N, size_t M>
void mm_sf_val(hls::stream<float_t> &out, hls::stream<hls::vector<T, N>> &tok_sf, s_wide_t &w_sf, const int N_DIM_SF, const int vCount){
	
	float_t arr[TOK_SF_MAX];
	const int sfCnt = N_DIM_SF;
	// const int vCount = (N_DIM_SF * M_DIM);
	
	mm_tok_sf:
	for (int i = 0; i < sfCnt/N; i++) {
		#pragma HLS PIPELINE
		hls::vector<T, N> tmp = tok_sf.read();
		for (int j = 0; j < N; j++) {
			#pragma HLS UNROLL
			arr[i * N + j] = tmp[j];
		}
	}
	
	// wide_t tmp_w;
	hls::vector<T, M> tmp_shift;

	mm_sf_out:
	for (int ii = 0; ii < vCount; ii++) {
		#pragma HLS PIPELINE II=1
		
		int idx = ii % M;
		int kdx = ii % sfCnt;
		
		if (idx == 0) {
			// tmp_w = ;
			tmp_shift = to_mfdvt(w_sf.read());
		}

		float_t tmpo = tmp_shift[0] * arr[kdx];
		out.write(tmpo);

		for (int jj = 0 ; jj < (M - 1); jj++) {
			#pragma HLS UNROLL
			tmp_shift[jj] = tmp_shift[jj + 1];
		}
	}
}

void mm_part_out (hls::stream<float_t> &out, s_wide_t &w, s_idata_v_t &tok_w, hls::stream<float_t> &sf_in, const int N_DIM_SF, const int vCount){
	
	idata_v_t arr[TOK_QUANT_MAX];
	#pragma HLS BIND_STORAGE variable=arr type=ram_2p impl=bram
	// #pragma hls ARRAY_PARTITION variable=arr dim=1 type=complete
	// const int vCount = N_DIM_SF * M_DIM;
	constexpr int ratio = MODEL_SCALING_FACTOR / MAX_QUANT_ELEM;
	const int wCnt = N_DIM_SF * ratio;

	mm_w:
	for (int i = 0; i < wCnt; i++) {
		#pragma HLS PIPELINE II=1
		arr[i] = tok_w.read();
	}

	mm_w_out:
	for (int ii = 0; ii < vCount; ii++) {
		#pragma HLS PIPELINE II=ratio
		int32_t prod[ratio]{};
		int32_t part_int = 0;
		#pragma HLS ARRAY_PARTITION variable=prod dim=1 type=complete
		
		for (int jj = 0; jj < ratio; jj++) {
			const int base_idx = (ii % N_DIM_SF) * ratio + jj; 
			#pragma HLS UNROLL
			idata_v_t curr_tok = arr[base_idx];
			idata_v_t curr_w = to_idvt(w.read());
			for (int kk = 0; kk < MAX_QUANT_ELEM; kk++) {
				#pragma HLS UNROLL
				prod[jj] += curr_tok[kk] * curr_w[kk];
			}
			part_int += prod[jj];
		}
		
		float_t part_fl = (float_t) part_int * sf_in.read();
		out.write(part_fl);
	}
}

void mm_reduce_add (hls::stream<float_t> &out, hls::stream<float_t> &in, const int SF_CNT, const int M_DIM){
	
	constexpr int ratio = TOK_SF_MAX/SM_FL_ELEM;
	float_t sf_arr[ratio]{};
	#pragma HLS ARRAY_PARTITION variable=sf_arr dim=1 type=complete
	
	for (int i = 0 ; i < M_DIM; i++) {
		for ( int j = 0; j < SF_CNT; j++) {
			#pragma HLS PIPELINE II = SM_FL_ELEM
			fdata_v_t tmp;
			for (int k = 0; k < SM_FL_ELEM; k++) {
				#pragma HLS UNROLL
				tmp[k] = in.read();
			}
			sf_arr[j] = tmp.reduce_add();
		}
		float_t final_out = 0;
		for (int jj = 0; jj < ratio; jj++) {
			final_out += sf_arr[jj];
		}
	}
}


void mm_w_val(hls::stream<float_t> &out, s_wide_t &w, s_idata_v_t &tok_w, hls::stream<float_t> &sf_in, const int N_DIM, const int M_DIM){
	
	idata_v_t arr[TOK_QUANT_MAX];
	const int vCount = N_DIM * M_DIM / MAX_QUANT_ELEM;
	const int wCnt = N_DIM / MAX_QUANT_ELEM;

	mm_w_sf:
	for (int i = 0; i < wCnt; i++) {
		#pragma HLS PIPELINE II=1
		arr[i] = tok_w.read();
	}

	mm_w_out:
	for (int ii = 0; ii < vCount; ii++) {
		#pragma hls PIPELINE II=1
		int32_t prod = 0;
		int idx = ii % wCnt;
		idata_v_t curr_tok = arr[idx];
			idata_v_t curr_w = to_idvt(w.read());
		for (int jj = 0; jj < MAX_QUANT_ELEM; jj++) {
			#pragma HLS UNROLL
			prod += (int32_t) (curr_tok[jj] * curr_w[jj]);
		}
		float_t tmpf = (float_t) prod * sf_in.read();
		out.write(tmpf);
	}
}


// template<typename T, int N>
void mm_w_sum(hls::stream<float_t> &out, s_wide_t &w, s_idata_v_t &tok_w, hls::stream<float_t> &sf_in, const int N_DIM, const int M_DIM){
	
	idata_v_t arr[TOK_QUANT_MAX];
	// const int vCount = N_DIM * M_DIM / MAX_QUANT_ELEM;
	const int wCnt = N_DIM / MAX_QUANT_ELEM;
	const int sfCount = N_DIM / MODEL_SCALING_FACTOR;
	mm_w_sf:
	for (int i = 0; i < wCnt; i++) {
		#pragma HLS PIPELINE II=1
		arr[i] = tok_w.read();
	}

	float_t psum_out[4]{};
	float_t sum_out = 0;
	#pragma HLS ARRAY_PARTITION variable=psum_out dim=1 type=complete
	
	gemv_out:
	for (int i = 0; i < M_DIM; i++) {
		
		partial_sum:
		for (int jj = 0; jj < sfCount; jj++) {
			#pragma hls PIPELINE II=1
			// int idx = jj % wCnt;
			idata_v_t curr_tok = arr[jj];
			idata_v_t curr_w = to_idvt(w.read());
			
			int32_t prod = 0;
			for (int kk = 0; kk < MAX_QUANT_ELEM; kk++) {
				#pragma HLS UNROLL
				prod += (int32_t) (curr_tok[kk] * curr_w[kk]);
			}

			psum_out[jj % 4] += (float_t) prod * sf_in.read();

			// float_t t = psum_out[7];
			
			// for (int k = (TOK_SF_MAX - 1); k > 0; k--) {
			// 	#pragma HLS UNROLL
			// 	psum_out[k] = psum_out[k - 1];
			// }
			// psum_out[0] = (float_t)prod * sf_in.read() + t;
			
		}
		
		for (int k = 0; k < 4; k++) {
			#pragma HLS UNROLL
			sum_out += psum_out[k];
			psum_out[k] = 0;
		}

		out.write(sum_out);
		
		// for (int k = 0; k < TOK_SF_MAX; k++) {
		// 	#pragma HLS UNROLL
		// 	psum_out[k] = 0;
		// }
		
		sum_out = 0;
	}
}

// void mm_sum_out(hls::stream<float_t> &out, hls::stream<float_t> &in, const int N_DIM, const int M_DIM){
	
// 	const int sCnt = N_DIM / MODEL_SCALING_FACTOR;
	
// 	int idx = 0;
		
// 	float_t psum[4]{};
// 	#pragma HLS ARRAY_PARTITION variable=psum dim=1 type=complete
	
// 	sum_out:
// 	for (int i = 0; i < M_DIM * sCnt; i++) {
		
// 		// psum_out:
// 		// for (int j = 0; j < sCnt; j++) {
// 			#pragma HLS PIPELINE II=1
// 			psum[idx & 3] += in.read(); // instead of i % 4
// 		// }
		
// 		if (idx++ == (sCnt - 1)) {
// 			float_t sum = psum[0] + psum[1] + psum[2] + psum[3];
// 			out.write(sum);
// 			idx = 0;
// 			for (int j = 0; j < 4; j++) {
// 				#pragma HLS UNROLL
// 				psum[j] = 0;
// 			}
// 		}
// 	}
// }

void mm_sum_out(hls::stream<float_t> &out, hls::stream<float_t> &in, const int N_DIM, const int M_DIM){
	
	const int sCnt = N_DIM / MODEL_SCALING_FACTOR;
	
	int idx = 0;
	
	sum_out:
	for (int i = 0; i < M_DIM; i++) {
		
		float_t psum[4]{};
		#pragma HLS ARRAY_PARTITION variable=psum dim=1 type=complete
		
		psum_out:
		for (int j = 0; j < sCnt; j++) {
			#pragma HLS PIPELINE II=1
			psum[j & 3] += in.read(); // instead of i % 4
		}
		
		float_t sum = psum[0] + psum[1] + psum[2] + psum[3];
		out.write(sum);
	}
}

/* ***************************************************************************************** */

void alt_mat_mult_main(hls::stream<my_float_t> &out, s_idata_v_t &w, s_fdata_v_t &w_sf, \
                      s_idata_v_t &tok, s_fdata_v_t &tok_sf, const int N_DIM, const int M_DIM){

  const int sfCount = N_DIM / (SM_FL_ELEM * MODEL_SCALING_FACTOR);
  const int TOK_ARR_SIZE = N_DIM / MAX_QUANT_ELEM;
  const int SUM_FACTOR = MODEL_SCALING_FACTOR / MAX_QUANT_ELEM;
  // const int SF_2_Q_RATIO = MODEL_SCALING_FACTOR / MAX_QUANT_ELEM;

  //for now, assume idvt is 512 and only 512. 256 and 128 would require amm_calc to have 
  // another factor that handles 
  
  fdata_v_t arr_sf[TOK_SF_MAX];
  idata_v_t arr[TOK_QUANT_MAX];
  #pragma HLS BIND_STORAGE variable=arr_sf type=ram_2p impl=bram
  // #pragma HLS BIND_STORAGE variable=arr impl=srl

  amm_tok_sf:
  for (size_t i = 0; i < sfCount; i++){ // vCount here is 1/4 vCount in send_wtok!!
    #pragma HLS PIPELINE II=1
    #pragma HLS LOOP_TRIPCOUNT max = TOK_SF_MAX min=MODEL_ELEMENTS/(MODEL_SCALING_FACTOR * SM_FL_ELEM )  
    arr_sf[i] = tok_sf.read();
    for (size_t j = 0; j < ( SM_FL_ELEM); j++) {
      arr[i * (SM_FL_ELEM) + j] = tok.read();
    }
  }
  
  amm_calc:
  for (size_t i = 0; i < M_DIM; i++) {
    #pragma HLS LOOP_TRIPCOUNT max=MODEL_TOKENS min=MODEL_ELEMENTS
    //output M_DIM float elements
    float_t sum_out = 0;
    for (size_t j = 0 ; j < sfCount; j++) {
		// #pragma HLS LOOP_FLATTEN
    #pragma HLS LOOP_TRIPCOUNT max = TOK_SF_MAX min=MODEL_ELEMENTS/(MODEL_SCALING_FACTOR * SM_FL_ELEM )  
      //read the next set of scaling factors
      fdata_v_t vec_tok_sf = arr_sf[j];
      fdata_v_t vec_w_sf = w_sf.read();
      amm_k_calc:
      for (size_t k = 0; k < SM_FL_ELEM; k++) {
        //do our calculations
#pragma HLS PIPELINE II=1 rewind //style=frp -- 340
        
        float_t cur_tok_sf = vec_tok_sf[k] * vec_w_sf[k];
        // my_float_t cur_w_sf = vec_w_sf[k];
        
        //read the next set of weights
        idata_v_t curr_tok;
        idata_v_t curr_w;
        
        int32_t prod = 0;
        // int32_t comb_prod = 0;
        
        curr_w = w.read();
        curr_tok = arr[j * SM_FL_ELEM + k];
        // prod = 0;
        
        for (size_t m = 0; m < MAX_QUANT_ELEM; m++) {
          prod += (int32_t) curr_w[m] * curr_tok[m];
        }
        sum_out += (float_t)prod * cur_tok_sf;// * cur_w_sf;
      }
    }
    out.write(sum_out);
  }
}

void s_GeMV_kernel(hls::stream<my_float_t> &out, s_fdata_v_t &tok_sf, s_idata_v_t &tok_q, //
    s_wide_t &s_wsf, s_wide_t &s_w, const int N_DIM, const int M_DIM){
	
  const int N_DIM_SF = N_DIM / MODEL_SCALING_FACTOR;
	const int vCount = M_DIM * N_DIM_SF;
	const int SF_CNT = N_DIM_SF / SM_FL_ELEM;
  
  #pragma HLS DATAFLOW
  hls::stream<my_float_t> part_out;
	hls::stream<float_t> sf_out;
	#pragma HLS STREAM variable=part_out depth = 64
	#pragma HLS STREAM variable=sf_out depth = 64
  
	mm_sf_val<float_t, SM_FL_ELEM, MAX_FL_ELEM>(sf_out, tok_sf, s_wsf, N_DIM_SF, vCount);
	mm_part_out(part_out, s_w, tok_q, sf_out, N_DIM_SF, vCount);
	mm_reduce_add(out, part_out, SF_CNT, M_DIM);
  return;
}

