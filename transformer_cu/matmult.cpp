
// #include "matmult.h"
// #include <memory>
#include "mha_forward.h"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <hls_vector.h>



template<typename T, size_t N, size_t M>
void mm_sf_val(hls::stream<float_t> &out, hls::stream<hls::vector<T, N>> &tok_sf, hls::stream<hls::vector<T, M>> &w_sf, const int N_DIM, const int M_DIM){
	
	float_t arr[TOK_SF_MAX];
	const int sfCnt = N_DIM / MODEL_SCALING_FACTOR;
	const int vCount = (N_DIM * M_DIM)/(MODEL_SCALING_FACTOR);
	
	mm_tok_sf:
	for (int i = 0; i < sfCnt/N; i++) {
		#pragma HLS PIPELINE
		hls::vector<T, N> tmp = tok_sf.read();
		for (int j = 0; j < N; j++) {
			#pragma HLS UNROLL
			arr[i * N + j] = tmp[j];
		}
	}

	hls::vector<T, M> tmp_shift;

	mm_sf_out:
	for (int ii = 0; ii < vCount; ii++) {
		#pragma HLS PIPELINE II=1
		
		int idx = ii % M;
		int kdx = ii % sfCnt;
		
		if (idx == 0) {
			tmp_shift = w_sf.read();
		}

		float_t tmpo = tmp_shift[0] * arr[kdx];
		out.write(tmpo);

		for (int jj = 0 ; jj < (M - 1); jj++) {
			#pragma HLS UNROLL
			tmp_shift[jj] = tmp_shift[jj + 1];
		}
	}
}

// template<typename T, int N>
void mm_w_val(hls::stream<float_t> &out, s_idata_v_t &w, s_idata_v_t &tok_w, hls::stream<float_t> &sf_in, const int N_DIM, const int M_DIM){
	
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
		idata_v_t curr_w = w.read();
		for (int jj = 0; jj < MAX_QUANT_ELEM; jj++) {
			#pragma HLS UNROLL
			prod += (int32_t) (curr_tok[jj] * curr_w[jj]);
		}
		float_t tmpf = (float_t) prod * sf_in.read();
		out.write(tmpf);
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
    s_mfdata_v_t &s_wsf, s_idata_v_t &s_w, const int N_DIM, const int M_DIM){

  constexpr int mm_thr = 2;
  // const int num = N_DIM * M_DIM ;
  // const int num_sf = N_DIM * M_DIM / (MODEL_SCALING_FACTOR );
  const int w_count = N_DIM * M_DIM / MAX_QUANT_ELEM;
  const int mf_sf_count = N_DIM * M_DIM / (MODEL_SCALING_FACTOR * MAX_FL_ELEM);
  const int sm_sf_count = N_DIM * M_DIM / (MODEL_SCALING_FACTOR * SM_FL_ELEM);
  const int sfCount = N_DIM / (MODEL_SCALING_FACTOR * SM_FL_ELEM);
  const int qCount = N_DIM / MAX_QUANT_ELEM;
  
  #pragma HLS DATAFLOW
  
  s_fdata_v_t s_vd_wsf("s_vd_wsf");
  #pragma HLS BIND_STORAGE variable=s_vd_wsf type=fifo impl=uram
  #pragma HLS STREAM variable=s_vd_wsf type=fifo depth=4096
  
  #pragma HLS STREAM variable=tok_q type=fifo depth=32
  idata_v_t w_arr[TOK_QUANT_MAX];

  hls::stream<my_float_t> out_thread[mm_thr];
  s_fdata_v_t d_tok_sf[mm_thr];
  s_idata_v_t d_tok[mm_thr];
  s_fdata_v_t d_wsf[mm_thr];
  s_idata_v_t d_w[mm_thr];
  #pragma HLS STREAM variable=d_wsf depth = 96// MODEL_HIDDEN_DIM/MAX_FL_ELEM
  #pragma HLS STREAM variable=d_w depth = 384// MODEL_HIDDEN_DIM/MAX_FL_ELEM
  #pragma HLS STREAM variable=d_tok_sf depth=4
  #pragma HLS STREAM variable=d_tok depth=8
  #pragma HLS BIND_STORAGE variable=d_w type=fifo impl=bram
  #pragma HLS BIND_STORAGE variable=d_wsf type=fifo impl=bram
  

  inf_split_tee(d_tok_sf, tok_sf, (N_DIM / (MODEL_SCALING_FACTOR * SM_FL_ELEM)));
  inf_split_tee(d_tok, tok_q, (N_DIM / MAX_QUANT_ELEM));
  
  vec_down_converter(s_vd_wsf, s_wsf, sm_sf_count);

  inf_round_robin(d_wsf, s_vd_wsf, (N_DIM / (MODEL_SCALING_FACTOR * SM_FL_ELEM)), M_DIM);
  inf_round_robin(d_w, s_w, (N_DIM / MAX_QUANT_ELEM), M_DIM);

  for (int i = 0; i < mm_thr; i++) {
    #pragma HLS UNROLL
    alt_mat_mult_main(out_thread[i], d_w[i], d_wsf[i], d_tok[i], d_tok_sf[i], N_DIM, M_DIM/mm_thr);
  }
  
  rr_merge<my_float_t, mm_thr>(out, out_thread, M_DIM);
  return;
}


// void s_GeMV_kernel(hls::stream<my_float_t> &out, s_fdata_v_t &tok_sf, s_idata_v_t &tok_q, //
//     s_mfdata_v_t &s_wsf, s_idata_v_t &s_w, const int N_DIM, const int M_DIM){

//   constexpr int mm_thr = 2;
//   // const int num = N_DIM * M_DIM ;
//   // const int num_sf = N_DIM * M_DIM / (MODEL_SCALING_FACTOR );
//   const int w_count = N_DIM * M_DIM / MAX_QUANT_ELEM;
//   const int mf_sf_count = N_DIM * M_DIM / (MODEL_SCALING_FACTOR * MAX_FL_ELEM);
//   const int sm_sf_count = N_DIM * M_DIM / (MODEL_SCALING_FACTOR * SM_FL_ELEM);
//   const int sfCount = N_DIM / (MODEL_SCALING_FACTOR * SM_FL_ELEM);
//   const int qCount = N_DIM / MAX_QUANT_ELEM;
  
//   #pragma HLS DATAFLOW
  
//   s_fdata_v_t s_vd_wsf("s_vd_wsf");
//   #pragma HLS BIND_STORAGE variable=s_vd_wsf type=fifo impl=uram
//   #pragma HLS STREAM variable=s_vd_wsf type=fifo depth=4096
  
//   #pragma HLS STREAM variable=tok_q type=fifo depth=32
//   idata_v_t w_arr[TOK_QUANT_MAX];

//   hls::stream<my_float_t> out_thread[mm_thr];
//   s_fdata_v_t d_tok_sf[mm_thr];
//   s_idata_v_t d_tok[mm_thr];
//   s_fdata_v_t d_wsf[mm_thr];
//   s_idata_v_t d_w[mm_thr];
// 	hls::stream<float_t> sf_out, w_out;
// 	#pragma HLS STREAM variable=sf_out depth = 64
// 	#pragma HLS STREAM variable=w_out depth = 64
//   #pragma HLS STREAM variable=d_wsf depth = 96// MODEL_HIDDEN_DIM/MAX_FL_ELEM
//   #pragma HLS STREAM variable=d_w depth = 384// MODEL_HIDDEN_DIM/MAX_FL_ELEM
//   #pragma HLS STREAM variable=d_tok_sf depth=4
//   #pragma HLS STREAM variable=d_tok depth=8
//   #pragma HLS BIND_STORAGE variable=d_w type=fifo impl=bram
//   #pragma HLS BIND_STORAGE variable=d_wsf type=fifo impl=bram
  
// 	mm_sf_val(sf_out, tok_sf, s_wsf, N_DIM, M_DIM);
// 	mm_w_val(w_out, s_w, tok_q, sf_out, N_DIM, M_DIM);
// 	mm_sum_out(out, w_out, N_DIM, M_DIM);
//   return;
// }