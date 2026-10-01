
// #include "matmult.h"
// #include <memory>
#include "mha_forward.h"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <hls_vector.h>

// int32_t rtl_code(wide_t a, wide_t b);

int32_t dot_prod(wide_t a, wide_t b) {
#pragma HLS EXPRESSION_BALANCE off
    ap_uint<528> ap = a;
    ap_uint<528> bp = b;
    ap_int<32> acc = 0;
 
    for (int g = 0; g < 22; g++) {
#pragma HLS UNROLL
        ap_int<8> x0 = ap.range(24 * g + 7,  24 * g);
        ap_int<8> x1 = ap.range(24 * g + 15, 24 * g + 8);
        ap_int<8> x2 = ap.range(24 * g + 23, 24 * g + 16);
        ap_int<8> y0 = bp.range(24 * g + 7,  24 * g);
        ap_int<8> y1 = bp.range(24 * g + 15, 24 * g + 8);
        ap_int<8> y2 = bp.range(24 * g + 23, 24 * g + 16);
 
        acc = x0 * y0 + x1 * y1 + x2 * y2 + acc;
    }
    return acc;
}
 


void mm_rtl_w(hls::stream<float_t> &out, s_wide_t &w, s_idata_v_t &tok_w, hls::stream<float_t> &sf_in, const int N_DIM, const int M_DIM){
	
	wide_t arr[TOK_QUANT_MAX];
	const int vCount = N_DIM * M_DIM ;
	const int wCnt = N_DIM ;

	mm_w_sf:
	for (int i = 0; i < wCnt; i++) {
		#pragma HLS PIPELINE II=1
		arr[i] = to_wide_t(tok_w.read());
	}

	mm_w_out:
	for (int ii = 0; ii < vCount; ii++) {
		#pragma hls PIPELINE II=1 style=frp
		int idx = ii % wCnt;
		wide_t curr_tok = arr[idx];
        wide_t curr_w = w.read();
		int32_t prod = dot_prod(curr_tok, curr_w);
		float_t tmpf = (float_t) prod * sf_in.read();
		out.write(tmpf);
	}
}

void mm_psum(hls::stream<float_t> &out, hls::stream<float_t> &in, const int N_DIM, const int M_DIM){
    
    for (int i = 0; i < (M_DIM * N_DIM ); i++) {
        #pragma HLS PIPELINE 
        fdata_v_t tmp;
        for (int j = 0; j < 4; j++) {
            #pragma HLS UNROLL
            tmp[j] = in.read();
        }
        float_t tmp_o = tmp.reduce_add();
        out.write(tmp_o);
    }
}

void mm_final_sum(hls::stream<float_t> &out, hls::stream<float_t> &in, const int N_DIM, const int M_DIM){
    
    float_t acc{};
    int j = 0;
    for (int i = 0; i < (M_DIM * N_DIM); i++) {
        #pragma HLS PIPELINE II=4
        acc += in.read();
        if (j != (N_DIM - 1)) {
            j++;
        }else {
            j = 0;
            out.write(acc);
            acc = 0;
        }
    }
}


template<typename T, size_t N, size_t M>
void gemv_sf_val(hls::stream<float_t> &out, hls::stream<hls::vector<T, N>> &tok_sf, s_wide_t &w_sf, const int N_DIM, const int M_DIM){
	
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

void s_rtl_GeMV_kernel(hls::stream<my_float_t> &out, s_fdata_v_t &tok_sf, s_idata_v_t &tok_q, //
    s_wide_t &s_wsf, s_wide_t &s_w, const int N_DIM, const int M_DIM){

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

//   hls::stream<my_float_t> out_thread[mm_thr];
//   s_fdata_v_t d_tok_sf[mm_thr];
//   s_idata_v_t d_tok[mm_thr];
//   hls::stream<float_t> d_wsf[mm_thr];
//   s_wide_t d_w[mm_thr];


  
  hls::stream<float_t> s_psum;
  hls::stream<float_t> s_sum;

	hls::stream<float_t> sf_out;
	#pragma HLS STREAM variable=sf_out depth = 64
	#pragma HLS STREAM variable=s_psum depth = 64
	#pragma HLS STREAM variable=s_sum depth = 64
//   #pragma HLS STREAM variable=d_wsf depth = 96// MODEL_HIDDEN_DIM/MAX_FL_ELEM
//   #pragma HLS STREAM variable=d_w depth = 384// MODEL_HIDDEN_DIM/MAX_FL_ELEM
//   #pragma HLS STREAM variable=d_tok_sf depth=4
//   #pragma HLS STREAM variable=d_tok depth=8
//   #pragma HLS BIND_STORAGE variable=d_w type=fifo impl=bram
//   #pragma HLS BIND_STORAGE variable=d_wsf type=fifo impl=bram
  	
	gemv_sf_val<float_t, SM_FL_ELEM, MAX_FL_ELEM>(sf_out, tok_sf, s_wsf, N_DIM, M_DIM);
    mm_rtl_w(s_psum, s_w, tok_q, sf_out, qCount, M_DIM);
    mm_psum(s_sum, s_psum, sfCount, M_DIM);
    mm_final_sum(out, s_sum, sfCount, M_DIM);
  return;
}
