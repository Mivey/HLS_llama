#include "mha_forward.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <hls_stream.h>
#include <hls_vector.h>
#include <limits>
#include <utils/x_hls_defines.h>

const int BURST_LEN = 8;
template<typename T, size_t N, int M, int P> // M = burst, P = # of gemv outputs
void pipo_intake(hls::vector<T, N> pipo_buf[M][P], hls::stream<T> (&gemv_out)[P]){

	typedef hls::vector<T, N> gdata_v_t;

	for (int i = 0; i < M; i++) {
		intake_data:
		for (int k = 0; k < P; k++) {
			#pragma HLS PIPELINE
			gdata_v_t temp;
			for (int j = 0; j < N; j++) {
				temp[j] = gemv_out[k].read();
			}
			pipo_buf[i][k] = temp;
		}
	}
}

template<typename T, size_t N, int M=BURST_LEN, int P>
void pipo_out(hls::vector<T, N> *out, hls::vector<T, N> pipo_buf[M][P], int chunk_idx, const int M_DIM){

  const int chunk_offset = chunk_idx * M; 
  const int stride = M_DIM / (P * N); 
  
  for (int i = 0; i < P; i++) {
		pipo_out_stride:
    for (int j = 0; j < M; j++) {
      #pragma HLS PIPELINE II=1
      out[(i * stride) + chunk_offset + j] = pipo_buf[j][i];
    }
  }
}
template<typename T, size_t N, int M=BURST_LEN, int P>
void alt_pipo_out(hls::vector<T, N> *out, hls::vector<T, N> pipo_buf[M][P], int chunk_idx, const int M_DIM){

  const int chunk_offset = chunk_idx * M; 
  const int stride = M_DIM / (P * N); 
  
  for (int i = 0; i < P; i++) {
		pipo_out_stride:
    for (int j = 0; j < M; j++) {
      #pragma HLS PIPELINE II=1
      out[(i * stride) + chunk_offset + j] = pipo_buf[j][i];
    }
  }
}

template<typename T, size_t N, int P>
void gemv_combo(hls::vector<T, N> *out, hls::stream<T> (&gemv_out)[P], const int M_DIM){
	
	typedef hls::vector<T, N> gdata_v_t;
	const int total_chunks = M_DIM / (BURST_LEN * P * N);
	
	gc_df_reg:
	for (int df = 0; df < total_chunks; df++) {
		#pragma HLS DATAFLOW
		gdata_v_t my_pipo_arr[BURST_LEN][P]; 
		#pragma HLS ARRAY_PARTITION variable=my_pipo_arr dim=2 type=complete
		
		pipo_intake<T, N, BURST_LEN, P>(my_pipo_arr, gemv_out);
		pipo_out<T, N, BURST_LEN, P>(out, my_pipo_arr, df, M_DIM);
	}
}

template<typename T, size_t N, int P>
void gemv_split(hls::vector<T, N> *out, hls::stream<T> (&gemv_out)[P], const int M_DIM){
	
	const int offset = MODEL_TOKENS / (N * P);
	typedef hls::vector<T, N> gdata_v_t;
	const int c_idx = M_DIM / (P * N);
	for (int i = 0; i < c_idx; i++) {
		for (int j = 0; j < P; j++) {
			#pragma HLS UNROLL
			gdata_v_t data;
			for (int k = 0; k < N; k++) {
				data[k] = gemv_out[j].read();
			}
			out[i + j * offset] = data;
		}
	}
}

/*====================================================================================================================================*/

constexpr int REG_SIZE = 64;

struct ProbIndex{
  short index;
  my_float_t prob;
};

template<typename T, size_t N>
void systolic_sort(hls::stream<hls::vector<T, N>> &s_logit, hls::stream<int> &s_val, int* pick, const float temperature, const float coin){
    
  // ProbIndex init = {0, std::numeric_limits<float>::lowest()};
  ProbIndex st[REG_SIZE]; // systolic temp array
  ProbIndex reg[REG_SIZE]; // stored 64 values
	
	#pragma HLS ARRAY_PARTITION variable=st complete dim=1
  #pragma HLS ARRAY_PARTITION variable=reg complete dim=1
	
	for (int ii = 0; ii < REG_SIZE; ii++) {
		#pragma HLS UNROLL
		st[ii] = {-1, std::numeric_limits<float>::lowest()};
		reg[ii] = {-1, std::numeric_limits<float>::lowest()};
	}
	
  systolic_sort:
  for (int i = 0; i < MODEL_TOKENS / SM_FL_ELEM; i++) {
    fdata_v_t temp = s_logit.read();
		int tmp_dx = s_val.read();
		ss_vector:
    for (int j = 0; j < SM_FL_ELEM; j++) {
      #pragma HLS PIPELINE
      //shift register 
      for (int k = 0; k < (REG_SIZE - 1); k++) { st[k] = st[k + 1]; }
      st[REG_SIZE - 1] = {(short)(tmp_dx + j), temp[j]};
      
      for (int k = 0; k < REG_SIZE; k++) {
        if (st[k].prob > reg[k].prob) {
          ProbIndex swap = reg[k];
          reg[k] = st[k];
          st[k] = swap;
        }
      }
    }
  }

  flush:
  for (int i = 0; i < REG_SIZE; i++) {
    #pragma HLS PIPELINE
    for (int k = 0; k < (REG_SIZE - 1); k++) { st[k] = st[k + 1]; }
    st[REG_SIZE - 1] = {-1, std::numeric_limits<float>::lowest()};
    
    for (int k = 0; k < REG_SIZE; k++) {
      // if (st[k].prob > reg[k].prob) {
      //   ProbIndex swap = reg[k];
      //   reg[k] = st[k];
      //   st[k] = swap;
      // }
			bool do_swap = (st[k].prob > reg[k].prob);
			ProbIndex next_reg = do_swap ? st[k] : reg[k];
			ProbIndex next_st = do_swap ? reg[k] : st[k];

			reg[k] = next_reg;
			st[k] = next_st;
    }


  }
  
  // finished sort. reg now should have largest REG_SIZE (64) values, with max_val @ reg[REG_SIZE-1]

  const my_float_t INV_TEMP = 1/temperature;
  my_float_t max_val = reg[(REG_SIZE - 1)].prob;
  my_float_t final_soft_sum = 0.0f;

  softmax_exp_loop:
	for (int i = 0; i < REG_SIZE; i++) {
    #pragma HLS PIPELINE
    my_float_t curr_val = reg[i].prob;
		my_float_t calc = hls::expf((curr_val - max_val) * INV_TEMP);
		final_soft_sum += calc;
		reg[i].prob = calc;
	}
	my_float_t inv_soft_sum = 1.0f/final_soft_sum;

	softmax_normalize_loop:
	for (int i = 0; i < REG_SIZE; i++) {
    #pragma HLS PIPELINE
    reg[i].prob *= inv_soft_sum;
	}  
  
  my_float_t coin_sum = 0.0f;
	bool found = false;

	*pick = reg[REG_SIZE - 1].index;
  
  coin_flip:
  for (int i = (REG_SIZE - 1); i >= 0; i--) {
		#pragma HLS PIPELINE
		
    coin_sum += reg[i].prob;
		
    if (coin_sum > coin && !found) {
			*pick = reg[i].index;
      found = true;
    }
  }
  return;
}

template<typename T, size_t N, int P>
void gemv_split(hls::vector<T, N> *out, hls::stream<hls::vector<T, N>> &sys_sort, hls::stream<int> &sys_val, hls::stream<T> (&gemv_out)[P], const int M_DIM, const int BOOP){
	
	const int offset = MODEL_TOKENS / (N * P);
	typedef hls::vector<T, N> gdata_v_t;
	const int c_idx = M_DIM / (P * N);
	gdata_v_t min;
	std::fill(min.begin(), min.end(), std::numeric_limits<float>::lowest());
	for (int i = 0; i < c_idx; i++) {
		for (int j = 0; j < P; j++) {
			#pragma HLS UNROLL
			gdata_v_t data;
			for (int k = 0; k < N; k++) {
				#pragma HLS PIPELINE II=1
				data[k] = gemv_out[j].read();
			}
			int idx = i + j*offset;
			out[idx] = data;
			if (BOOP == 0) {
				sys_sort.write(data);
				sys_val.write(idx);
			}
			else {
			sys_sort.write(min);
			sys_val.write(0);
			}
		}
	}
}
