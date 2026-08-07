#include "mha_forward.h"
#include <cstdint>

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
	
	const int offset = INTERNAL_DATA_SIZE / (N * P);
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
constexpr int ARR_PART = 2;

struct ProbIndex{
  short index;
  my_float_t prob;
};

// template<typename T, size_t N>
void systolic_sort(hls::stream<ProbIndex> &ss_val, ProbIndex *reg, const int M_DIM){
  // #pragma HLS INLINE
  // ProbIndex init = {0, std::numeric_limits<float>::lowest()};
  ProbIndex st[REG_SIZE]; // systolic temp array
  // ProbIndex reg[REG_SIZE]; // stored 64 values
	
	#pragma HLS ARRAY_PARTITION variable=st complete dim=1
  //  // up a level
	
	for (int ii = 0; ii < REG_SIZE; ii++) {
		#pragma HLS PIPELINE II=1
		st[ii] = {-1, std::numeric_limits<my_float_t>::lowest()};
		reg[ii] = {-1, std::numeric_limits<my_float_t>::lowest()};
	}
	
  systolic_sort:
  for (int i = 0; i < (M_DIM + REG_SIZE); i++) {
		#pragma HLS LOOP_TRIPCOUNT max=(MODEL_TOKENS + REG_SIZE)
		#pragma HLS pipeline II=3
		ProbIndex tmp_pi = ss_val.read();
      for (int k = 0; k < (REG_SIZE - 1); k++) { 
				#pragma HLS UNROLL
				st[k] = st[k + 1]; // shift registers in action
				}
      st[REG_SIZE - 1] = tmp_pi; 
      
      for (int k = 0; k < REG_SIZE; k++) {
				#pragma HLS UNROLL
        if (st[k].prob > reg[k].prob) {
          ProbIndex swap = reg[k];
          reg[k] = st[k];
          st[k] = swap;
        }
      }
    // }
  }
}
  
void insertion_sort(hls::stream<ProbIndex> &ss_val, ProbIndex *reg, const int M_DIM){
  // #pragma HLS INLINE
  // ProbIndex init = {0, std::numeric_limits<float>::lowest()};
  ProbIndex st[REG_SIZE]; // systolic temp array
  // ProbIndex reg[REG_SIZE]; // stored 64 values
	
	#pragma HLS ARRAY_PARTITION variable=st complete dim=1
  //  // up a level
	init_loop:
	for (int ii = 0; ii < REG_SIZE; ii++) {
		#pragma HLS UNROLL
		st[ii] = {-1, std::numeric_limits<my_float_t>::lowest()};
		// reg[ii] = {-1, std::numeric_limits<my_float_t>::lowest()};
	}
	
  systolic_sort:
  for (int i = 0; i < (M_DIM + REG_SIZE); i++) {
		#pragma HLS LOOP_TRIPCOUNT max=(MODEL_TOKENS + REG_SIZE)
		#pragma HLS pipeline II=3
		ProbIndex tmp_pi = ss_val.read();
		ProbIndex current = tmp_pi;
		
		for (int k = 0; k < REG_SIZE; k++) {
			#pragma HLS UNROLL
			if (current.prob > st[k].prob) {
				ProbIndex swap = st[k];
				st[k] = current;
				current = swap;
			}
		}
  }
	
	copy_out:
	for (int i = 0; i < REG_SIZE; i++) {
		#pragma HLS UNROLL
		reg[i] = st[i];
	}
}
  
  // finished sort. reg now should have largest REG_SIZE (64) values, with max_val @ reg[REG_SIZE-1]
/* ================================================= separate function ===============================*/
void ss_final(ProbIndex *reg, fdata_v_t *pick, const float temperature, const float topp, const float coin){
  
	const my_float_t INV_TEMP = (temperature == 0) ? 1.0f : 1/temperature; 
	fdata_v_t tpick;
	int32_t padd_pick = (int32_t) reg[REG_SIZE - 1].index;
	if (temperature < 0.0f) {
		// if greedy selection or w/e, bypass it all and just return the biggest value.
		// int32_t padd_pick = (int32_t) reg[REG_SIZE - 1].index;
		tpick[0] = reinterpret_cast<float_t&>(padd_pick);
		pick[0] = tpick;
		return;
	}
	
  my_float_t max_val = reg[(REG_SIZE - 1)].prob;
  my_float_t final_soft_sum = 0.0f;
	my_float_t sm_reg[REG_SIZE];
	#pragma HLS ARRAY_PARTITION variable=sm_reg dim=1 type=complete

  softmax_exp_loop:
	for (int i = 0; i < REG_SIZE; i++) {
    #pragma HLS PIPELINE
    my_float_t curr_val = reg[i].prob;
		my_float_t calc = hls::expf((curr_val - max_val) * INV_TEMP);
		final_soft_sum += calc;
		sm_reg[i] = calc;
	}
	my_float_t inv_soft_sum = 1.0f/final_soft_sum;
	int sel_val = reg[REG_SIZE -1].index;
	my_float_t accum_top{};
	const my_float_t target_val = (topp < coin) ? topp : coin;
	// if temperature is zero, then it's gready. If not, then temp 

	softmax_normalize_loop:
	for (int i = (REG_SIZE - 1); i >= 0; i--) {
    #pragma HLS PIPELINE
		accum_top += sm_reg[i] * inv_soft_sum;
    // reg[i].prob *= inv_soft_sum;
		if (accum_top > target_val) {
			sel_val = reg[i].index;
			break;
		}
	}  
  // tpick[0] = (my_float_t) sel_val;
	tpick[0] = reinterpret_cast<float_t&>(sel_val);
	pick[0] = tpick;
  return;
}

template<typename T, size_t N, int P>
void gemv_split(hls::vector<T, N> *out, hls::stream<ProbIndex> &sys_sort, hls::stream<T> (&gemv_out)[P], const int M_DIM, const bool BOOP){
	
	const int offset = INTERNAL_DATA_SIZE / (N * P);
	typedef hls::vector<T, N> gdata_v_t;
	const int c_idx = M_DIM / (P * N);
	for (int i = 0; i < c_idx; i++) {
		#pragma hls LOOP_TRIPCOUNT min=(MODEL_ELEMENTS / (P * N)) max=(MODEL_TOKENS / (P * N))
		for (int j = 0; j < P; j++) {
			// #pragma HLS UNROLL
			gdata_v_t data;
			int idx = i + j*offset;
			for (int k = 0; k < N; k++) {
				#pragma HLS PIPELINE II=1
				
				T temp = gemv_out[j].read();
				ProbIndex ss_val;
				data[k] = temp;
				if (BOOP) {
					ss_val.prob = std::numeric_limits<my_float_t>::lowest();
				} else {
					ss_val.prob = temp;
				}
				ss_val.index = idx + k;

				sys_sort.write(ss_val);
			}
			if (BOOP) {
				out[idx] = data;
			}
			// out[idx] = data;
		}
	}

	ProbIndex ss_val = {32420, std::numeric_limits<my_float_t>::lowest()};
	
	flush:
	for (int i = 0; i < REG_SIZE; i++) {
		#pragma HLS PIPELINE II=1
		sys_sort.write(ss_val);
	}
}