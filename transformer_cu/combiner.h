#include "mha_forward.h"
// #include <cstdint>
// #include <ios>


/*====================================================================================================================================*/

constexpr int REG_SIZE = 64;
constexpr int ARR_PART = 2;

struct ProbIndex{
  short index;
  my_float_t prob;
};

  
void insertion_sort(hls::stream<ProbIndex> &ss_val, ProbIndex *reg, const int M_DIM){//
  ProbIndex st[REG_SIZE]; // systolic temp array
  
  #pragma HLS ARRAY_PARTITION variable=st complete dim=1
  //  // up a level
  init_loop:
  for (int ii = 0; ii < REG_SIZE; ii++) {
    #pragma HLS UNROLL
    st[ii] = {-1, std::numeric_limits<my_float_t>::lowest()};
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
  
void ss_final(ProbIndex *reg, int32_t &pick, const float temperature, const float topp, const float coin){//
  
  const my_float_t INV_TEMP = (temperature == 0) ? 1.0f : 1/temperature; 
  if (temperature <= 0.0f) {
    // if greedy selection or w/e, bypass it all and just return the biggest value.
    // int32_t padd_pick = (int32_t) reg[REG_SIZE - 1].index;
    // tpick[0] = reinterpret_cast<float_t&>(padd_pick);
    pick = (int32_t) reg[0].index;;
    return;
  }
  
  my_float_t max_val = reg[0].prob;
  my_float_t final_soft_sum = 0.0f;
  my_float_t sm_reg[REG_SIZE];
	// int delme[REG_SIZE];
  #pragma HLS ARRAY_PARTITION variable=sm_reg dim=1 type=complete

  softmax_exp_loop:
  for (int i = 0; i < REG_SIZE; i++) {
    #pragma HLS PIPELINE
    my_float_t curr_val = reg[i].prob;
		// delme[i] = (int) reg[i].index;
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
  for (int i = 0; i < REG_SIZE; i++) {
  // for (int i = (REG_SIZE - 1); i >= 0; i--) {
    #pragma HLS PIPELINE
    accum_top += sm_reg[i] * inv_soft_sum;
    // reg[i].prob *= inv_soft_sum;
    if (accum_top > target_val) {
      sel_val = reg[i].index;
      break;
    }
  }  
	pick = sel_val;
  return;
}

template<typename T, size_t N, int P>//
void gemv_split(hls::vector<T, N> *out, hls::stream<ProbIndex> &sys_sort, hls::stream<T> (&gemv_out)[P], 
                const int M_DIM, const bool BOOP//, hls::stream<bool> &done
                #ifdef __ULTRADEBUG__
                  , fdata_v_t *data_out, const int SAVE_ADDR
                #endif
                ){
  
  const int offset = INTERNAL_DATA_SIZE / (N * P);
  typedef hls::vector<T, N> gdata_v_t;
  const int c_idx = M_DIM / (P * N);
  for (int i = 0; i < c_idx; i++) {
    #pragma hls LOOP_TRIPCOUNT min=(MODEL_ELEMENTS / (P * N)) max=(MODEL_TOKENS / (P * N))
    for (int j = 0; j < P; j++) {
      // #pragma HLS UNROLL
      gdata_v_t data;
      int idx = i + j*offset;
      int fidx = i * 4 + j * (MODEL_TOKENS / 2);
      for (int k = 0; k < N; k++) {
        #pragma HLS PIPELINE II=1
        
        T temp = gemv_out[j].read();
        ProbIndex ss_val;
        data[k] = temp;
        if (BOOP) {
          ss_val.prob = std::numeric_limits<my_float_t>::lowest();
        } else {
          ss_val.prob = temp;
          // ss_val.index = k + j * 4
        }
        ss_val.index = fidx + k;

        sys_sort.write(ss_val);
      }
      if (BOOP) {
        out[idx] = data;
        #ifdef __ULTRADEBUG__
          int udx = i + j * (M_DIM / (N * P)) + SAVE_ADDR;
          data_out[udx] = data;
        #endif
      }
      // out[idx] = data;
    }
  }
  // bool next = done.read(); // lets cu_selecter start the next calculation. 
  ProbIndex ss_val = {32420, std::numeric_limits<my_float_t>::lowest()};
  
  flush:
  for (int i = 0; i < REG_SIZE; i++) {
    #pragma HLS PIPELINE II=1
    sys_sort.write(ss_val);
  }
}
