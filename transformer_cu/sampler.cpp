#include "mha_forward.h"
#include <limits>

constexpr int REG_SIZE = 64;

struct ProbIndex{
  short index = -1;
  my_float_t prob = std::numeric_limits<float>::lowest();
};

void systolic_sort(fdata_v_t *logit, int pick, const float temperature, const float coin){
    
  // ProbIndex init = {0, std::numeric_limits<float>::lowest()};
  ProbIndex st[REG_SIZE]; // systolic temp array
  ProbIndex reg[REG_SIZE]; // stored 64 values
  systolic_sort:
  for (int i = 0; i < MODEL_TOKENS / SM_FL_ELEM; i++) {
    fdata_v_t temp = logit[i];
    for (int j = 0; j < SM_FL_ELEM; j++) {
      #pragma HLS PIPELINE
      //shift register 
      for (int k = 0; k < (REG_SIZE - 1); k++) { st[i] = st[i + 1]; }
      st[REG_SIZE - 1] = {(i * SM_FL_ELEM + j), temp[j]};
      
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
    for (int k = 0; k < (REG_SIZE - 1); k++) { st[i] = st[i + 1]; }
    st[REG_SIZE - 1] = {-1, std::numeric_limits<float>::lowest()};
    
    for (int k = 0; k < REG_SIZE; k++) {
      if (st[k].prob > reg[k].prob) {
        ProbIndex swap = reg[k];
        reg[k] = st[k];
        st[k] = swap;
      }
    }
  }
  
  // finished sort. reg now should have largest REG_SIZE (64) values, with max_val @ reg[REG_SIZE-1]

  //
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
  
  coin_flip:
  for (int i = (REG_SIZE - 1); i >= 0; i--) {
    coin_sum += reg[i].prob;
    pick = reg[i].index;
    if (coin_sum > coin) {
      break;
    }
  }
  return;
}