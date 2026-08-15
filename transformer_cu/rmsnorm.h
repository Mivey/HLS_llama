#ifndef MARK_RMS
#define MARK_RMS
#include "mha_forward.h"
#include <cmath>
#include <cstddef>
#include <hls_math.h>
#include <hls_vector.h>

// void rmsnorm_kernel(s_fdata_v_t &s_tokens_out, fdata_v_t *diff, fdata_v_t *weights, fdata_v_t *res_con, const int CURR_LAYER, const int INIT, const int offset);
void rmsnorm_kernel(s_fdata_v_t &s_tokens_out, fdata_v_t *diff, fdata_v_t *weights, fdata_v_t *res_con, const int CURR_LAYER, const int offset);

template<typename T, size_t N>
void rmsnorm(hls::stream<hls::vector<T, N>> &o, hls::stream<hls::vector<T, N>> &d, hls::vector<T, N> *x, hls::stream<hls::vector<T, N>> &w){
  
  typedef hls::vector<T, N> tmp_t;
  tmp_t arr[MODEL_ELEMENTS/N];
  const int acc_lag = 16;
  float_t ss[acc_lag]{};
  
  rms_mac:
  for (int i = 0; i < (MODEL_ELEMENTS / N); i++) {
    #pragma HLS PIPELINE
    
		// fdata_v_t xbl = x[i];
		// x[i] += d.read();
	
    tmp_t x_part = x[i] + d.read();
		arr[i] = x_part;
    float_t psum{};
    
    for (int j = 0; j < N; j++) {
			#pragma HLS UNROLL
			my_float_t tss = x_part[j] * x_part[j];
			psum += tss; 
    }
	ss[i % acc_lag] += psum;
	// arr[i] = x[i];
  }
	float_t ftss = 0.0f;
	
	rms_sum:
	for (int i = 0; i < acc_lag; i++) {
		#pragma HLS UNROLL
		ftss += ss[i];
	}

  float_t flss = (ftss / MODEL_ELEMENTS + 1e-5);
  // float_t fdss = hls::sqrtf(flss);
	// float_t fss = hls::recipf(fdss);
	float_t fss = 1.0f / (hls::sqrtf(flss));

  data_out:
  for (int i = 0 ; i < MODEL_ELEMENTS/N; i++) {
    #pragma HLS PIPELINE II=1
		tmp_t vw = w.read();
    tmp_t tw = vw * fss;
		tmp_t tws = tw * arr[i];
    o.write(tws);
		x[i] = arr[i];
  }
}


#endif
