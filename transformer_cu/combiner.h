#include "mha_forward.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <hls_vector.h>
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