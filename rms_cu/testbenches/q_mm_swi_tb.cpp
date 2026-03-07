#include "../forward.h"
#include "../matmul.h"
#include "../mha.h"
#include <cstdio>
#include <hls_math.h>
#include <hls_stream.h>
#include <stdio.h>
#include <streambuf>
#include <vector>
#include <fstream>
#include <iostream>
#include <cmath>
#include <bitset>
#include "tb_main.h"


// int mat_mul_tb(){
int rqm_swiglu_tb(){
	std::cout<<"starting mha testbench"<<std::endl;
	/* ===================== open neede files =========================== */

	/* ========== input files ============================= */
	std::ifstream rms_ffn_dat("top_data/FTP_25_rms_ffn_weight.bin", std::ios::binary);
	std::ifstream token_res_con_dat("top_data/FTP_25_token_res_con_out.bin", std::ios::binary);
	
	std::ifstream w1_sf_dat("top_data/FTP_25_quant_w1_sf.bin", std::ios::binary);
	std::ifstream w1_q_dat("top_data/FTP_25_quant_w1.bin", std::ios::binary);

	std::ifstream w3_sf_dat("top_data/FTP_25_quant_w3_sf.bin", std::ios::binary);
	std::ifstream w3_q_dat("top_data/FTP_25_quant_w3.bin", std::ios::binary);

	std::ifstream w2_sf_dat("top_data/FTP_25_quant_w2_sf.bin", std::ios::binary);
	std::ifstream w2_q_dat("top_data/FTP_25_quant_w2.bin", std::ios::binary);

	/* =========================== output data ===================== */
	std::ifstream tokens_output("top_data/TOP_25_tokens.bin", std::ios::binary);


/* =================== check if files opened successfully */
	if (!rms_ffn_dat.is_open() ) {
	std::cout<<"rms_ffn_dat. Already off to a bad start. boop."<<std::endl;
	exit(EXIT_FAILURE);
	}
	if (!token_res_con_dat.is_open() ) {
	std::cout<<"token_res_con_dat. Already off to a bad start."<<std::endl;
	exit(EXIT_FAILURE);
	}
	
	if (!w1_sf_dat.is_open() ) {
	std::cout<<"w1_sf_dat. Already off to a bad start."<<std::endl;
	exit(EXIT_FAILURE);
	}
	if (!w1_q_dat.is_open() ) {
	std::cout<<"w1_q_dat. Already off to a bad start."<<std::endl;
	exit(EXIT_FAILURE);
	}
	
	if (!w3_sf_dat.is_open() ) {
	std::cout<<"w3_sf_dat. Already off to a bad start."<<std::endl;
	exit(EXIT_FAILURE);
	}
	if (!w3_q_dat.is_open() ) {
	std::cout<<"w3_q_dat. Already off to a bad start."<<std::endl;
	exit(EXIT_FAILURE);
	}
	
	if (!w2_sf_dat.is_open() ) {
	std::cout<<"w2_sf_dat. Already off to a bad start."<<std::endl;
	exit(EXIT_FAILURE);
	}
	if (!w2_q_dat.is_open() ) {
	std::cout<<"w2_q_dat. Already off to a bad start."<<std::endl;
	exit(EXIT_FAILURE);
	}

	if (!tokens_output.is_open() ) {
	std::cout<<"tokens_output. Already off to a bad start."<<std::endl;
	exit(EXIT_FAILURE);
	}

	/* ============================== constants related to tb ===================================== */
	const int pos = 150;
	const int hd_sf_size = MODEL_ELEMENTS * MODEL_HIDDEN_DIM / MODEL_SCALING_FACTOR * 4;
	const int hd_tok_size = MODEL_ELEMENTS * MODEL_HIDDEN_DIM * 1;
	const int rms_tok_size = 768 * 4;

	const int hd_tok_cnt = hd_tok_size / sizeof(idata_v_t);
	const int hd_sf_cnt = hd_sf_size / sizeof(fdata_v_t);
	const int rms_tok_cnt = rms_tok_size / sizeof(fdata_v_t);

/* ===================================== declare our vectors ===================================== */
	std::vector<fdata_v_t> rms_ffn_arr(rms_tok_cnt);
	std::vector<fdata_v_t> tok_rescon_arr(rms_tok_cnt);

	std::vector<idata_v_t> w1_q_arr(hd_tok_cnt);
	std::vector<fdata_v_t> w1_sf_arr(hd_sf_cnt);

	std::vector<idata_v_t> w3_q_arr(hd_tok_cnt);
	std::vector<fdata_v_t> w3_sf_arr(hd_sf_cnt);

	std::vector<idata_v_t> w2_q_arr(hd_tok_cnt);
	std::vector<fdata_v_t> w2_sf_arr(hd_sf_cnt);
	
	std::vector<std::vector<fdata_v_t>> tok_out_arr(2, std::vector<fdata_v_t>(rms_tok_cnt));

	/* ================================== read data into array =================================== */

	rms_ffn_dat.read(reinterpret_cast<char *>(rms_ffn_arr.data()), rms_tok_size);
	token_res_con_dat.read(reinterpret_cast<char *>(tok_rescon_arr.data()), rms_tok_size);

	w1_q_dat.read(reinterpret_cast<char *>(w1_q_arr.data()), hd_tok_size);
	w1_sf_dat.read(reinterpret_cast<char *>(w1_sf_arr.data()), hd_sf_size);

	w3_q_dat.read(reinterpret_cast<char *>(w3_q_arr.data()), hd_tok_size);
	w3_sf_dat.read(reinterpret_cast<char *>(w3_sf_arr.data()), hd_sf_size);

	w2_q_dat.read(reinterpret_cast<char *>(w2_q_arr.data()), hd_tok_size);
	w2_sf_dat.read(reinterpret_cast<char *>(w2_sf_arr.data()), hd_sf_size);

	tokens_output.read(reinterpret_cast<char *>(tok_out_arr[0].data()), rms_tok_size);


/* ===================================== Declare the streams ========================================= */
/* remember - if it's suppsoed to be m_axi, no need to create a stream. just use the created vector, ie: foo_arr.data() */

	// hls::stream<fdata_v_t> rms_ffn_in("RMS FFN data input");
	hls::stream<fdata_v_t> tok_rc_in("Token Residual Connection data input");

	// hls::stream<idata_v_t> w1_q_in("W1 Quantized data input");
	// hls::stream<fdata_v_t> w1_sf_in("W1 Quantized SF data input");

	// hls::stream<idata_v_t> w3_q_in("W3 Quantized data input");
	// hls::stream<fdata_v_t> w3_sf_in("W3 Quantized SF data input");

	// hls::stream<idata_v_t> w2_q_in("W2 Quantized data input");
	// hls::stream<fdata_v_t> w2_sf_in("W2 Quantized SF data input");

	/* ============================ write inputs to the streams ====================== */
	
	
	token_rc_data_loop:
	for (int i = 0; i < rms_tok_cnt; i++) {
		tok_rc_in.write(tok_rescon_arr[i]);
	}



	/* ============================ Call the function(s) ====================================== */

final_third(tok_out_arr[1].data(), rms_ffn_arr.data(), w1_sf_arr.data(), w1_q_arr.data(), w2_sf_arr.data(), w2_q_arr.data(), w3_sf_arr.data(), w3_q_arr.data(), tok_rc_in);
// final_third(tok_out_arr[1].data(), rms_ffn_arr.data(), w1_sf_arr.data(), w1_q_arr.data(), w2_sf_arr.data(), w2_q_arr.data(), w3_sf_arr.data(), w3_q_arr.data(), s_fdata_v_t &tokens_in)
	/* =============================== get all the data =================================== */

// int qq = xb2_before_arr.size();
// 	for (int i = 0; i < xb2_cnt; i++) {
// 		xb2_arr[1][i] = xb2_out.read();
// 	}


	/*  ====================================== process the results ============================ */

	std::cout<< "========================= xb2 array data ========================"<<std::endl;
	parse_results<fdata_v_t, float>(tok_out_arr[0], tok_out_arr[1]);

	return 0;

}