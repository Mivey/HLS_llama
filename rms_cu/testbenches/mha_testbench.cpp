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
int mha_tb(){
	std::cout<<"starting mha testbench"<<std::endl;
	/* ===================== open neede files =========================== */

	/* ========== input files ============================= */
	std::ifstream input_tokens_dat("top_data/TP_25_tokens.bin", std::ios::binary);
	std::ifstream query_dat("top_data/AP_25_out_query.bin", std::ios::binary);

	std::ifstream key_cache_dat("top_data/FTP_25_head_major_key_out.bin", std::ios::binary);
	std::ifstream value_cache_dat("top_data/FTP_25_head_major_value_out.bin", std::ios::binary);

	std::ifstream wo_sf_dat("top_data/TP_25_quant_wo_att_out_sf.bin", std::ios::binary);
	std::ifstream wo_q_dat("top_data/TP_25_quant_wo_att_out.bin", std::ios::binary);

	/* =========================== output data ===================== */
	std::ifstream xb2_dat("top_data/TOP_25_xb2_mm_output.bin", std::ios::binary);
	std::ifstream xb2_before_dat("top_data/TOP_25_xb2_mm_output_A1.bin", std::ios::binary);
	// std::ifstream att_score_dat("top_data/AP_25_att_score.bin", std::ios::binary);
	// std::ifstream att_score_dat("top_data/AP_25_att_softmax.bin", std::ios::binary);
	std::ifstream att_score_dat("top_data/AP_25_xb_before.bin", std::ios::binary);
	std::ifstream output_tokens_dat("top_data/FTP_25_token_res_con_out.bin", std::ios::binary);
	//att_sm_ws


/* =================== check if files opened successfully */
	if (!input_tokens_dat.is_open() ) {
	std::cout<<"input_tokens_dat. Already off to a bad start. boop."<<std::endl;
	exit(EXIT_FAILURE);
	}
	if (!query_dat.is_open() ) {
	std::cout<<"query_dat. Already off to a bad start. boop."<<std::endl;
	exit(EXIT_FAILURE);
	}
	if (!key_cache_dat.is_open() ) {
	std::cout<<"key_cache_dat. Already off to a bad start."<<std::endl;
	exit(EXIT_FAILURE);
	}
	
	if (!value_cache_dat.is_open() ) {
	std::cout<<"value_cache_dat. Already off to a bad start."<<std::endl;
	exit(EXIT_FAILURE);
	}
	if (!att_score_dat.is_open() ) {
	std::cout<<"attscoretad. Already off to a bad start."<<std::endl;
	exit(EXIT_FAILURE);
	}
	if (!wo_sf_dat.is_open() ) {
	std::cout<<"wo_sf_dat. Already off to a bad start."<<std::endl;
	exit(EXIT_FAILURE);
	}

	if (!wo_q_dat.is_open() ) {
	std::cout<<"wo_q_dat. Already off to a bad start."<<std::endl;
	exit(EXIT_FAILURE);
	}
	if (!xb2_dat.is_open() ) {
	std::cout<<"xb2_dat. Already off to a bad start."<<std::endl;
	exit(EXIT_FAILURE);
	}
	if (!xb2_before_dat.is_open() ) {
	std::cout<<"xb2_before_dat. Already off to a bad start."<<std::endl;
	exit(EXIT_FAILURE);
	}
	if (!output_tokens_dat.is_open() ) {
	std::cout<<"output_tokens_dat. Already off to a bad start."<<std::endl;
	exit(EXIT_FAILURE);
	}

	/* ============================== constants related to tb ===================================== */
	const int pos = 150;
	const int sf_el = MODEL_ELEMENTS / 64;
	const int wo_sf_size = 36864;
	const int wo_size = 589824;
	const int cache_size = 1024*768*4 * 12;
	const int t_size = 768 * 4;
	const int xb2_size = 3072;
	const int tokens_size = MODEL_ELEMENTS * 4;

	const int wo_cnt = wo_size / sizeof(idata_v_t);
	const int wo_sf_cnt = wo_sf_size / sizeof(fdata_v_t);
	const int cache_cnt = cache_size / sizeof(fdata_v_t);
	const int tok_cnt = t_size / sizeof(mfdata_v_t);
	const int tokens_cnt = tokens_size / sizeof(mfdata_v_t);
	const int xb2_cnt = xb2_size / sizeof(mfdata_v_t);		// (768 elements  * 4 bytes/elements) / (4 elements/ point * 4 bytes/elemnts) = 192 points

/* ===================================== declare our vectors ===================================== */
	std::vector<mfdata_v_t> tokens_arr(tokens_cnt);
	std::vector<mfdata_v_t> query_arr(tok_cnt);

	std::vector<idata_v_t> wo_q_arr(wo_cnt);
	std::vector<fdata_v_t> wo_sf_arr(wo_sf_cnt);


	std::vector<mfdata_v_t> key_arr(cache_cnt);
	std::vector<mfdata_v_t> value_arr(cache_cnt);

	std::vector<std::vector<mfdata_v_t>> xb2_arr(2, std::vector<mfdata_v_t>(xb2_cnt));
	std::vector<std::vector<mfdata_v_t>> xb2_before_arr(2, std::vector<mfdata_v_t>(xb2_cnt));
	std::vector<std::vector<my_float_t>> att_score_arr(2, std::vector<my_float_t>(pos));
	std::vector<std::vector<mfdata_v_t>> tokens_out_arr(2, std::vector<mfdata_v_t>(tokens_cnt));

	std::cout<<"this is fine"<<std::endl;

	/* ================================== read data into array =================================== */

	input_tokens_dat.read(reinterpret_cast<char *>(tokens_arr.data()), tokens_size);
	query_dat.read(reinterpret_cast<char *>(query_arr.data()), t_size);

	wo_q_dat.read(reinterpret_cast<char *>(wo_q_arr.data()), wo_size);
	wo_sf_dat.read(reinterpret_cast<char *>(wo_sf_arr.data()), wo_sf_size);
	std::cout<<"this is fine 2"<<std::endl;

	key_cache_dat.read(reinterpret_cast<char *>(key_arr.data()), cache_size);
	value_cache_dat.read(reinterpret_cast<char *>(value_arr.data()), cache_size);

	std::cout<<"this is fine 3"<<std::endl;

	xb2_dat.read(reinterpret_cast<char *>(xb2_arr[0].data()), xb2_size);
	xb2_before_dat.read(reinterpret_cast<char *>(xb2_before_arr[0].data()), xb2_size);
	att_score_dat.read(reinterpret_cast<char *>(att_score_arr[0].data()), pos * 4);
	output_tokens_dat.read(reinterpret_cast<char *>(tokens_out_arr[0].data()), tokens_size);

std::cout<<"this is fine 4"<<std::endl;

/* ===================================== Declare the streams ========================================= */
/* remember - if it's suppsoed to be m_axi, no need to create a stream. just use the created vector, ie: foo_arr.data() */

	hls::stream<mfdata_v_t> qa_in("Query data input");
	hls::stream<idata_v_t> wo_q_in("Wo Quantized data input");
	hls::stream<fdata_v_t> wo_sf_in("Wo Quantized SF data input");

	hls::stream<mfdata_v_t> kc_in("Key Cache data input");
	hls::stream<mfdata_v_t> vc_in("Value Cache data input");
	
	hls::stream<mfdata_v_t> xb2_out("xb2 output Stream");
	hls::stream<mfdata_v_t> xb2_before_out("xb2 before output Stream");
	hls::stream<mfdata_v_t> tokens_in("Input Tokens");
	hls::stream<mfdata_v_t> tokens_out("Input Tokens");
	hls::stream<my_float_t> att_score_out("Output of MHA Iterate");

	/* ============================ write inputs to the streams ====================== */
	
	
	tokens_data_loop:
	for (int i = 0; i < tokens_cnt; i++) {
		tokens_in.write(tokens_arr[i]);
	}
	quant_data_loop:
	for (int i = 0; i < tok_cnt; i++) {
		qa_in.write(query_arr[i]);
	}
	for (int i = 0; i < cache_cnt; i++) {
		vc_in.write(value_arr[i]);
	}
	// quant_data_loop:
	for (int i = 0; i < cache_cnt; i++) {
		kc_in.write(key_arr[i]);
	}
	// std::cout<<"this is fine?"<<std::endl;
	// quant_data_loop:
	for (int i = 0; i < wo_cnt; i++) {
		wo_q_in.write(wo_q_arr[i]);
	}
	// quant_data_loop:
	for (int i = 0; i < wo_sf_cnt; i++) {
		wo_sf_in.write(wo_sf_arr[i]);
	}


	/* ============================ Call the function(s) ====================================== */

	// mha_kernel(xb2_out, key_arr.data(), value_arr.data(), tokens_in, wo_q_in, wo_sf_in, qa_in, 150);
	// test_mha_kernel(xb2_out, key_arr.data(), value_arr.data(),  qa_in, 150, 0);
// mha_kernel(tokens_out, key_arr.data(), value_arr.data(), tokens_in, wo_sf_arr.data(), wo_q_arr.data(), qa_in, 150, 0);
mha_kernel(tokens_out, kc_in, vc_in, tokens_in, wo_sf_in, wo_q_in, qa_in, 150, 0);
// mha_debug(xb2_before_out, att_score_out, kc_in, vc_in, qa_in, pos);

	/* =============================== get all the data =================================== */
std::cout<<"this is fine post"<<std::endl;
int qq = xb2_before_arr.size();
	for (int i = 0; i < xb2_cnt; i++) {
		tokens_out_arr[1][i] = tokens_out.read();
	}
	// for (int i = 0; i < xb2_cnt; i++) {
	// 	tokens_out_arr[1][i] = tokens_out.read();
	// }
	// for (int i = 0; i < pos; i++) {
	// 	float tmp = att_score_out.read();
	// 	att_score_arr[1][i] = tmp;
	// }

	// int pass = 0; 
	// int fail = 0;
	// for (int i = 0; i < pos; i++) {
	// 	att_score_arr[1][i] = att_score_out.read();
	// 	if(att_score_arr[0][i] != att_score_arr[1][i]){
	// 		fail++;
	// 		std::cout<<"Golden:\t"<<att_score_arr[0][i]<<"\t\tResult:\t"<<att_score_arr[1][i]<<std::endl;
	// 	} else {
	// 		pass++;
	// 	}
	// }
	// std::cout<<"passed: "<< pass<<std::endl;
	// std::cout<<"failed: "<< fail<<std::endl;

	/*  ====================================== process the results ============================ */

	std::cout<< "========================= xb2 array data ========================"<<std::endl;
	parse_results<mfdata_v_t, float>(tokens_out_arr[0], tokens_out_arr[1]);
	

	return 0;

}