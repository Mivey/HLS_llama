#include "../forward.h"

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
int rqm_rope_mha_tb(){
	std::cout<<"starting First Third testbench"<<std::endl;
	/* ===================== open neede files =========================== */

	/* ========== input files ============================= */
	std::ifstream input_tokens_dat("top_data/TP_25_tokens.bin", std::ios::binary);

	std::ifstream rms_att_w("top_data/TP_25_rms_att_weight.bin", std::ios::binary);

	std::ifstream wq_sf_dat("top_data/TP_25_quant_wq_sf.bin", std::ios::binary);
	std::ifstream wq_q_dat("top_data/TP_25_quant_wq.bin", std::ios::binary);
	std::ifstream wk_sf_dat("top_data/TP_25_quant_wk_sf.bin", std::ios::binary);
	std::ifstream wk_q_dat("top_data/TP_25_quant_wk.bin", std::ios::binary);
	std::ifstream wv_sf_dat("top_data/NTP_25_quant_wv_sf.bin", std::ios::binary);
	std::ifstream wv_q_dat("top_data/NTP_25_quant_wv.bin", std::ios::binary);

	std::ifstream key_cache_dat("top_data/TP_key_cache.bin", std::ios::binary);
	std::ifstream value_cache_dat("top_data/TP_value_cache.bin", std::ios::binary);

	std::ifstream wo_sf_dat("top_data/TP_25_quant_wo_att_out_sf.bin", std::ios::binary);
	std::ifstream wo_q_dat("top_data/TP_25_quant_wo_att_out.bin", std::ios::binary);

	/* =========================== output data ===================== */
	std::ifstream out_query_dat("top_data/AP_25_out_query.bin", std::ios::binary);
	std::ifstream out_value_dat("top_data/NTP_25_out_value.bin", std::ios::binary);// probably wrong. regen and rename
	std::ifstream out_key_dat("top_data/TOP_25_rope_out_key.bin", std::ios::binary);
	std::ifstream output_tokens_dat("top_data/FTP_25_token_res_con_out.bin", std::ios::binary);


/* =================== check if files opened successfully */
	if (!input_tokens_dat.is_open() ) {
	std::cout<<"input_tokens_dat. Already off to a bad start. boop."<<std::endl;
	exit(EXIT_FAILURE);
	}
	if (!rms_att_w.is_open() ) {
	std::cout<<"rms_att_w. Already off to a bad start."<<std::endl;
	exit(EXIT_FAILURE);
	}
	
	if (!wq_sf_dat.is_open() ) {
	std::cout<<"wq_sf_dat. Already off to a bad start."<<std::endl;
	exit(EXIT_FAILURE);
	}
	if (!wq_q_dat.is_open() ) {
	std::cout<<"wq_q_dat. Already off to a bad start."<<std::endl;
	exit(EXIT_FAILURE);
	}

	if (!wk_sf_dat.is_open() ) {
	std::cout<<"wk_sf_dat. Already off to a bad start."<<std::endl;
	exit(EXIT_FAILURE);
	}
	if (!wk_q_dat.is_open() ) {
	std::cout<<"wk_q_dat. Already off to a bad start."<<std::endl;
	exit(EXIT_FAILURE);
	}
	
	if (!wv_sf_dat.is_open() ) {
	std::cout<<"wv_sf_dat. Already off to a bad start."<<std::endl;
	exit(EXIT_FAILURE);
	}
	if (!wv_q_dat.is_open() ) {
	std::cout<<"wv_q_dat. Already off to a bad start."<<std::endl;
	exit(EXIT_FAILURE);
	}

	if (!out_query_dat.is_open() ) {
	std::cout<<"out_query_dat. Already off to a bad start."<<std::endl;
	exit(EXIT_FAILURE);
	}
	if (!out_value_dat.is_open() ) {
	std::cout<<"out_value_dat. Already off to a bad start."<<std::endl;
	exit(EXIT_FAILURE);
	}
	if (!out_key_dat.is_open() ) {
	std::cout<<"out_key_dat. Already off to a bad start."<<std::endl;
	exit(EXIT_FAILURE);
	}

	std::cout<<"Opened all the files sucessfully"<<std::endl;

	/* ============================== constants related to tb ===================================== */
	const int pos = 150;

	const int out_data_size = MODEL_ELEMENTS * 4;
	const int quant_data_size = MODEL_ELEMENTS * MODEL_ELEMENTS * 1;
	const int sf_data_size = MODEL_ELEMENTS * MODEL_ELEMENTS * 4 / MODEL_SCALING_FACTOR;
	const int rms_w_size = MODEL_ELEMENTS * 4;
	const int tokens_size = MODEL_ELEMENTS * 4;
	const int wo_size = 589824;
	const int wo_sf_size = 36864;
	const int cache_size = 1024*768*4 * 12;
	
	const int out_data_cnt = MODEL_ELEMENTS / MAX_W_F;// out_data_size / sizeof(fdata_v_t);
	const int quant_data_cnt = quant_data_size / sizeof(idata_v_t);
	const int sf_data_cnt = sf_data_size / sizeof(fdata_v_t);
	const int rms_w_cnt = rms_w_size / sizeof(fdata_v_t);
	const int tokens_cnt = tokens_size / sizeof(fdata_v_t);
	const int wo_cnt = wo_size / sizeof(idata_v_t);
	const int wo_sf_cnt = wo_sf_size / sizeof(fdata_v_t);
	const int cache_cnt = cache_size / sizeof(fdata_v_t);
	

/* ===================================== declare our vectors ===================================== */
	std::vector<fdata_v_t> tokens_arr(tokens_cnt);

	std::vector<fdata_v_t> rms_w_arr(rms_w_cnt);

	std::vector<idata_v_t> wq_q_arr(quant_data_cnt);
	std::vector<fdata_v_t> wq_sf_arr(sf_data_cnt);

	std::vector<idata_v_t> wk_q_arr(quant_data_cnt);
	std::vector<fdata_v_t> wk_sf_arr(sf_data_cnt);

	std::vector<idata_v_t> wv_q_arr(quant_data_cnt);
	std::vector<fdata_v_t> wv_sf_arr(sf_data_cnt);

	std::vector<idata_v_t> wo_q_arr(wo_cnt);
	std::vector<fdata_v_t> wo_sf_arr(wo_sf_cnt);


	std::vector<fdata_v_t> okey_arr(cache_cnt);
	std::vector<fdata_v_t> ovalue_arr(cache_cnt);

	std::vector<std::vector<fdata_v_t>> query_arr(2, std::vector<fdata_v_t>(out_data_cnt));
	std::vector<std::vector<fdata_v_t>> key_arr(2, std::vector<fdata_v_t>(out_data_cnt));
	std::vector<std::vector<fdata_v_t>> value_arr(2, std::vector<fdata_v_t>(out_data_cnt));
	std::vector<std::vector<fdata_v_t>> out_tok_arr(2, std::vector<fdata_v_t>(tokens_cnt));

	/* ================================== read data into array =================================== */

	input_tokens_dat.read(reinterpret_cast<char *>(tokens_arr.data()), tokens_size);

	rms_att_w.read(reinterpret_cast<char *>(rms_w_arr.data()), rms_w_size);

	wq_q_dat.read(reinterpret_cast<char *>(wq_q_arr.data()), quant_data_size);
	wq_sf_dat.read(reinterpret_cast<char *>(wq_sf_arr.data()), sf_data_size);

	wk_q_dat.read(reinterpret_cast<char *>(wk_q_arr.data()), quant_data_size);
	wk_sf_dat.read(reinterpret_cast<char *>(wk_sf_arr.data()), sf_data_size);

	wv_q_dat.read(reinterpret_cast<char *>(wv_q_arr.data()), quant_data_size);
	wv_sf_dat.read(reinterpret_cast<char *>(wv_sf_arr.data()), sf_data_size);

	wo_q_dat.read(reinterpret_cast<char *>(wo_q_arr.data()), wo_size);
	wo_sf_dat.read(reinterpret_cast<char *>(wo_sf_arr.data()), wo_sf_size);
	std::cout<<"this is fine 2"<<std::endl;

	key_cache_dat.read(reinterpret_cast<char *>(okey_arr.data()), cache_size);
	value_cache_dat.read(reinterpret_cast<char *>(ovalue_arr.data()), cache_size);

	out_query_dat.read(reinterpret_cast<char *>(query_arr[0].data()), out_data_size);
	out_key_dat.read(reinterpret_cast<char *>(key_arr[0].data()), out_data_size);
	out_value_dat.read(reinterpret_cast<char *>(value_arr[0].data()), out_data_size);
	input_tokens_dat.read(reinterpret_cast<char *>(out_tok_arr[0].data()), tokens_size);
	std::cout<<"Loaded the files into memory"<<std::endl;


/* ===================================== Declare the streams ========================================= */
/* remember - if it's suppsoed to be m_axi, no need to create a stream. just use the created vector, ie: foo_arr.data() */

	hls::stream<fdata_v_t> tokens_in("Tokens data input");

	hls::stream<fdata_v_t> query_out("Query data output");

	s_fdata_v_t tokens_out_stream("output tokens stream");

	/* ============================ write inputs to the streams ====================== */
		
	tokens_data_loop:
	// for (int i = 0; i < tokens_cnt; i++) {
	// 	tokens_in.write(tokens_arr[i]);
	// }

	// for (int i = 0;  i < sf_data_cnt; i++) {
	// 	fdata_v_t foo = wv_sf_arr[i];
	// 	idata_v_t bar = wv_q_arr[i];
	// }

	// query_data_loop:
	// for (int i = 0; i < out_data_cnt; i++) {
	// 	query_out.write(query_arr[0][i]);
	// }
	std::cout<<"Delcared and Loaded the Streams"<<std::endl;

	/* ============================ Call the function(s) ====================================== */

	mtop(tokens_out_stream, key_arr[1].data(), value_arr[1].data(), tokens_arr.data(), okey_arr.data(), ovalue_arr.data(), rms_w_arr.data(), wq_sf_arr.data(), wq_q_arr.data(), wk_sf_arr.data(), wk_q_arr.data(), wv_sf_arr.data(), wv_q_arr.data(), wo_sf_arr.data(), wo_q_arr.data(), 150, 0);
	
	/* =============================== get all the data =================================== */

	std::cout<<"Parse Kernel Outputs"<<std::endl;
	// for (int i = 0; i < out_data_cnt; i++) {
	// 	query_arr[1][i] = query_out.read();
	// }
	for (int i = 0; i < out_data_cnt; i++) {
		out_tok_arr[1][i] = tokens_out_stream.read();
	}
	std::cout<<"Process Results"<<std::endl;


	/*  ====================================== process the results ============================ */

	std::cout<< "========================= Query array data ========================"<<std::endl;
	// parse_results<fdata_v_t, float>(query_arr[0], query_arr[1]);

	std::cout<< "========================= Value array data ========================"<<std::endl;
	// parse_results<fdata_v_t, float>(value_arr[0], value_arr[1]);

	std::cout<< "========================= Key array data ========================"<<std::endl;
	// parse_results<fdata_v_t, float>(key_arr[0], key_arr[1]);

	std::cout<< "========================= Tokens array data ========================"<<std::endl;
	parse_results<fdata_v_t, float>(tokens_arr, out_tok_arr[1]);

	return 0;

}