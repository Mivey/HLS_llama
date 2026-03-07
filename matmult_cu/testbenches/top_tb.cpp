// #include "../forward.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
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


int top_tb(){
	std::cout<<"starting First Third testbench"<<std::endl;
	/* ===================== open neede files =========================== */

	/* ========== input files ============================= */

	std::ifstream wq_sf_dat("weights/query_sf.bin", std::ios::binary);
	std::ifstream wq_q_dat("weights/query_q.bin", std::ios::binary);
	std::ifstream wk_sf_dat("weights/key_sf.bin", std::ios::binary);
	std::ifstream wk_q_dat("weights/key_q.bin", std::ios::binary);
	std::ifstream wv_sf_dat("weights/value_sf.bin", std::ios::binary);
	std::ifstream wv_q_dat("weights/value_q.bin", std::ios::binary);
	
	std::ifstream w1_sf_dat("weights/MLP_1_sf.bin", std::ios::binary);
	std::ifstream w1_q_dat("weights/MLP_1_q.bin", std::ios::binary);

	std::ifstream w3_sf_dat("weights/MLP_3_sf.bin", std::ios::binary);
	std::ifstream w3_q_dat("weights/MLP_3_q.bin", std::ios::binary);
	std::ifstream w2_sf_dat("weights/FFN2_sf.bin", std::ios::binary);
	std::ifstream w2_q_dat("weights/FFN2_w.bin", std::ios::binary);
	/* =========================== output data ===================== */
	std::ifstream key_output("seed_42069/150_output_k_tokens.bin", std::ios::binary);
	std::ifstream w1_output("seed_42069/150_output_w1_tokens.bin", std::ios::binary);
	std::ifstream w2_output("seed_42069/150_output_w2_tokens.bin", std::ios::binary);
	std::ifstream qkv_tokens("seed_42069/qkv_tokens.bin", std::ios::binary);
	std::ifstream w1w3_tokens("seed_42069/w1w3_tokens.bin", std::ios::binary);
	std::ifstream w2_tokens("seed_42069/w2_tokens.bin", std::ios::binary);


/* =================== check if files opened successfully */
	if (!key_output.is_open() ) {
	std::cout<<"key_output. Already off to a bad start."<<std::endl;
	exit(EXIT_FAILURE);
	}
	if (!w1_output.is_open() ) {
	std::cout<<"w1_output. Already off to a bad start."<<std::endl;
	exit(EXIT_FAILURE);
	}
	if (!qkv_tokens.is_open() ) {
	std::cout<<"qkv_tokens. Already off to a bad start."<<std::endl;
	exit(EXIT_FAILURE);
	}
	if (!w1w3_tokens.is_open() ) {
	std::cout<<"w1w3_tokens. Already off to a bad start."<<std::endl;
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
	
	std::cout<<"test1"<<std::endl;
	std::cout<<"Opened all the files sucessfully"<<std::endl;

	/* ============================== constants related to tb ===================================== */
	int pos = 150;
	const int layer_cnt = 12;
	const int out_data_size = MODEL_ELEMENTS * 4;
	const int quant_data_size = MODEL_ELEMENTS * MODEL_ELEMENTS * layer_cnt;
	const int sf_data_size = quant_data_size * 4 / MODEL_SCALING_FACTOR;
	const int rms_w_size = MODEL_ELEMENTS * 4 * layer_cnt;
	const int tokens_size = MODEL_ELEMENTS * 4;
	const int tok_w1_size = MODEL_HIDDEN_DIM * 4;
	const int logits_size = MODEL_TOKENS * 4;
	const int logits_quant_size = MODEL_ELEMENTS * MODEL_TOKENS * 1;
	const int logits_sf_size = MODEL_ELEMENTS * MODEL_TOKENS / MODEL_SCALING_FACTOR * 4;
	
	// const int wo_size = 589824;
	const int cache_size = 1024*768*4 * 12;
	const int t_size = 768 * 4;
	const int xb2_size = 3072;
	const int hd_tok_size = MODEL_ELEMENTS * MODEL_HIDDEN_DIM * layer_cnt;
	const int hd_sf_size = hd_tok_size / MODEL_SCALING_FACTOR * 4;
	
	const int out_data_cnt = out_data_size / sizeof(fdata_v_t);
	const int quant_data_cnt = quant_data_size / sizeof(idata_v_t);
	const int sf_data_cnt = sf_data_size / sizeof(fdata_v_t);
	const int rms_w_cnt = rms_w_size / sizeof(fdata_v_t);
	const int tokens_cnt = tokens_size / sizeof(fdata_v_t);
	const int tok_w1_cnt = tok_w1_size / sizeof(fdata_v_t);
	const int logits_cnt = logits_size / sizeof(fdata_v_t);
	const int logits_q_cnt = logits_quant_size / sizeof(idata_v_t);
	const int logits_sf_cnt = logits_sf_size / sizeof(fdata_v_t);
	
	const int cache_cnt = cache_size / sizeof(fdata_v_t);
	const int tok_cnt = t_size / sizeof(fdata_v_t);
	const int xb2_cnt = xb2_size / sizeof(fdata_v_t);
	const int hd_tok_cnt = hd_tok_size / sizeof(idata_v_t);
	const int hd_sf_cnt = hd_sf_size / sizeof(fdata_v_t);

	

/* ===================================== declare our vectors ===================================== */

std::vector<fdata_v_t> dummy_tok(MODEL_HIDDEN_DIM * 4 / sizeof(fdata_v_t));
std::vector<idata_v_t> dummy_w(MODEL_HIDDEN_DIM * MODEL_ELEMENTS * 12 / sizeof(idata_v_t));
std::vector<fdata_v_t> dummy_sf((MODEL_HIDDEN_DIM * MODEL_ELEMENTS / MODEL_SCALING_FACTOR) * 4 * 12 / sizeof(fdata_v_t));
	std::vector<std::vector<fdata_v_t>> dummy_out(2, std::vector<fdata_v_t>(MODEL_HIDDEN_DIM * 4 / sizeof(fdata_v_t)));

	std::fill(dummy_tok.begin(), dummy_tok.end(), 0.5f);
	std::fill(dummy_w.begin(), dummy_w.end(), 1);
	std::fill(dummy_sf.begin(), dummy_sf.end(), .750f);
	std::fill(dummy_out[0].begin(), dummy_out[0].end(), 768.0f);

	std::vector<fdata_v_t> tokens_arr(tokens_cnt);
	std::vector<fdata_v_t> golden_arr(tokens_cnt);
	std::vector<fdata_v_t> actual_arr(tokens_cnt);
	std::vector<fdata_v_t> tok_w1_arr(tokens_cnt);
	std::vector<fdata_v_t> tok_w2_arr(tok_w1_cnt);
	std::vector<fdata_v_t> rms_final_arr(rms_w_cnt);

	std::vector<idata_v_t> wq_q_arr(quant_data_cnt);
	std::vector<fdata_v_t> wq_sf_arr(sf_data_cnt);

	std::vector<idata_v_t> wcls_q_arr(logits_q_cnt);
	std::vector<fdata_v_t> wcls_sf_arr(logits_sf_cnt);

	std::vector<idata_v_t> wk_q_arr(quant_data_cnt);
	std::vector<fdata_v_t> wk_sf_arr(sf_data_cnt);

	std::vector<idata_v_t> wv_q_arr(quant_data_cnt);
	std::vector<fdata_v_t> wv_sf_arr(sf_data_cnt);

	std::vector<idata_v_t> wo_q_arr(quant_data_cnt);
	std::vector<fdata_v_t> wo_sf_arr(sf_data_cnt);

	std::vector<fdata_v_t> key_in_arr(cache_cnt);
	std::vector<fdata_v_t> value_in_arr(cache_cnt);

	std::vector<idata_v_t> w1_q_arr(hd_tok_cnt);
	std::vector<fdata_v_t> w1_sf_arr(hd_sf_cnt);

	std::vector<idata_v_t> w3_q_arr(hd_tok_cnt);
	std::vector<fdata_v_t> w3_sf_arr(hd_sf_cnt);

	std::vector<idata_v_t> w2_q_arr(hd_tok_cnt);
	std::vector<fdata_v_t> w2_sf_arr(hd_sf_cnt);

	std::vector<std::vector<fdata_v_t>> key_arr(2, std::vector<fdata_v_t>(cache_cnt));
	std::vector<std::vector<fdata_v_t>> value_arr(2, std::vector<fdata_v_t>(cache_cnt));
	std::vector<std::vector<fdata_v_t>> tok_out_arr(2, std::vector<fdata_v_t>(tokens_cnt));
	std::vector<std::vector<fdata_v_t>> tok_w1_out_arr(2, std::vector<fdata_v_t>(tok_w1_cnt));
	std::vector<std::vector<fdata_v_t>> tok_w2_out_arr(2, std::vector<fdata_v_t>(tokens_cnt));
	std::vector<std::vector<fdata_v_t>> log_out_arr(2, std::vector<fdata_v_t>(logits_cnt));
	std::vector<std::vector<my_float_t>> att_score_arr(2, std::vector<my_float_t>(MODEL_ELEMENTS));

	/* ================================== read data into array =================================== */

	qkv_tokens.read(reinterpret_cast<char *>(tokens_arr.data()), tokens_size);
	key_output.read(reinterpret_cast<char *>(tok_out_arr[0].data()), tokens_size);
	w1w3_tokens.read(reinterpret_cast<char *>(tok_w1_arr.data()), tokens_size);
	w2_tokens.read(reinterpret_cast<char *>(tok_w2_arr.data()), tok_w1_size);
	w1_output.read(reinterpret_cast<char *>(tok_w1_out_arr[0].data()), tok_w1_size);
	w2_output.read(reinterpret_cast<char *>(golden_arr.data()), tokens_size);
	
	wq_q_dat.read(reinterpret_cast<char *>(wq_q_arr.data()), quant_data_size);
	wq_sf_dat.read(reinterpret_cast<char *>(wq_sf_arr.data()), sf_data_size);

	wk_q_dat.read(reinterpret_cast<char *>(wk_q_arr.data()), quant_data_size);
	wk_sf_dat.read(reinterpret_cast<char *>(wk_sf_arr.data()), sf_data_size);

	wv_q_dat.read(reinterpret_cast<char *>(wv_q_arr.data()), quant_data_size);
	wv_sf_dat.read(reinterpret_cast<char *>(wv_sf_arr.data()), sf_data_size);

	w1_q_dat.read(reinterpret_cast<char *>(w1_q_arr.data()), hd_tok_size);
	w1_sf_dat.read(reinterpret_cast<char *>(w1_sf_arr.data()), hd_sf_size);
	w2_q_dat.read(reinterpret_cast<char *>(w2_q_arr.data()), hd_tok_size);
	w2_sf_dat.read(reinterpret_cast<char *>(w2_sf_arr.data()), hd_sf_size);

	w3_q_dat.read(reinterpret_cast<char *>(w3_q_arr.data()), hd_tok_size);
	w3_sf_dat.read(reinterpret_cast<char *>(w3_sf_arr.data()), hd_sf_size);
	
	std::cout<<"Loaded the files into memory"<<std::endl;


/* ===================================== Declare the streams ========================================= */
/* remember - if it's suppsoed to be m_axi, no need to create a stream. just use the created vector, ie: foo_arr.data() */

	/* ============================ write inputs to the streams ====================== */
		

int curr_pos = 150;
	std::cout<<"Delcared and Loaded the Streams"<<std::endl;

	/* ============================ Call the function(s) ====================================== */

	// matmult_kernel(tok_out_arr[1].data(), tokens_arr.data(), wk_sf_arr.data(), wk_q_arr.data(), MODEL_ELEMENTS, MODEL_ELEMENTS, 0);
	std::cout<<"first one done"<<std::endl;

	// matmult_kernel(tok_w1_out_arr[1].data(), tok_w1_arr.data(), w1_sf_arr.data(), w1_q_arr.data(), MODEL_ELEMENTS, MODEL_HIDDEN_DIM, 0, 0);
	// matmult_kernel(tok_w1_out_arr[1].data(), tok_w1_arr.data(), w1_sf_arr.data(), w1_q_arr.data(), MODEL_ELEMENTS, MODEL_HIDDEN_DIM, 0);
	double_matmult_kernel(actual_arr.data(), tok_w2_arr.data(), w2_sf_arr.data(), w2_q_arr.data(), MODEL_HIDDEN_DIM, MODEL_ELEMENTS, 0, 0);
	// matmult_kernel(tok_w2_out_arr[1].data(), tok_w2_arr.data(), w2_sf_arr.data(), w2_q_arr.data(), MODEL_HIDDEN_DIM, MODEL_ELEMENTS, 0);
	
	/*  ====================================== process the results ============================ */
	// std::cout<< "========================= Tokens output array data ========================"<<std::endl;
	// parse_results<fdata_v_t, float>(tok_out_arr[0], tok_out_arr[1]);
	std::cout<< "========================= Tokens output array data ========================"<<std::endl;
	// parse_results<fdata_v_t, float>(dummy_out[0], dummy_out[1]);
	parse_results<fdata_v_t, float>(golden_arr, actual_arr);

	return 0;

}