// #include "../forward.h"

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
	std::ifstream input_tokens_dat("seed_42069/150_input_tokens.bin", std::ios::binary);

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

	std::ifstream w2_sf_dat("weights/MLP_2_sf.bin", std::ios::binary);
	std::ifstream w2_q_dat("weights/MLP_2_q.bin", std::ios::binary);
	std::ifstream mha_tok_dat("seed_42069/Post_MHA_tokens.bin", std::ios::binary);
	/* =========================== output data ===================== */

	std::ifstream out_value_dat("seed_42069_conv/150_output_value_cache_head_maj.bin", std::ios::binary);
	std::ifstream out_key_dat("seed_42069_conv/150_output_key_cache_head_maj.bin", std::ios::binary);
	
	std::ifstream key_cache_dat("seed_42069/150_output_k_tokens.bin", std::ios::binary);
	std::ifstream value_cache_dat("seed_42069/150_output_v_tokens.bin", std::ios::binary);
	std::ifstream query_input("seed_42069/150_output_q_tokens.bin", std::ios::binary);
	std::ifstream w1_output("seed_42069/150_output_w1_tokens.bin", std::ios::binary);
	std::ifstream qkv_tokens("seed_42069/qkv_tokens.bin", std::ios::binary);
	std::ifstream w1w3_tokens("seed_42069/w1w3_tokens.bin", std::ios::binary);

	// std::ifstream logits_output("seed_42069/150_logits_output.bin", std::ios::binary);


/* =================== check if files opened successfully */
	if (!mha_tok_dat.is_open() ) {
	std::cout<<"mha_tok_dat. Already off to a bad start. boop."<<std::endl;
	exit(EXIT_FAILURE);
	}	if (!input_tokens_dat.is_open() ) {
	std::cout<<"input_tokens_dat. Already off to a bad start. boop."<<std::endl;
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
	if (!query_input.is_open() ) {
	std::cout<<"query_input. Already off to a bad start."<<std::endl;
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
	
	if (!out_value_dat.is_open() ) {
	std::cout<<"out_value_dat. Already off to a bad start."<<std::endl;
	exit(EXIT_FAILURE);
	}
	if (!out_key_dat.is_open() ) {
	std::cout<<"out_key_dat. Already off to a bad start."<<std::endl;
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
	// const int sf_el = MODEL_ELEMENTS / 64;
	// const int wo_sf_size = 36864;
	
	// const int wo_size = 589824;
	const int cache_size = 1024*768*4 * 12;
	const int t_size = 768 * 4;
	const int xb2_size = 3072;
	const int hd_tok_size = MODEL_ELEMENTS * MODEL_HIDDEN_DIM * layer_cnt;
	const int hd_sf_size = hd_tok_size / MODEL_SCALING_FACTOR * 4;
	// const int rms_tok_size = MODEL_ELEMENTS * 4 * layer_cnt;
	
	const int out_data_cnt = out_data_size / sizeof(mfdata_v_t);
	const int quant_data_cnt = quant_data_size / sizeof(idata_v_t);
	const int sf_data_cnt = sf_data_size / sizeof(fdata_v_t);
	const int rms_w_cnt = rms_w_size / sizeof(mfdata_v_t);
	const int tokens_cnt = tokens_size / sizeof(mfdata_v_t);
	const int tok_w1_cnt = tok_w1_size / sizeof(mfdata_v_t);
	const int logits_cnt = logits_size / sizeof(mfdata_v_t);
	const int logits_q_cnt = logits_quant_size / sizeof(idata_v_t);
	const int logits_sf_cnt = logits_sf_size / sizeof(fdata_v_t);
	
	// const int wo_cnt = wo_size / sizeof(idata_v_t);
	// const int wo_sf_cnt = wo_sf_size / sizeof(fdata_v_t);
	const int cache_cnt = cache_size / sizeof(mfdata_v_t);
	const int tok_cnt = t_size / sizeof(mfdata_v_t);
	const int xb2_cnt = xb2_size / sizeof(mfdata_v_t);
	const int hd_tok_cnt = hd_tok_size / sizeof(idata_v_t);
	const int hd_sf_cnt = hd_sf_size / sizeof(fdata_v_t);
	// const int rms_tok_cnt = rms_tok_size / sizeof(mfdata_v_t);

	

/* ===================================== declare our vectors ===================================== */

	std::vector<mfdata_v_t> tokens_arr(tokens_cnt);
	// std::vector<mfdata_v_t> val_in_rope_arr(tokens_cnt);
	// std::vector<mfdata_v_t> key_in_rope_arr(tokens_cnt);

	std::vector<mfdata_v_t> rms_w_arr(rms_w_cnt);

	std::vector<mfdata_v_t> rms_final_arr(rms_w_cnt);

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

	std::vector<mfdata_v_t> key_in_arr(tokens_cnt * 3);
	std::vector<mfdata_v_t> value_in_arr(tokens_cnt * 3);
	std::vector<mfdata_v_t> key_arr_a(cache_cnt);
	std::vector<mfdata_v_t> value_arr_a(cache_cnt);

	std::vector<mfdata_v_t> rms_ffn_arr(rms_w_cnt);
	std::vector<mfdata_v_t> mha_tok_gold(tokens_cnt);
	std::vector<mfdata_v_t> mha_tok_act(tokens_cnt);

	std::vector<idata_v_t> w1_q_arr(hd_tok_cnt);
	std::vector<fdata_v_t> w1_sf_arr(hd_sf_cnt);

	std::vector<idata_v_t> w3_q_arr(hd_tok_cnt);
	std::vector<fdata_v_t> w3_sf_arr(hd_sf_cnt);

	std::vector<idata_v_t> w2_q_arr(hd_tok_cnt);
	std::vector<fdata_v_t> w2_sf_arr(hd_sf_cnt);

	// std::vector<std::vector<fdata_v_t>> query_arr(2, std::vector<fdata_v_t>(out_data_cnt));
	std::vector<std::vector<mfdata_v_t>> key_arr(2, std::vector<mfdata_v_t>(cache_cnt));
	std::vector<std::vector<mfdata_v_t>> value_arr(2, std::vector<mfdata_v_t>(cache_cnt));
	std::vector<std::vector<mfdata_v_t>> tok_out_arr(2, std::vector<mfdata_v_t>(tokens_cnt));
	std::vector<std::vector<mfdata_v_t>> tok_w1_out_arr(2, std::vector<mfdata_v_t>(tok_w1_cnt));
	std::vector<std::vector<mfdata_v_t>> log_out_arr(2, std::vector<mfdata_v_t>(logits_cnt));
	std::vector<std::vector<my_float_t>> att_score_arr(2, std::vector<my_float_t>(MODEL_ELEMENTS));

	/* ================================== read data into array =================================== */

	// input_tokens_dat.read(reinterpret_cast<char *>(tokens_arr.data()), tokens_size);
// 	key_output
// w1_output
// qkv_tokens
// w1w3_tokens		mha_tok_dat
	query_input.read(reinterpret_cast<char *>(tokens_arr.data()), tokens_size);
	// value_input.read(reinterpret_cast<char *>(val_in_rope_arr.data()), tokens_size);
	// key_input.read(reinterpret_cast<char *>(key_in_rope_arr.data()), tokens_size);
	// key_output.read(reinterpret_cast<char *>(tok_out_arr[0].data()), tokens_size);
	// w1w3_tokens.read(reinterpret_cast<char *>(tok_w1_arr.data()), tokens_size);
	// w1_output.read(reinterpret_cast<char *>(tok_w1_out_arr[0].data()), tok_w1_size);
	
	wq_q_dat.read(reinterpret_cast<char *>(wq_q_arr.data()), quant_data_size);
	wq_sf_dat.read(reinterpret_cast<char *>(wq_sf_arr.data()), sf_data_size);

	wk_q_dat.read(reinterpret_cast<char *>(wk_q_arr.data()), quant_data_size);
	wk_sf_dat.read(reinterpret_cast<char *>(wk_sf_arr.data()), sf_data_size);

	wv_q_dat.read(reinterpret_cast<char *>(wv_q_arr.data()), quant_data_size);
	wv_sf_dat.read(reinterpret_cast<char *>(wv_sf_arr.data()), sf_data_size);

	w1_q_dat.read(reinterpret_cast<char *>(w1_q_arr.data()), hd_tok_size);
	w1_sf_dat.read(reinterpret_cast<char *>(w1_sf_arr.data()), hd_sf_size);
	mha_tok_dat.read(reinterpret_cast<char *>(mha_tok_gold.data()), tokens_size);

	w3_q_dat.read(reinterpret_cast<char *>(w3_q_arr.data()), hd_tok_size);
	w3_sf_dat.read(reinterpret_cast<char *>(w3_sf_arr.data()), hd_sf_size);

	w2_q_dat.read(reinterpret_cast<char *>(w2_q_arr.data()), hd_tok_size);
	w2_sf_dat.read(reinterpret_cast<char *>(w2_sf_arr.data()), hd_sf_size);
	

	key_cache_dat.read(reinterpret_cast<char *>(key_in_arr.data()), tokens_size);
	value_cache_dat.read(reinterpret_cast<char *>(value_in_arr.data()), tokens_size);

	out_key_dat.read(reinterpret_cast<char *>(key_arr[0].data()), cache_size);
	out_value_dat.read(reinterpret_cast<char *>(value_arr[0].data()), cache_size);
	// memcpy(key_arr[0].data(), key_arr_a.data(), cache_size);
	// memcpy(value_arr[0].data(), value_arr_a.data(), cache_size);
	memcpy(key_arr_a.data(), key_arr[0].data(), cache_size);
	memcpy(value_arr_a.data(), value_arr[0].data(), cache_size);
	
	std::cout<<"Loaded the files into memory"<<std::endl;


/* ===================================== Declare the streams ========================================= */
/* remember - if it's suppsoed to be m_axi, no need to create a stream. just use the created vector, ie: foo_arr.data() */

	/* ============================ write inputs to the streams ====================== */
		

int curr_pos = 150;
	std::cout<<"Delcared and Loaded the Streams"<<std::endl;

	/* ============================ Call the function(s) ====================================== */

	// void matmult_kernel(mfdata_v_t *out, mfdata_v_t *fl_tok, fdata_v_t *w_sf, idata_v_t *w, const int N_DIM, const int M_DIM, const int CURR_LAYER)

	// matmult_kernel(tok_out_arr[1].data(), tokens_arr.data(), wk_sf_arr.data(), wk_q_arr.data(), MODEL_ELEMENTS, MODEL_ELEMENTS, 0);
	// qkv_tokens.read(reinterpret_cast<char *>(tokens_arr.data()), tokens_size);
	// key_output.read(reinterpret_cast<char *>(tok_out_arr[0].data()), tokens_size);
// 	tok_w1_arr
// tok_w1_out_arr
// mha_kernel(tokens_arr.data(), mfdata_v_t *key_cache, mfdata_v_t *value_cache, mfdata_v_t *key_cache_in, mfdata_v_t *value_cache_in, const int POS, const int CURR_LAYER)
	mha_kernel(tokens_arr.data(), key_arr_a.data(), value_arr_a.data(), key_in_arr.data(), value_in_arr.data(), curr_pos, 0);
/*======= get all the data =================================== */

	// for (int i = 0; i < MODEL_ELEMENTS; i++) {
	// 	// float foo = att_score_arr[i].data();
	// 	std::cout<< "golden: "<<att_score_arr[0][i]<<std::endl;
	// }

	/*  ====================================== process the results ============================ */
	// std::cout<<"token_arr size: "<<log_out_arr[1].size()<<std::endl;
	// std::cout<<"key size: "<<key_in_arr.size()<<std::endl;
	// std::cout<<"value size: "<<value_in_arr.size()<<std::endl;
	std::cout<< "========================= Tokens output array data ========================"<<std::endl;
	parse_results<mfdata_v_t, float>(mha_tok_gold, tokens_arr);
	// std::cout<< "========================= Tokens output array data ========================"<<std::endl;
	// parse_results<mfdata_v_t, float>(tok_w1_out_arr[0], tok_w1_out_arr[1]);

	std::cout<< "========================= Value cache array data ========================"<<std::endl;
	parse_cache_results<mfdata_v_t, float>(value_arr[0], value_arr_a);

	std::cout<< "========================= Key cache array data ========================"<<std::endl;
	parse_cache_results<mfdata_v_t, float>(key_arr[0], key_arr_a);

	return 0;

}