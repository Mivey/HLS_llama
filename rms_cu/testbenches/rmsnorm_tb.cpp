#include "../forward.h"
// #include "../matmul.h"
#include "../rmsnorm.h"

#include "tb_main.h"

// int mat_mul_tb(){
int rmsnorm_tb(){
	std::cout<<"starting testbench"<<std::endl;
	/* ===================== open neede files =========================== */

	/* ========== input files ============================= */
	std::ifstream tokens_dat("top_data/P_25_tokens.bin", std::ios::binary);
	std::ifstream rms_att_w_dat("top_data/P_25_rms_att_weight.bin", std::ios::binary);

	/* =========================== output data ===================== */
	std::ifstream rms_out_dat("top_data/P_25_rms_outputt.bin", std::ios::binary);

/* =================== check if files opened successfully */
	if (!tokens_dat.is_open() ) {
	std::cout<<"tokens_dat. Already off to a bad start. boop."<<std::endl;
	exit(EXIT_FAILURE);
	}
	if (!rms_att_w_dat.is_open() ) {
	std::cout<<"rms_att_w_dat. Already off to a bad start. boop."<<std::endl;
	exit(EXIT_FAILURE);
	}
	if (!rms_out_dat.is_open() ) {
	std::cout<<"rms_out_dat. Already off to a bad start."<<std::endl;
	exit(EXIT_FAILURE);
	}
	

	/* ============================== constants related to tb ===================================== */
	const int pos = 150;
	const int sf_el = MODEL_ELEMENTS / 64;
	const int w_sf_size = 36864;
	const int w_size = 589824;
	const int t_sf_size = 48;
	const int t_size = 768;
	const int out_size = 3072;

	const int w_cnt = w_size / sizeof(idata_v_t);
	const int w_sf_cnt = w_sf_size / sizeof(fdata_v_t);
	const int tok_sf_cnt = t_sf_size / sizeof(fdata_v_t);
	const int tok_cnt = t_size / sizeof(idata_v_t);
	const int out_cnt = out_size / sizeof(fdata_v_t);		// (768 elements  * 4 bytes/elements) / (4 elements/ point * 4 bytes/elemnts) = 192 points

/* ===================================== declare our vectors ===================================== */
	std::vector<fdata_v_t> token_arr(out_cnt);
	std::vector<fdata_v_t> rms_att_arr(out_cnt);
	
	std::vector<std::vector<fdata_v_t>> rms_out_arr(2, std::vector<fdata_v_t>(out_cnt));

	/* ================================== read data into array =================================== */

	tokens_dat.read(reinterpret_cast<char *>(token_arr.data()), out_size);
	rms_att_w_dat.read(reinterpret_cast<char *>(rms_att_arr.data()), out_size);

	rms_out_dat.read(reinterpret_cast<char *>(rms_out_arr[0].data()), out_size);


/* ===================================== Declare the streams ========================================= */
/* remember - if it's suppsoed to be m_axi, no need to create a stream. just use the created vector, ie: foo_arr.data() */

	hls::stream<fdata_v_t> token_in("Tokens data input");
	hls::stream<fdata_v_t> rms_att_w_in("RMS Weights data input");

	hls::stream<fdata_v_t> rms_att_out("RMS output Stream");

	/* ============================ write inputs to the streams ====================== */
	
	
	quant_data_loop:
	for (int i = 0; i < out_cnt; i++) {
		token_in.write(token_arr[i]);
	}
	for (int i = 0; i < out_cnt; i++) {
		rms_att_w_in.write(rms_att_arr[i]);
	}


	/* ============================ Call the function(s) ====================================== */

    // rmsnorm_kernel<MODEL_ELEMENTS>(rms_att_out, token_in, rms_att_w_in);
		rms_top(rms_att_out, token_in, rms_att_w_in);

	/* =============================== get all the data =================================== */

	for (int i = 0; i < out_cnt; i++) {
		rms_out_arr[1][i] = rms_att_out.read();
	}


	/*  ====================================== process the results ============================ */

	std::cout<< "========================= rms norm array data ========================"<<std::endl;
	parse_results<fdata_v_t, float>(rms_out_arr[0], rms_out_arr[1]);
	return 0;

}