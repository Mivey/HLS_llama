#include "tb_main.h"

// int mat_mul_tb(){
int rope_tb(){
	std::cout<<"starting testbench"<<std::endl;
	/* ===================== open neede files =========================== */

	/* ========== input files ============================= */
	std::ifstream query_dat("top_data/P_25_out_query.bin", std::ios::binary);

	std::ifstream key_dat("top_data/P_25_out_key.bin", std::ios::binary);

	/* =========================== output data ===================== */
	std::ifstream query_rope_out("top_data/P_25_rope_out_query.bin", std::ios::binary);
	std::ifstream key_rope_out("top_data/P_25_rope_out_key.bin", std::ios::binary);

/* =================== check if files opened successfully */
	if (!query_dat.is_open() ) {
	std::cout<<"query_dat. Already off to a bad start. boop."<<std::endl;
	exit(EXIT_FAILURE);
	}
	if (!key_dat.is_open() ) {
	std::cout<<"key_dat. Already off to a bad start."<<std::endl;
	exit(EXIT_FAILURE);
	}
	
	if (!query_rope_out.is_open() ) {
	std::cout<<"query_rope_out. Already off to a bad start."<<std::endl;
	exit(EXIT_FAILURE);
	}
	if (!key_rope_out.is_open() ) {
	std::cout<<"key_rope_out. Already off to a bad start."<<std::endl;
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
	const int out_cnt = out_size / sizeof(mfdata_v_t);		// (768 elements  * 4 bytes/elements) / (4 elements/ point * 4 bytes/elemnts) = 192 points

/* ===================================== declare our vectors ===================================== */
	std::vector<mfdata_v_t> query_arr(out_cnt);
	std::vector<mfdata_v_t> key_arr(out_cnt);
	
	std::vector<std::vector<mfdata_v_t>> q_rope_arr(2, std::vector<mfdata_v_t>(out_cnt));
	std::vector<std::vector<mfdata_v_t>> k_rope_arr(2, std::vector<mfdata_v_t>(out_cnt));

	/* ================================== read data into array =================================== */

	query_dat.read(reinterpret_cast<char *>(query_arr.data()), out_size);
	key_dat.read(reinterpret_cast<char *>(key_arr.data()), out_size);

	query_rope_out.read(reinterpret_cast<char *>(q_rope_arr[0].data()), out_size);
	key_rope_out.read(reinterpret_cast<char *>(k_rope_arr[0].data()), out_size);


/* ===================================== Declare the streams ========================================= */
/* remember - if it's suppsoed to be m_axi, no need to create a stream. just use the created vector, ie: foo_arr.data() */

	hls::stream<mfdata_v_t> query_in("Query data input");
	hls::stream<mfdata_v_t> key_in("Key data input");

	hls::stream<mfdata_v_t> qr_out("Query rope output Stream");
	hls::stream<mfdata_v_t> kr_out("Key rope output Stream");

	/* ============================ write inputs to the streams ====================== */
	
	
	quant_data_loop:
	for (int i = 0; i < out_cnt; i++) {
		query_in.write(query_arr[i]);
	}
	// quant_data_loop:
	for (int i = 0; i < out_cnt; i++) {
		key_in.write(key_arr[i]);
	}


	/* ============================ Call the function(s) ====================================== */

	// rope_kernel<ELEMENTS, ELEMENTS, SCALING_FACTOR, 150>(qr_out, query_in);
	rope_kernel (qr_out, query_in, MODEL_ELEMENTS, 150);

	/* =============================== get all the data =================================== */

	for (int i = 0; i < out_cnt; i++) {
		q_rope_arr[1][i] = qr_out.read();
	}
	// for (int i = 0; i < out_cnt; i++) {
	// 	k_rope_arr[1][i] = kr_out.read();
	// }


	/*  ====================================== process the results ============================ */

	std::cout<< "========================= query rope array data ========================"<<std::endl;
	parse_results<mfdata_v_t, float>(q_rope_arr[0], q_rope_arr[1]);
	std::cout<< "========================= key rope array data ========================"<<std::endl;
	// parse_results<fdata_v_t, float>(k_rope_arr[0], k_rope_arr[1]);

	return 0;

}