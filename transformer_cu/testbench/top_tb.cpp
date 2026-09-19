
// #include "../forward.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
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

/* ============================================================================
 * WIDTH-GENERIC DEVICE LAYOUT
 *
 * Everything below derives from MAX_DW via mha_forward.h. Nothing here assumes
 * 512 bits. The layout the kernel expects, for any MAX_DW:
 *
 *   one beat            = sizeof(wide_t)            = MAX_DW/8 bytes
 *   one frame           = SF_FRAME beats of scales, then QUANT_FRAME beats of int8
 *   scales in a frame   = SF_FRAME    * MAX_FL_ELEM
 *   weights in a frame  = QUANT_FRAME * MAX_QUANT_ELEM
 *
 * The two must satisfy  scales * MODEL_SCALING_FACTOR == weights, which holds
 * for every MAX_DW because both sides come out to 24*MAX_DW. That invariant is
 * the whole reason SF_FRAME=12 / QUANT_FRAME=192 is width-independent, so it is
 * asserted rather than assumed.
 *
 * The OLD testbench encoded two accidents of MAX_DW=512:
 *   FRAME_SF_BYTES = QUANT_FRAME * sizeof(my_float_t)   -- only == SF_FRAME*beat at 512
 *   FRAME_W_BYTES  = QUANT_FRAME * MODEL_SCALING_FACTOR -- uses 64 as "bytes per beat"
 * Both are now derived from sizeof(wide_t).
 * ========================================================================== */

/* the KV cache element type, as transformer_cu actually declares its ports.
 * If/when MID_DW becomes independently plumbed, change this one alias. */
using cache_v_t = mfdata_v_t;

static constexpr size_t BEAT_BYTES      = sizeof(wide_t);                    // MAX_DW/8
static constexpr size_t SCALES_PER_BEAT = MAX_FL_ELEM;                       // MAX_DW/32
static constexpr size_t Q_PER_BEAT      = MAX_QUANT_ELEM;                    // MAX_DW/8

static constexpr size_t FRAME_SF_BYTES  = SF_FRAME    * BEAT_BYTES;
static constexpr size_t FRAME_W_BYTES   = QUANT_FRAME * BEAT_BYTES;
static constexpr size_t FRAME_BYTES     = FRAME_SF_BYTES + FRAME_W_BYTES;    // TXFR_FRAME beats
static constexpr size_t FRAME_SCALES    = SF_FRAME    * SCALES_PER_BEAT;
static constexpr size_t FRAME_ELEMS     = QUANT_FRAME * Q_PER_BEAT;

static constexpr int    NPORT           = 2;                                 // w_0 / w_1

/* ---- structural invariants; these must hold or the frames are wrong ---- */
static_assert(sizeof(my_float_t) == 4, "device side assumes 32-bit float");
static_assert(MAX_DW >= 128 && (MAX_DW & (MAX_DW - 1)) == 0,
              "MAX_DW must be a power of two, at least 128 (SM_DW)");
static_assert(BEAT_BYTES == MAX_DW / 8,       "wide_t must be exactly one AXI beat");
static_assert(sizeof(idata_v_t)  == BEAT_BYTES, "idata_v_t must be one beat");
static_assert(sizeof(mfdata_v_t) == BEAT_BYTES, "mfdata_v_t must be one beat");
static_assert(sizeof(fdata_v_t)  == SM_DW / 8,  "fdata_v_t must be one SM_DW beat");
static_assert(FRAME_BYTES == TXFR_FRAME * BEAT_BYTES, "frame must be TXFR_FRAME beats");
static_assert(FRAME_SCALES * MODEL_SCALING_FACTOR == FRAME_ELEMS,
              "a frame's scale factors must cover exactly that frame's weights");
static_assert(MODEL_HEAD_SIZE % MAX_FL_ELEM == 0,
              "head size must be a whole number of cache beats");
static_assert(QUANT_MODIFIER == 1,
              "to_idvt() unpacks the full beat; QUANT_MODIFIER != 1 silently drops data");

struct axi_reg_t {
	int POS;
	int N_DIM;
	int M_DIM;
	int rms_att_W;
	int rms_ffn_W;
	int rms_final_W;
};

/* ---------------------------------------------------------------------------
 * Walk a run of `n` consecutively-stored quantized tensors, recording the file
 * offset of each tensor's int8 block and of its scale-factor block.
 *
 * llama2.c runq.c init_quantized_tensors() stores, per tensor:
 *     [ n_elem int8 ][ n_elem/GS float ]
 * ------------------------------------------------------------------------- */
static void scan_tensors(std::ifstream &f, size_t n_elem,
                         size_t *w_ptr, size_t *sf_ptr, int n)
{
	const size_t w_bytes  = n_elem * sizeof(int8_t);
	const size_t sf_bytes = (n_elem / MODEL_SCALING_FACTOR) * sizeof(my_float_t);

	for (int i = 0; i < n; i++) {
		w_ptr[i] = static_cast<size_t>(f.tellg());
		f.seekg(w_bytes, std::ios::cur);          // relative!
		sf_ptr[i] = static_cast<size_t>(f.tellg());
		f.seekg(sf_bytes, std::ios::cur);         // relative!
		if (!f) { std::cerr << "scan_tensors: ran off the end of the checkpoint\n"; exit(EXIT_FAILURE); }
	}
}

struct TensorRef { size_t w_base, sf_base, n_elem; };

/* read exactly n bytes at absolute offset off, or die */
static void read_at(std::ifstream &f, size_t off, char *dst, size_t n, const char *what)
{
	f.seekg(static_cast<std::streamoff>(off), std::ios::beg);
	f.read(dst, static_cast<std::streamsize>(n));
	if (f.gcount() != static_cast<std::streamsize>(n)) {
		std::cerr << "short read on " << what << ": wanted " << n
		          << " B at " << off << ", got " << f.gcount() << std::endl;
		exit(EXIT_FAILURE);
	}
}

/* ---------------------------------------------------------------------------
 * Emit one GeMV's weights into the interleaved device layout.
 *
 * The tensors in `ts` are concatenated into one logical matmul, chopped into
 * FRAME_ELEMS-sized frames, split into NPORT contiguous row-halves, then
 * interleaved frame-by-frame so mm2ds_input_data's PORT/NPORT stride hands each
 * port a contiguous run of output rows.
 *
 * Emission order:  f0(port0) f0(port1) f1(port0) f1(port1) ...
 * where fj(portk) is frame (k*per_port + j) of the concatenated tensor.
 *
 * Width-generic: the only things that change with MAX_DW are FRAME_ELEMS,
 * FRAME_SCALES and the two byte counts, all derived from sizeof(wide_t).
 * Narrower beats simply mean more, smaller frames covering the same bytes.
 * ------------------------------------------------------------------------- */
static void pack_gemv(std::ifstream &f, char *dst, size_t &idx,
                      const TensorRef *ts, int n_ts, const char *label)
{
	/* total element count across the concatenated tensors */
	size_t tot_elem = 0;
	for (int t = 0; t < n_ts; t++) {
		if (ts[t].n_elem % FRAME_ELEMS != 0) {
			std::cerr << "pack_gemv(" << label << "): tensor " << t << " has "
			          << ts[t].n_elem << " elements, not a multiple of FRAME_ELEMS="
			          << FRAME_ELEMS << " (MAX_DW=" << MAX_DW << ")" << std::endl;
			exit(EXIT_FAILURE);
		}
		tot_elem += ts[t].n_elem;
	}

	const size_t n_frame = tot_elem / FRAME_ELEMS;
	if (n_frame % NPORT != 0) {
		std::cerr << "pack_gemv(" << label << "): " << n_frame
		          << " frames does not divide across " << NPORT << " ports" << std::endl;
		exit(EXIT_FAILURE);
	}
	const size_t per_port = n_frame / NPORT;

	/* Stage the whole GeMV once instead of seeking twice per frame. At narrow
	 * widths there are 4x as many frames, so the old seek-per-frame pattern got
	 * expensive; this also makes the frame maths pure offset arithmetic.
	 * Worst case is the embedding table: ~24.5 MB int8 + ~1.5 MB scales. */
	std::vector<int8_t> wbuf(tot_elem);
	std::vector<my_float_t> sfbuf(tot_elem / MODEL_SCALING_FACTOR);

	size_t we = 0, se = 0;
	for (int t = 0; t < n_ts; t++) {
		const size_t nsf = ts[t].n_elem / MODEL_SCALING_FACTOR;
		read_at(f, ts[t].w_base,  reinterpret_cast<char*>(wbuf.data() + we),
		        ts[t].n_elem * sizeof(int8_t), "tensor weights");
		read_at(f, ts[t].sf_base, reinterpret_cast<char*>(sfbuf.data() + se),
		        nsf * sizeof(my_float_t), "tensor scale factors");
		we += ts[t].n_elem;
		se += nsf;
	}

	for (size_t j = 0; j < per_port; j++) {
		for (int k = 0; k < NPORT; k++) {
			const size_t fr = static_cast<size_t>(k) * per_port + j;

			std::memcpy(dst + idx, sfbuf.data() + fr * FRAME_SCALES, FRAME_SF_BYTES);
			idx += FRAME_SF_BYTES;

			std::memcpy(dst + idx, wbuf.data() + fr * FRAME_ELEMS, FRAME_W_BYTES);
			idx += FRAME_W_BYTES;
		}
	}
}

/* ---------------------------------------------------------------------------
 * Mirror weight_fsm()'s frame accounting so a layout mismatch shows up here,
 * with a readable message, instead of as a cosim deadlock.
 * ------------------------------------------------------------------------- */
static void verify_frame_plan(size_t packed_beats)
{
	struct Stage { int N, M; const char *name; };
	const Stage per_layer[] = {
		{MODEL_ELEMENTS, MODEL_ELEMENTS * 3,   "QKV"},
		{MODEL_ELEMENTS, MODEL_ELEMENTS,       "O"},
		{MODEL_ELEMENTS, MODEL_HIDDEN_DIM * 2, "gate+up"},
		{MODEL_HIDDEN_DIM, MODEL_ELEMENTS,     "down"},
	};
	const Stage final_stage = {MODEL_ELEMENTS, MODEL_TOKENS, "classifier"};

	size_t beats = 0;
	auto account = [&](const Stage &s) {
		const size_t bytes = (size_t)s.N * s.M * 17 / 16;
		if (bytes % (BEAT_BYTES * TXFR_FRAME * NPORT) != 0) {
			std::cerr << "verify_frame_plan: stage " << s.name << " (" << s.N << "x" << s.M
			          << ") is " << bytes << " B, not a multiple of "
			          << (BEAT_BYTES * TXFR_FRAME * NPORT)
			          << " B (" << NPORT << " frames at MAX_DW=" << MAX_DW << ")" << std::endl;
			exit(EXIT_FAILURE);
		}
		beats += bytes / BEAT_BYTES;
	};

	for (int l = 0; l < MODEL_NUM_LAYERS; l++)
		for (const auto &s : per_layer) account(s);
	account(final_stage);

	if (beats != packed_beats) {
		std::cerr << "verify_frame_plan: FSM will consume " << beats
		          << " beats but the packed buffer holds " << packed_beats << std::endl;
		exit(EXIT_FAILURE);
	}
}

static void dump_layout(size_t packed_size)
{
	std::cout << "---- device layout (MAX_DW=" << MAX_DW << ", SM_DW=" << SM_DW << ") ----\n"
	          << "  beat            " << BEAT_BYTES      << " B\n"
	          << "  scales/beat     " << SCALES_PER_BEAT << "\n"
	          << "  int8/beat       " << Q_PER_BEAT      << "\n"
	          << "  frame           " << FRAME_BYTES     << " B = " << TXFR_FRAME << " beats ("
	                                  << FRAME_SF_BYTES  << " B sf + " << FRAME_W_BYTES << " B w)\n"
	          << "  frame covers    " << FRAME_ELEMS     << " weights / " << FRAME_SCALES << " scales\n"
	          << "  packed buffer   " << packed_size     << " B = "
	                                  << packed_size / BEAT_BYTES << " beats, "
	                                  << packed_size / FRAME_BYTES << " frames\n"
	          << "  quantizer emits " << (MODEL_SCALING_FACTOR / (Q_PER_BEAT ? Q_PER_BEAT : 1))
	          << " idata_v_t per scale group\n"
	          << "-----------------------------------------------" << std::endl;
}

int top_tb(){
	std::cout<<"starting First Third testbench"<<std::endl;

	axi_reg_t axi_regs{};

/* ====== INPUT DATA AND CHECKS ============ INPUT DATA AND CHECKS ====== */
	std::string checkpoint = "weights/stories110M_q8.bin";
	std::ifstream file(checkpoint, std::ios::binary | std::ios::ate);
	std::ifstream coin_data("newgolden/150_coin.bin", std::ios::binary);
	std::ifstream token_data("newgolden/150_tokens.bin", std::ios::binary);
	std::ifstream data_output("newgolden/150_pre_quantized.bin", std::ios::binary);
	std::ifstream gemv_data_output("newgolden/150_post_matmul.bin", std::ios::binary);
	std::ifstream out_key_dat("newgolden/150_key_cache.bin", std::ios::binary);
	std::ifstream out_value_dat("newgolden/150_value_cache.bin", std::ios::binary);
	std::ifstream input_tokens("newgolden/150_01_rms_att_in.bin", std::ios::binary);

	struct { std::ifstream *s; const char *n; } fl[] = {
		{&file, "checkpoint"}, {&coin_data, "coin_data"}, {&token_data, "token_data"},
		{&data_output, "data_output"}, {&gemv_data_output, "gemv_data_output"},
		{&out_key_dat, "out_key_dat"}, {&out_value_dat, "out_value_dat"},
		{&input_tokens, "input_tokens"},
	};
	for (auto &e : fl) {
		if (!e.s->is_open()) {
			std::cerr << "Could not open " << e.n << ". Already off to a bad start." << std::endl;
			exit(EXIT_FAILURE);
		}
	}

/* ===== MEMORY VECTOR ARRAY CONSTRUCTOR ===== */

	const size_t rms_att_size   = MODEL_ELEMENTS * MODEL_NUM_LAYERS * sizeof(my_float_t);
	const size_t rms_ffn_size   = rms_att_size;
	const size_t rms_final_size = MODEL_ELEMENTS * sizeof(my_float_t);

	/* element counts, not byte counts */
	const size_t nn_elem    = (size_t)MODEL_ELEMENTS * MODEL_ELEMENTS;      // wq/wk/wv/wo
	const size_t nm_elem    = (size_t)MODEL_ELEMENTS * MODEL_HIDDEN_DIM;    // w1/w2/w3
	const size_t embed_elem = (size_t)MODEL_ELEMENTS * MODEL_TOKENS;        // q_tokens

	const size_t embed_size    = embed_elem * sizeof(int8_t);
	const size_t embed_sf_cnt  = embed_elem / MODEL_SCALING_FACTOR;
	const size_t embed_sf_size = embed_sf_cnt * sizeof(my_float_t);

	const size_t data_out_size      = ((MODEL_ELEMENTS * 3 + MODEL_HIDDEN_DIM) * MODEL_NUM_LAYERS + MODEL_ELEMENTS) * sizeof(my_float_t);
	const size_t gemv_data_out_size = ((MODEL_ELEMENTS * 5 + MODEL_HIDDEN_DIM * 2) * MODEL_NUM_LAYERS + MODEL_ELEMENTS) * sizeof(my_float_t);

	/* opened with ios::ate, so tellg() here is the file size */
	const size_t file_size = static_cast<size_t>(file.tellg());
	file.seekg(0, std::ios::beg);

	/* raw int8 payload of every quantized tensor in the model */
	const size_t q_raw_size = (size_t)MODEL_ELEMENTS *
	    ((MODEL_ELEMENTS * 4 + MODEL_HIDDEN_DIM * 3) * MODEL_NUM_LAYERS + MODEL_TOKENS);
	/* interleaved buffer also carries one float per group -> 17/16 of the raw size */
	const size_t packed_size = q_raw_size + q_raw_size * sizeof(my_float_t) / MODEL_SCALING_FACTOR;

	const size_t rms_size     = (size_t)MODEL_ELEMENTS * (MODEL_NUM_LAYERS * 2 + 1) * sizeof(my_float_t);
	const size_t dequant_size = embed_elem * sizeof(my_float_t);

	if (packed_size % FRAME_BYTES != 0) {
		std::cerr << "packed_size " << packed_size << " B is not a whole number of "
		          << FRAME_BYTES << " B frames at MAX_DW=" << MAX_DW << std::endl;
		exit(EXIT_FAILURE);
	}
	if ((packed_size / FRAME_BYTES) % NPORT != 0) {
		std::cerr << "frame count " << packed_size / FRAME_BYTES
		          << " does not divide across " << NPORT << " ports" << std::endl;
		exit(EXIT_FAILURE);
	}

	std::cout << "checkpoint " << file_size << " B" << std::endl;
	dump_layout(packed_size);
	verify_frame_plan(packed_size / BEAT_BYTES);

	/* the buffer the kernel actually reads through w_0 / w_1 */
	std::vector<wide_t> quant_arr(packed_size / BEAT_BYTES);

	/* Staging for the embedding table. Kept FLAT rather than as idata_v_t /
	 * mfdata_v_t: the old code indexed embed_q_arr[i/64][i%64], which assumes an
	 * idata_v_t holds 64 lanes and reads out of bounds for any MAX_DW < 512. */
	std::vector<int8_t>     embed_q_arr(embed_elem);
	std::vector<my_float_t> embed_sf_arr(embed_sf_cnt);

	std::vector<fdata_v_t> sf_w0_arr(dequant_size / sizeof(fdata_v_t));
	std::vector<fdata_v_t> rms_w_arr(rms_size / sizeof(fdata_v_t));
	std::vector<fdata_v_t> data_out_arr(data_out_size / sizeof(fdata_v_t));
	std::vector<fdata_v_t> GeMV_data_out_arr(gemv_data_out_size / sizeof(fdata_v_t));

	std::fill(data_out_arr.begin(), data_out_arr.end(), fdata_v_t(0));

	char *q_ptr   = reinterpret_cast<char*>(quant_arr.data());
	char *rms_ptr = reinterpret_cast<char*>(rms_w_arr.data());

/* ==== RMS NORM DATA ==== */
	size_t rms_idx = 0;
	file.seekg(256, std::ios::beg);              // v2 header is 256 bytes

	axi_regs.rms_att_W = 0;
	file.read(rms_ptr + rms_idx, rms_att_size);
	rms_idx += rms_att_size;

	axi_regs.rms_ffn_W = axi_regs.rms_att_W + (int)rms_att_size;
	file.read(rms_ptr + rms_idx, rms_ffn_size);
	rms_idx += rms_ffn_size;

	axi_regs.rms_final_W = axi_regs.rms_ffn_W + (int)rms_ffn_size;
	file.read(rms_ptr + rms_idx, rms_final_size);
	rms_idx += rms_final_size;

	if (!file || rms_idx != rms_size) {
		std::cerr << "rmsnorm read failed (" << rms_idx << " vs " << rms_size << ")\n";
		exit(EXIT_FAILURE);
	}

/* ==== SCAN THE QUANTIZED REGION ====
 * File order (runq.c memory_map_weights): q_tokens, wq, wk, wv, wo, w1, w2, w3.
 * Note w2 precedes w3 on disk even though the kernel consumes w1, w3, w2.
 */
	size_t embed_w_ptr = 0, embed_sf_ptr = 0;
	size_t query_w_ptr[MODEL_NUM_LAYERS], query_sf_ptr[MODEL_NUM_LAYERS];
	size_t key_w_ptr[MODEL_NUM_LAYERS],   key_sf_ptr[MODEL_NUM_LAYERS];
	size_t value_w_ptr[MODEL_NUM_LAYERS], value_sf_ptr[MODEL_NUM_LAYERS];
	size_t out_w_ptr[MODEL_NUM_LAYERS],   out_sf_ptr[MODEL_NUM_LAYERS];
	size_t w1_w_ptr[MODEL_NUM_LAYERS],    w1_sf_ptr[MODEL_NUM_LAYERS];
	size_t w2_w_ptr[MODEL_NUM_LAYERS],    w2_sf_ptr[MODEL_NUM_LAYERS];
	size_t w3_w_ptr[MODEL_NUM_LAYERS],    w3_sf_ptr[MODEL_NUM_LAYERS];

	scan_tensors(file, embed_elem, &embed_w_ptr, &embed_sf_ptr, 1);
	scan_tensors(file, nn_elem, query_w_ptr, query_sf_ptr, MODEL_NUM_LAYERS);
	scan_tensors(file, nn_elem, key_w_ptr,   key_sf_ptr,   MODEL_NUM_LAYERS);
	scan_tensors(file, nn_elem, value_w_ptr, value_sf_ptr, MODEL_NUM_LAYERS);
	scan_tensors(file, nn_elem, out_w_ptr,   out_sf_ptr,   MODEL_NUM_LAYERS);
	scan_tensors(file, nm_elem, w1_w_ptr,    w1_sf_ptr,    MODEL_NUM_LAYERS);
	scan_tensors(file, nm_elem, w2_w_ptr,    w2_sf_ptr,    MODEL_NUM_LAYERS);
	scan_tensors(file, nm_elem, w3_w_ptr,    w3_sf_ptr,    MODEL_NUM_LAYERS);

	/* stories110M ties the classifier to the embedding, so nothing should follow w3 */
	const size_t scan_end = static_cast<size_t>(file.tellg());
	if (scan_end != file_size) {
		std::cerr << "WARNING: scan ended at " << scan_end << " but file is " << file_size
		          << " B (" << (long long)file_size - (long long)scan_end
		          << " B unaccounted - separate wcls?)" << std::endl;
	}

/* ==== BUILD THE DEQUANTIZED EMBEDDING TABLE ====
 * Flat indexing, no vector-lane assumptions: element i belongs to scale group
 * i/MODEL_SCALING_FACTOR regardless of how wide a beat happens to be. */
	read_at(file, embed_w_ptr,  reinterpret_cast<char*>(embed_q_arr.data()),
	        embed_size, "embedding int8");
	read_at(file, embed_sf_ptr, reinterpret_cast<char*>(embed_sf_arr.data()),
	        embed_sf_size, "embedding scales");

	for (size_t i = 0; i < embed_elem; i++) {
		const float dq = static_cast<float>(embed_q_arr[i])
		               * embed_sf_arr[i / MODEL_SCALING_FACTOR];
		sf_w0_arr[i / SM_FL_ELEM][i % SM_FL_ELEM] = dq;
	}

	/* reclaim the int8 staging before the packer allocates its own */
	std::vector<int8_t>().swap(embed_q_arr);
	std::vector<my_float_t>().swap(embed_sf_arr);

/* ==== REPACK EVERYTHING INTO THE INTERLEAVED DEVICE LAYOUT ====
 * Consumption order must match the FSM in weight_fsm():
 *   per layer: QKV (one 3*768-row GeMV), O, w1+w3, w2
 *   then once: embedding table as the classifier
 */
	size_t q_idx = 0;
	for (int i = 0; i < MODEL_NUM_LAYERS; i++) {
		const TensorRef qkv[3] = {{query_w_ptr[i], query_sf_ptr[i], nn_elem},
		                          {key_w_ptr[i],   key_sf_ptr[i],   nn_elem},
		                          {value_w_ptr[i], value_sf_ptr[i], nn_elem}};
		const TensorRef o  [1] = {{out_w_ptr[i],   out_sf_ptr[i],   nn_elem}};
		const TensorRef w13[2] = {{w1_w_ptr[i],    w1_sf_ptr[i],    nm_elem},
		                          {w3_w_ptr[i],    w3_sf_ptr[i],    nm_elem}};
		const TensorRef w2 [1] = {{w2_w_ptr[i],    w2_sf_ptr[i],    nm_elem}};

		pack_gemv(file, q_ptr, q_idx, qkv, 3, "qkv");
		pack_gemv(file, q_ptr, q_idx, o,   1, "o");
		pack_gemv(file, q_ptr, q_idx, w13, 2, "w1w3");
		pack_gemv(file, q_ptr, q_idx, w2,  1, "w2");
	}
	const TensorRef emb[1] = {{embed_w_ptr, embed_sf_ptr, embed_elem}};
	pack_gemv(file, q_ptr, q_idx, emb, 1, "classifier");

	if (q_idx != packed_size) {
		std::cerr << "packer wrote " << q_idx << " B into a " << packed_size
		          << " B buffer" << std::endl;
		exit(EXIT_FAILURE);
	}
	std::cout << "packed " << q_idx << " B into " << q_idx / FRAME_BYTES
	          << " frames across " << NPORT << " ports" << std::endl;

/* ============================== constants related to tb ============================== */
	const int layer_cnt      = MODEL_NUM_LAYERS;
	const int tokens_size    = MODEL_ELEMENTS * (int)sizeof(my_float_t);
	const int tok_w1_size    = MODEL_HIDDEN_DIM * (int)sizeof(my_float_t);
	const int logits_size    = INTERNAL_DATA_SIZE * (int)sizeof(my_float_t);
	const size_t cache_size  = (size_t)MODEL_SEQUENCE_LEN * MODEL_ELEMENTS * sizeof(my_float_t) * MODEL_NUM_LAYERS;

	const int cache_cnt      = (int)(cache_size / sizeof(cache_v_t));
	const int tokens_cnt     = tokens_size / (int)sizeof(fdata_v_t);
	const int tok_w1_cnt     = tok_w1_size / (int)sizeof(fdata_v_t);
	const int logits_cnt     = logits_size / (int)sizeof(fdata_v_t);
	const int data_goa_cnt   = (MODEL_ELEMENTS * 3 + MODEL_HIDDEN_DIM) * layer_cnt / SM_FL_ELEM;
	const int data_gemv_goa_cnt = (MODEL_ELEMENTS * 5 + MODEL_HIDDEN_DIM * 2) * layer_cnt / SM_FL_ELEM;

	assert(cache_size % sizeof(cache_v_t) == 0 && "cache must be a whole number of beats");

/* ===================================== declare our vectors ===================================== */

	std::vector<fdata_v_t> tokens_arr(tokens_cnt * 3);
	std::vector<fdata_v_t> swiglu_arr(tok_w1_cnt * 2);
	std::vector<fdata_v_t> output_arr(logits_cnt);
	std::vector<fdata_v_t> golden_output_arr(data_goa_cnt);
	std::vector<fdata_v_t> golden_gemv_output_arr(data_gemv_goa_cnt);

	std::fill(golden_output_arr.begin(), golden_output_arr.end(), fdata_v_t(0));
	std::fill(golden_gemv_output_arr.begin(), golden_gemv_output_arr.end(), fdata_v_t(0));

	/* read golden references, clamped to the destination so a stale/oversized
	 * golden file can't scribble past the vector */
	auto read_clamped = [](std::ifstream &s, char *dst, size_t cap, const char *what) {
		s.seekg(0, std::ios::end);
		const size_t sz = static_cast<size_t>(s.tellg());
		s.seekg(0, std::ios::beg);
		if (sz > cap) {
			std::cerr << "WARNING: " << what << " is " << sz << " B, buffer is " << cap
			          << " B - truncating" << std::endl;
		}
		s.read(dst, std::min(sz, cap));
		return std::min(sz, cap);
	};

	read_clamped(data_output, reinterpret_cast<char*>(golden_output_arr.data()),
	             golden_output_arr.size() * sizeof(fdata_v_t), "150_pre_quantized.bin");
	read_clamped(gemv_data_output, reinterpret_cast<char*>(golden_gemv_output_arr.data()),
	             golden_gemv_output_arr.size() * sizeof(fdata_v_t), "150_post_matmul.bin");
	read_clamped(input_tokens, reinterpret_cast<char*>(output_arr.data()),
	             output_arr.size() * sizeof(fdata_v_t), "150_01_rms_att_in.bin");

/* ===================================== KV cache transpose =====================================
 * Byte-level transpose, so it is independent of the cache vector width as long
 * as one head is a whole number of beats (static_asserted above). */

	std::vector<cache_v_t> key_arr(cache_cnt);
	std::vector<cache_v_t> value_arr(cache_cnt);

	const size_t head_dim_bytes = MODEL_HEAD_SIZE * sizeof(my_float_t);
	static_assert(MODEL_HEAD_SIZE * sizeof(my_float_t) % sizeof(cache_v_t) == 0,
	              "one head must be a whole number of cache beats");
	std::vector<char> raw_token_major_buf(cache_size);

	/* golden caches are [layer][token][head][head_dim]; the kernel wants
	 * [layer][head][token][head_dim] */
	auto transpose_cache = [&](std::ifstream &src, cache_v_t *dstv, const char *what) {
		src.read(raw_token_major_buf.data(), cache_size);
		if ((size_t)src.gcount() != cache_size) {
			std::cerr << "short read on " << what << " (" << src.gcount()
			          << " of " << cache_size << ")" << std::endl;
			exit(EXIT_FAILURE);
		}
		char *dst = reinterpret_cast<char*>(dstv);
		const char *src_ptr = raw_token_major_buf.data();
		for (int l = 0; l < MODEL_NUM_LAYERS; l++)
			for (int h = 0; h < MODEL_NUM_HEADS; h++)
				for (int t = 0; t < MODEL_SEQUENCE_LEN; t++) {
					const size_t src_off = l * (MODEL_SEQUENCE_LEN * MODEL_NUM_HEADS * head_dim_bytes)
					                     + t * (MODEL_NUM_HEADS * head_dim_bytes)
					                     + h * head_dim_bytes;
					std::memcpy(dst, src_ptr + src_off, head_dim_bytes);
					dst += head_dim_bytes;
				}
		assert((size_t)(dst - reinterpret_cast<char*>(dstv)) == cache_size);
	};

	transpose_cache(out_key_dat,   key_arr.data(),   "key cache");
	transpose_cache(out_value_dat, value_arr.data(), "value cache");

/* ================================== token / coin setup =================================== */
/* ================================== read data into array =================================== */

	int curr_pos = 150;
	std::cout<<"Loaded the files into memory"<<std::endl;
	float coin;

	std::vector<int> ct(1024);
	std::fill(ct.begin(), ct.end(), 0);
	token_data.seekg(0, std::ios::end);
	int zz = token_data.tellg();
	token_data.seekg(0, std::ios::beg);
	token_data.read((reinterpret_cast<char*>(ct.data()) + 4), zz);

	token_data.seekg(0, std::ios::end);

	coin_data.seekg((curr_pos - 4) * 4);
	token_data.seekg((curr_pos - 1) * 4);

	int32_t next_token = ct.at(curr_pos + 1);
	ct.at(curr_pos + 1) = -1;
	char * coin_ptr = reinterpret_cast<char *>(&coin);

	coin_data.read(coin_ptr, 4);

	file.close();
	coin_data.close();
	token_data.close();
	data_output.close();
	gemv_data_output.close();
	out_key_dat.close();
	out_value_dat.close();
	input_tokens.close();

/* ============================ call the kernel ====================== */

	const float temperature = 0.9f;
	std::cout << "Loaded the files into memory" << std::endl;

	transformer_cu(
			sf_w0_arr.data(),                       // fdata_v_t *tokens (dequantized embedding table)
			quant_arr.data(), quant_arr.data(),     // wide_t *w_0, *w_1 - same buffer, port 1 starts one frame in
			rms_w_arr.data(),                       // fdata_v_t *weights
			key_arr.data(), value_arr.data(),       // cache_v_t *key_cache, *value_cache
			curr_pos,                               // const int POS
			axi_regs.rms_att_W, axi_regs.rms_ffn_W, axi_regs.rms_final_W,
			ct.data(),
			#ifdef __DEBUG__
			4, 0, 0, data_out_arr.data(),
			#endif
			#ifdef __ULTRADEBUG__
			GeMV_data_out_arr.data(),
			#endif
			temperature, coin, true, false
			);

	#ifdef __DEBUG__
	std::cout << "===== pre-quantized token data =====" << std::endl;
	parse_results<fdata_v_t, float>(golden_output_arr, data_out_arr);
	#endif

	#ifdef __ULTRADEBUG__
	std::cout << "===== post-matmul GeMV data =====" << std::endl;
	parse_results<fdata_v_t, float>(golden_gemv_output_arr, GeMV_data_out_arr);
	#endif

	const int32_t got = ct.at(curr_pos + 1);
	std::cout << "Golden token: \t" << next_token << "\t Actual token: \t" << got << std::endl;
	// return (got == next_token) ? 0 : 1;
	return 0;
}

// #include <algorithm>
// #include <cassert>
// #include <cstddef>
// #include <cstdint>
// #include <cstdio>
// #include <cstring>
// #include <hls_math.h>
// #include <hls_stream.h>
// #include <stdio.h>
// #include <streambuf>
// #include <vector>
// #include <fstream>
// #include <iostream>
// #include <cmath>
// #include <bitset>
// #include "tb_main.h"

// /* ---- sanity on the packed layout; these must hold or the frames are wrong ---- */
// static_assert(sizeof(my_float_t)  == 4,  "device side assumes 32-bit float");
// static_assert(sizeof(wide_t)      == 64, "wide_t must be one 512-bit AXI beat");
// static_assert(sizeof(idata_v_t)   == 64, "idata_v_t must be one 512-bit AXI beat");
// static_assert(sizeof(mfdata_v_t)  == 64, "mfdata_v_t must be one 512-bit AXI beat");
// static_assert(sizeof(fdata_v_t)   == 16, "fdata_v_t must be 4 floats");
// static_assert(SF_FRAME * sizeof(wide_t) == QUANT_FRAME * sizeof(my_float_t),
//               "SF_FRAME beats must hold exactly QUANT_FRAME scale factors");

// /* bytes moved per interleaved frame */
// static constexpr size_t FRAME_SF_BYTES = QUANT_FRAME * sizeof(my_float_t);          // 768
// static constexpr size_t FRAME_W_BYTES  = QUANT_FRAME * MODEL_SCALING_FACTOR;        // 12288
// static constexpr size_t FRAME_BYTES    = FRAME_SF_BYTES + FRAME_W_BYTES;            // 13056

// struct axi_reg_t {
// 	int POS;
// 	int N_DIM;
// 	int M_DIM;
// 	int rms_att_W;
// 	int rms_ffn_W;
// 	int rms_final_W;
// };

// /* ---------------------------------------------------------------------------
//  * Walk a run of `n` consecutively-stored quantized tensors, recording the file
//  * offset of each tensor's int8 block and of its scale-factor block.
//  *
//  * llama2.c runq.c init_quantized_tensors() stores, per tensor:
//  *     [ n_elem int8 ][ n_elem/GS float ]
//  * ------------------------------------------------------------------------- */
// static void scan_tensors(std::ifstream &f, size_t n_elem,
//                          size_t *w_ptr, size_t *sf_ptr, int n)
// {
// 	const size_t w_bytes  = n_elem * sizeof(int8_t);
// 	const size_t sf_bytes = (n_elem / MODEL_SCALING_FACTOR) * sizeof(my_float_t);

// 	for (int i = 0; i < n; i++) {
// 		w_ptr[i] = static_cast<size_t>(f.tellg());
// 		f.seekg(w_bytes, std::ios::cur);          // relative!
// 		sf_ptr[i] = static_cast<size_t>(f.tellg());
// 		f.seekg(sf_bytes, std::ios::cur);         // relative!
// 		if (!f) { std::cerr << "scan_tensors: ran off the end of the checkpoint\n"; exit(EXIT_FAILURE); }
// 	}
// }

// /* ---------------------------------------------------------------------------
//  * Repack one tensor into the interleaved device layout:
//  *     [192 scale floats][192*64 int8] repeated (n_elem / 12288) times
//  * ------------------------------------------------------------------------- */
// // static void pack_tensor(std::ifstream &f, char *dst, size_t &idx,
// //                         size_t w_base, size_t sf_base, size_t n_elem)
// // {
// // 	assert(n_elem % (QUANT_FRAME * MODEL_SCALING_FACTOR) == 0 &&
// // 	       "frame size must divide the tensor exactly");
// // 	const size_t n_frame = n_elem / (QUANT_FRAME * MODEL_SCALING_FACTOR);

// // 	for (size_t j = 0; j < n_frame; j++) {
// // 		f.seekg(sf_base + j * FRAME_SF_BYTES, std::ios::beg);
// // 		f.read(dst + idx, FRAME_SF_BYTES);
// // 		if (f.gcount() != (std::streamsize)FRAME_SF_BYTES) {
// // 			std::cerr << "pack_tensor: short read on scale factors\n"; exit(EXIT_FAILURE);
// // 		}
// // 		idx += FRAME_SF_BYTES;

// // 		f.seekg(w_base + j * FRAME_W_BYTES, std::ios::beg);
// // 		f.read(dst + idx, FRAME_W_BYTES);
// // 		if (f.gcount() != (std::streamsize)FRAME_W_BYTES) {
// // 			std::cerr << "pack_tensor: short read on weights\n"; exit(EXIT_FAILURE);
// // 		}
// // 		idx += FRAME_W_BYTES;
// // 	}
// // }

// struct TensorRef { size_t w_base, sf_base, n_elem; };

// /* Emit one GeMV's weights, split into NPORT contiguous row-halves and then
//  * interleaved frame-by-frame so mm2ds_input_data's even/odd pickup delivers
//  * each port a contiguous run of output rows. */
// static void pack_gemv(std::ifstream &f, char *dst, size_t &idx,
//                       const TensorRef *ts, int n_ts, int NPORT)
// {
// 	/* flatten the concatenated matmul into one frame list */
// 	std::vector<std::pair<size_t, size_t>> fr;      // (w_off, sf_off)
// 	for (int t = 0; t < n_ts; t++) {
// 		assert(ts[t].n_elem % (QUANT_FRAME * MODEL_SCALING_FACTOR) == 0);
// 		const size_t nf = ts[t].n_elem / (QUANT_FRAME * MODEL_SCALING_FACTOR);
// 		for (size_t j = 0; j < nf; j++)
// 			fr.emplace_back(ts[t].w_base  + j * FRAME_W_BYTES,
// 			                ts[t].sf_base + j * FRAME_SF_BYTES);
// 	}
// 	assert(fr.size() % NPORT == 0 && "GeMV frames must divide across ports");
// 	const size_t per_port = fr.size() / NPORT;

// 	for (size_t j = 0; j < per_port; j++) {
// 		for (int k = 0; k < NPORT; k++) {
// 			const auto &e = fr[k * per_port + j];       // <-- per_port, not n_frame

// 			f.seekg(e.second, std::ios::beg);
// 			f.read(dst + idx, FRAME_SF_BYTES);
// 			if (f.gcount() != (std::streamsize)FRAME_SF_BYTES) {
// 				std::cerr << "pack_gemv: short read on scale factors\n"; exit(EXIT_FAILURE);
// 			}
// 			idx += FRAME_SF_BYTES;

// 			f.seekg(e.first, std::ios::beg);
// 			f.read(dst + idx, FRAME_W_BYTES);
// 			if (f.gcount() != (std::streamsize)FRAME_W_BYTES) {
// 				std::cerr << "pack_gemv: short read on weights\n"; exit(EXIT_FAILURE);
// 			}
// 			idx += FRAME_W_BYTES;
// 		}
// 	}
// }

// int top_tb(){
// 	std::cout<<"starting First Third testbench"<<std::endl;

// 	axi_reg_t axi_regs{};

// /* ====== INPUT DATA AND CHECKS ============ INPUT DATA AND CHECKS ====== */
// 	std::string checkpoint = "weights/stories110M_q8.bin";
// 	std::ifstream file(checkpoint, std::ios::binary | std::ios::ate);
// 	std::ifstream coin_data("newgolden/150_coin.bin", std::ios::binary);
// 	std::ifstream token_data("newgolden/150_tokens.bin", std::ios::binary);
// 	std::ifstream data_output("newgolden/150_pre_quantized.bin", std::ios::binary);
// 	std::ifstream gemv_data_output("newgolden/150_post_matmul.bin", std::ios::binary);
// 	std::ifstream out_key_dat("newgolden/150_key_cache.bin", std::ios::binary);
// 	std::ifstream out_value_dat("newgolden/150_value_cache.bin", std::ios::binary);
// 	std::ifstream input_tokens("newgolden/150_01_rms_att_in.bin", std::ios::binary);

// 	struct { std::ifstream *s; const char *n; } fl[] = {
// 		{&file, "checkpoint"}, {&coin_data, "coin_data"}, {&token_data, "token_data"},
// 		{&data_output, "data_output"}, {&gemv_data_output, "gemv_data_output"},
// 		{&out_key_dat, "out_key_dat"}, {&out_value_dat, "out_value_dat"},
// 		{&input_tokens, "input_tokens"},
// 	};
// 	for (auto &e : fl) {
// 		if (!e.s->is_open()) {
// 			std::cerr << "Could not open " << e.n << ". Already off to a bad start." << std::endl;
// 			exit(EXIT_FAILURE);
// 		}
// 	}

// /* ===== MEMORY VECTOR ARRAY CONSTRUCTOR ===== */

// 	const size_t rms_att_size   = MODEL_ELEMENTS * MODEL_NUM_LAYERS * sizeof(my_float_t);
// 	const size_t rms_ffn_size   = rms_att_size;
// 	const size_t rms_final_size = MODEL_ELEMENTS * sizeof(my_float_t);

// 	/* element counts, not byte counts - scan_tensors/pack_tensor derive bytes */
// 	const size_t nn_elem    = (size_t)MODEL_ELEMENTS * MODEL_ELEMENTS;      // wq/wk/wv/wo
// 	const size_t nm_elem    = (size_t)MODEL_ELEMENTS * MODEL_HIDDEN_DIM;    // w1/w2/w3
// 	const size_t embed_elem = (size_t)MODEL_ELEMENTS * MODEL_TOKENS;        // q_tokens

// 	const size_t embed_size    = embed_elem * sizeof(int8_t);
// 	const size_t embed_sf_size = (embed_elem / MODEL_SCALING_FACTOR) * sizeof(my_float_t);

// 	const size_t data_out_size      = ((MODEL_ELEMENTS * 3 + MODEL_HIDDEN_DIM) * MODEL_NUM_LAYERS + MODEL_ELEMENTS) * sizeof(my_float_t);
// 	const size_t gemv_data_out_size = ((MODEL_ELEMENTS * 5 + MODEL_HIDDEN_DIM * 2) * MODEL_NUM_LAYERS + MODEL_ELEMENTS) * sizeof(my_float_t);

// 	/* opened with ios::ate, so tellg() here is the file size */
// 	const size_t file_size = static_cast<size_t>(file.tellg());
// 	file.seekg(0, std::ios::beg);

// 	/* raw int8 payload of every quantized tensor in the model */
// 	const size_t q_raw_size = (size_t)MODEL_ELEMENTS *
// 	    ((MODEL_ELEMENTS * 4 + MODEL_HIDDEN_DIM * 3) * MODEL_NUM_LAYERS + MODEL_TOKENS);
// 	/* interleaved buffer also carries one float per group -> 17/16 of the raw size */
// 	const size_t packed_size = q_raw_size + q_raw_size * sizeof(my_float_t) / MODEL_SCALING_FACTOR;

// 	const size_t rms_size     = (size_t)MODEL_ELEMENTS * (MODEL_NUM_LAYERS * 2 + 1) * sizeof(my_float_t);
// 	const size_t dequant_size = embed_elem * sizeof(my_float_t);

// 	assert(packed_size % sizeof(wide_t) == 0);
// 	assert(packed_size / sizeof(wide_t) % TXFR_FRAME == 0);

// 	std::cout << "checkpoint " << file_size << " B, packed weight buffer "
// 	          << packed_size << " B (" << packed_size / sizeof(wide_t) << " beats, "
// 	          << packed_size / sizeof(wide_t) / TXFR_FRAME << " frames)" << std::endl;

// 	/* the buffer the kernel actually reads through w_0 / w_1 */
// 	std::vector<wide_t> quant_arr(packed_size / sizeof(wide_t));
// 	/* staging for the embedding table only (int8 + scales), used to build sf_w0_arr */
// 	std::vector<idata_v_t> embed_q_arr(embed_size / sizeof(idata_v_t));
// 	std::vector<mfdata_v_t> embed_sf_arr(embed_sf_size / sizeof(mfdata_v_t));

// 	std::vector<fdata_v_t> sf_w0_arr(dequant_size / sizeof(fdata_v_t));
// 	std::vector<fdata_v_t> rms_w_arr(rms_size / sizeof(fdata_v_t));
// 	std::vector<fdata_v_t> data_out_arr(data_out_size / sizeof(fdata_v_t));
// 	std::vector<fdata_v_t> GeMV_data_out_arr(gemv_data_out_size / sizeof(fdata_v_t));

// 	std::fill(data_out_arr.begin(), data_out_arr.end(), fdata_v_t(0));

// 	char *q_ptr        = reinterpret_cast<char*>(quant_arr.data());
// 	char *embed_q_ptr  = reinterpret_cast<char*>(embed_q_arr.data());
// 	char *embed_sf_p   = reinterpret_cast<char*>(embed_sf_arr.data());
// 	char *rms_ptr      = reinterpret_cast<char*>(rms_w_arr.data());

// /* ==== RMS NORM DATA ==== */
// 	size_t rms_idx = 0;
// 	file.seekg(256, std::ios::beg);              // v2 header is 256 bytes

// 	axi_regs.rms_att_W = 0;
// 	file.read(rms_ptr + rms_idx, rms_att_size);
// 	rms_idx += rms_att_size;

// 	axi_regs.rms_ffn_W = axi_regs.rms_att_W + (int)rms_att_size;
// 	file.read(rms_ptr + rms_idx, rms_ffn_size);
// 	rms_idx += rms_ffn_size;

// 	axi_regs.rms_final_W = axi_regs.rms_ffn_W + (int)rms_ffn_size;
// 	file.read(rms_ptr + rms_idx, rms_final_size);
// 	rms_idx += rms_final_size;

// 	if (!file || rms_idx != rms_size) {
// 		std::cerr << "rmsnorm read failed (" << rms_idx << " vs " << rms_size << ")\n";
// 		exit(EXIT_FAILURE);
// 	}

// /* ==== SCAN THE QUANTIZED REGION ====
//  * File order (runq.c memory_map_weights): q_tokens, wq, wk, wv, wo, w1, w2, w3.
//  * Note w2 precedes w3 on disk even though the kernel consumes w1, w3, w2.
//  */
// 	size_t embed_w_ptr = 0, embed_sf_ptr = 0;
// 	size_t query_w_ptr[MODEL_NUM_LAYERS], query_sf_ptr[MODEL_NUM_LAYERS];
// 	size_t key_w_ptr[MODEL_NUM_LAYERS],   key_sf_ptr[MODEL_NUM_LAYERS];
// 	size_t value_w_ptr[MODEL_NUM_LAYERS], value_sf_ptr[MODEL_NUM_LAYERS];
// 	size_t out_w_ptr[MODEL_NUM_LAYERS],   out_sf_ptr[MODEL_NUM_LAYERS];
// 	size_t w1_w_ptr[MODEL_NUM_LAYERS],    w1_sf_ptr[MODEL_NUM_LAYERS];
// 	size_t w2_w_ptr[MODEL_NUM_LAYERS],    w2_sf_ptr[MODEL_NUM_LAYERS];
// 	size_t w3_w_ptr[MODEL_NUM_LAYERS],    w3_sf_ptr[MODEL_NUM_LAYERS];

// 	scan_tensors(file, embed_elem, &embed_w_ptr, &embed_sf_ptr, 1);
// 	scan_tensors(file, nn_elem, query_w_ptr, query_sf_ptr, MODEL_NUM_LAYERS);
// 	scan_tensors(file, nn_elem, key_w_ptr,   key_sf_ptr,   MODEL_NUM_LAYERS);
// 	scan_tensors(file, nn_elem, value_w_ptr, value_sf_ptr, MODEL_NUM_LAYERS);
// 	scan_tensors(file, nn_elem, out_w_ptr,   out_sf_ptr,   MODEL_NUM_LAYERS);
// 	scan_tensors(file, nm_elem, w1_w_ptr,    w1_sf_ptr,    MODEL_NUM_LAYERS);
// 	scan_tensors(file, nm_elem, w2_w_ptr,    w2_sf_ptr,    MODEL_NUM_LAYERS);
// 	scan_tensors(file, nm_elem, w3_w_ptr,    w3_sf_ptr,    MODEL_NUM_LAYERS);

// 	/* stories110M ties the classifier to the embedding, so nothing should follow w3 */
// 	const size_t scan_end = static_cast<size_t>(file.tellg());
// 	if (scan_end != file_size) {
// 		std::cerr << "WARNING: scan ended at " << scan_end << " but file is " << file_size
// 		          << " B (" << (long long)file_size - (long long)scan_end
// 		          << " B unaccounted - separate wcls?)" << std::endl;
// 	}

// /* ==== BUILD THE DEQUANTIZED EMBEDDING TABLE ==== */
// 	file.seekg(embed_w_ptr, std::ios::beg);
// 	file.read(embed_q_ptr, embed_size);
// 	file.seekg(embed_sf_ptr, std::ios::beg);
// 	file.read(embed_sf_p, embed_sf_size);
// 	if (!file) { std::cerr << "embedding read failed\n"; exit(EXIT_FAILURE); }

// 	for (size_t i = 0; i < embed_elem; i++) {
// 		const size_t group  = i / MODEL_SCALING_FACTOR;   // group of 64 sharing a scale
// 		const size_t q_sub  = i % MODEL_SCALING_FACTOR;   // lane within the int8 vector
// 		const size_t sfg    = group / MAX_FL_ELEM;        // which mfdata_v_t (16 scales)
// 		const size_t sfs    = group % MAX_FL_ELEM;

// 		const float dq = static_cast<float>(embed_q_arr[group][q_sub]) * embed_sf_arr[sfg][sfs];
// 		sf_w0_arr[i / SM_FL_ELEM][i % SM_FL_ELEM] = dq;
// 	}

// /* ==== REPACK EVERYTHING INTO THE INTERLEAVED DEVICE LAYOUT ====
//  * Consumption order must match the FSM in weight_fsm():
//  *   per layer: QKV (one 3*768-row GeMV), O, w1+w3, w2
//  *   then once: embedding table as the classifier
//  */
// 	size_t q_idx = 0;
// 	for (int i = 0; i < MODEL_NUM_LAYERS; i++) {
// 		const TensorRef qkv[3] = {{query_w_ptr[i], query_sf_ptr[i], nn_elem},
// 		                          {key_w_ptr[i],   key_sf_ptr[i],   nn_elem},
// 		                          {value_w_ptr[i], value_sf_ptr[i], nn_elem}};
// 		const TensorRef o  [1] = {{out_w_ptr[i],   out_sf_ptr[i],   nn_elem}};
// 		const TensorRef w13[2] = {{w1_w_ptr[i],    w1_sf_ptr[i],    nm_elem},
// 		                          {w3_w_ptr[i],    w3_sf_ptr[i],    nm_elem}};
// 		const TensorRef w2 [1] = {{w2_w_ptr[i],    w2_sf_ptr[i],    nm_elem}};

// 		pack_gemv(file, q_ptr, q_idx, qkv, 3, 2);
// 		pack_gemv(file, q_ptr, q_idx, o,   1, 2);
// 		pack_gemv(file, q_ptr, q_idx, w13, 2, 2);
// 		pack_gemv(file, q_ptr, q_idx, w2,  1, 2);
// 	}
// 	const TensorRef emb[1] = {{embed_w_ptr, embed_sf_ptr, embed_elem}};
// 	pack_gemv(file, q_ptr, q_idx, emb, 1, 2);

// /* ============================== constants related to tb ============================== */
// 	const int layer_cnt      = MODEL_NUM_LAYERS;
// 	const int tokens_size    = MODEL_ELEMENTS * (int)sizeof(my_float_t);
// 	const int tok_w1_size    = MODEL_HIDDEN_DIM * (int)sizeof(my_float_t);
// 	const int logits_size    = INTERNAL_DATA_SIZE * (int)sizeof(my_float_t);
// 	const size_t cache_size  = (size_t)MODEL_SEQUENCE_LEN * MODEL_ELEMENTS * sizeof(my_float_t) * MODEL_NUM_LAYERS;

// 	const int cache_cnt      = (int)(cache_size / sizeof(mfdata_v_t));
// 	const int tokens_cnt     = tokens_size / (int)sizeof(fdata_v_t);
// 	const int tok_w1_cnt     = tok_w1_size / (int)sizeof(fdata_v_t);
// 	const int logits_cnt     = logits_size / (int)sizeof(fdata_v_t);
// 	const int data_goa_cnt   = (MODEL_ELEMENTS * 3 + MODEL_HIDDEN_DIM) * layer_cnt / SM_FL_ELEM;
// 	const int data_gemv_goa_cnt = (MODEL_ELEMENTS * 5 + MODEL_HIDDEN_DIM * 2) * layer_cnt / SM_FL_ELEM;

// /* ===================================== declare our vectors ===================================== */

// 	std::vector<fdata_v_t> tokens_arr(tokens_cnt * 3);
// 	std::vector<fdata_v_t> swiglu_arr(tok_w1_cnt * 2);
// 	std::vector<fdata_v_t> output_arr(logits_cnt);
// 	std::vector<fdata_v_t> golden_output_arr(data_goa_cnt);
// 	std::vector<fdata_v_t> golden_gemv_output_arr(data_gemv_goa_cnt);

// 	std::fill(golden_output_arr.begin(), golden_output_arr.end(), fdata_v_t(0));
// 	std::fill(golden_gemv_output_arr.begin(), golden_gemv_output_arr.end(), fdata_v_t(0));

// 	/* read golden references, clamped to the destination so a stale/oversized
// 	 * golden file can't scribble past the vector */
// 	auto read_clamped = [](std::ifstream &s, char *dst, size_t cap, const char *what) {
// 		s.seekg(0, std::ios::end);
// 		const size_t sz = static_cast<size_t>(s.tellg());
// 		s.seekg(0, std::ios::beg);
// 		if (sz > cap) {
// 			std::cerr << "WARNING: " << what << " is " << sz << " B, buffer is " << cap
// 			          << " B - truncating" << std::endl;
// 		}
// 		s.read(dst, std::min(sz, cap));
// 		return std::min(sz, cap);
// 	};

// 	read_clamped(data_output, reinterpret_cast<char*>(golden_output_arr.data()),
// 	             golden_output_arr.size() * sizeof(fdata_v_t), "150_pre_quantized.bin");
// 	read_clamped(gemv_data_output, reinterpret_cast<char*>(golden_gemv_output_arr.data()),
// 	             golden_gemv_output_arr.size() * sizeof(fdata_v_t), "150_post_matmul.bin");
// 	read_clamped(input_tokens, reinterpret_cast<char*>(output_arr.data()),
// 	             output_arr.size() * sizeof(fdata_v_t), "150_01_rms_att_in.bin");

// /* ===================================== KV cache transpose ===================================== */

// 	std::vector<mfdata_v_t> key_arr(cache_cnt);
// 	std::vector<mfdata_v_t> value_arr(cache_cnt);

// 	const size_t head_dim_bytes = MODEL_HEAD_SIZE * sizeof(my_float_t);
// 	std::vector<char> raw_token_major_buf(cache_size);

// 	/* golden caches are [layer][token][head][head_dim]; the kernel wants
// 	 * [layer][head][token][head_dim] */
// 	auto transpose_cache = [&](std::ifstream &src, mfdata_v_t *dstv, const char *what) {
// 		src.read(raw_token_major_buf.data(), cache_size);
// 		if ((size_t)src.gcount() != cache_size) {
// 			std::cerr << "short read on " << what << " (" << src.gcount()
// 			          << " of " << cache_size << ")" << std::endl;
// 			exit(EXIT_FAILURE);
// 		}
// 		char *dst = reinterpret_cast<char*>(dstv);
// 		const char *src_ptr = raw_token_major_buf.data();
// 		for (int l = 0; l < MODEL_NUM_LAYERS; l++)
// 			for (int h = 0; h < MODEL_NUM_HEADS; h++)
// 				for (int t = 0; t < MODEL_SEQUENCE_LEN; t++) {
// 					const size_t src_off = l * (MODEL_SEQUENCE_LEN * MODEL_NUM_HEADS * head_dim_bytes)
// 					                     + t * (MODEL_NUM_HEADS * head_dim_bytes)
// 					                     + h * head_dim_bytes;
// 					std::memcpy(dst, src_ptr + src_off, head_dim_bytes);
// 					dst += head_dim_bytes;
// 				}
// 		assert((size_t)(dst - reinterpret_cast<char*>(dstv)) == cache_size);
// 	};

// 	transpose_cache(out_key_dat,   key_arr.data(),   "key cache");
// 	transpose_cache(out_value_dat, value_arr.data(), "value cache");

// /* ================================== token / coin setup =================================== */
// /* ================================== read data into array =================================== */

// 	int curr_pos = 150;
// 	std::cout<<"Loaded the files into memory"<<std::endl;
// 	float coin;
// 	// int32_t curr_token; 
	
// 	std::vector<int> ct(1024);
// 	std::fill(ct.begin(), ct.end(), 0);
// 	token_data.seekg(0, std::ios::end);
// 	int zz = token_data.tellg();
// 	token_data.seekg(0, std::ios::beg);
// 	token_data.read((reinterpret_cast<char*>(ct.data()) + 4), zz);
	
// 	token_data.seekg(0, std::ios::end);
	
// 	coin_data.seekg((curr_pos - 4) * 4);
// 	token_data.seekg((curr_pos - 1) * 4);
	
// 	int32_t next_token = ct.at(curr_pos + 1);
// 	ct.at(curr_pos + 1) = -1;
// 	char * coin_ptr = reinterpret_cast<char *>(&coin);
// 	// char * token_ptr = reinterpret_cast<char *>(&next_token);

// 	coin_data.read(coin_ptr, 4);
// 	// token_data.read(reinterpret_cast<char*>(&curr_token), 4);
// 	// token_data.read(token_ptr, 4);

// 	file.close();
// 	coin_data.close();
// 	token_data.close();
// 	data_output.close();
// 	gemv_data_output.close();
// 	out_key_dat.close();
// 	out_value_dat.close();
// 	input_tokens.close();

// /* ============================ call the kernel ====================== */

// 	const float temperature = 0.9f;
// 	std::cout << "Loaded the files into memory" << std::endl;

// 	transformer_cu(
// 			sf_w0_arr.data(),                       // fdata_v_t *tokens (dequantized embedding table)
// 			quant_arr.data(), quant_arr.data(),     // wide_t *w_0, *w_1 - same buffer, port 1 starts one frame in
// 			rms_w_arr.data(),                       // fdata_v_t *weights
// 			key_arr.data(), value_arr.data(),       // mfdata_v_t *key_cache, *value_cache
// 			curr_pos,                               // const int POS
// 			axi_regs.rms_att_W, axi_regs.rms_ffn_W, axi_regs.rms_final_W,
// 			ct.data(),
// 			#ifdef __DEBUG__
// 			4, 0, 0, data_out_arr.data(),
// 			#endif
// 			#ifdef __ULTRADEBUG__
// 			GeMV_data_out_arr.data(),
// 			#endif
// 			temperature, coin, true, false
// 			);

// 	#ifdef __DEBUG__
// 	std::cout << "===== pre-quantized token data =====" << std::endl;
// 	parse_results<fdata_v_t, float>(golden_output_arr, data_out_arr);
// 	#endif

// 	#ifdef __ULTRADEBUG__
// 	std::cout << "===== post-matmul GeMV data =====" << std::endl;
// 	parse_results<fdata_v_t, float>(golden_gemv_output_arr, GeMV_data_out_arr);
// 	#endif

// 	const int32_t got = ct.at(curr_pos + 1);
// 	std::cout << "Golden token: \t" << next_token << "\t Actual token: \t" << got << std::endl;
// 	// return (got == next_token) ? 0 : 1;
// 	return 0;
// }