// ============================================================================
//  compute_units.h  --  host-side drivers for the transformer_cu kernel
//
//  Two functionally identical engines with the same public API:
//
//    FastForward : bare-metal AXI-Lite register poking via xrt::ip
//    RunForward  : managed flow via xrt::kernel / xrt::run
//
//  Both treat `curr_token` as an m_axi INT ARRAY of MODEL_SEQUENCE_LEN
//  entries (matching the kernel's
//      #pragma HLS INTERFACE mode=m_axi port=curr_token bundle=token_val
//                                       depth=MODEL_SEQUENCE_LEN)
//
//  Per-step protocol:
//      host writes curr_token[pos]                (seed_prompt / set_token)
//      startForward(pos, coin)  -> sync TO device, set POS/coin, launch
//      endForward(pos)          -> wait done, sync FROM device,
//                                  return curr_token[pos+1]
//
//  During prefill the kernel does NOT write curr_token[pos+1], so
//  endForward() simply hands back the prompt token the host already seeded.
//  The same call therefore works in both phases.
// ============================================================================
#ifndef COMPUTE_UNITS_H
#define COMPUTE_UNITS_H

#include "xrt/xrt_device.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_kernel.h"
#include "experimental/xrt_ip.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

// Point this at wherever your build drops the generated driver header.
#ifndef TRANSFORMER_CU_HW_HEADER
#define TRANSFORMER_CU_HW_HEADER \
    "../transformer_cu/transformer_cu/hls/impl/misc/drivers/transformer_cu_v1_0/src/xtransformer_cu_hw.h"
#endif
#include TRANSFORMER_CU_HW_HEADER

// ============================================================================
//  Model geometry -- MUST match transformer_cu/mha_forward.h
// ============================================================================
namespace llama_hw {

constexpr int MODEL_ELEMENTS       = 768;
constexpr int MODEL_HIDDEN_DIM     = 2048;
constexpr int MODEL_NUM_HEADS      = 12;
constexpr int MODEL_NUM_LAYERS     = 12;
constexpr int MODEL_TOKENS         = 32000;
constexpr int MODEL_SEQUENCE_LEN   = 1024;
constexpr int MODEL_SCALING_FACTOR = 64;   // quantization group size (GS)

// The kernel hardcodes top-p to 0.9 in ss_final(); the host -p flag is ignored.
constexpr float KERNEL_TOPP = 0.9f;

// ---------------------------------------------------------------------------
//  xrt::kernel argument indices -- positional, taken from the C signature of
//  transformer_cu() in transformer_kernel.cpp (non-__DEBUG__ build).
//  Verify with:  xclbinutil --info -i <xclbin> | grep -A3 transformer_cu
// ---------------------------------------------------------------------------
enum KArg : int {
    ARG_TOKENS        = 0,   // fdata_v_t*  dequantized fp32 embedding table
    ARG_W_SF_0        = 1,   // mfdata_v_t* weight scale factors, bank 0
    ARG_W_0           = 2,   // idata_v_t*  int8 weights, bank 0
    ARG_W_SF_1        = 3,
    ARG_W_1           = 4,
    ARG_WEIGHTS       = 5,   // fdata_v_t*  rmsnorm weights
    ARG_KEY_CACHE     = 6,
    ARG_VALUE_CACHE   = 7,
    ARG_POS           = 8,
    ARG_QKV_W         = 9,
    ARG_QKV_SF_W      = 10,
    ARG_OUT_W         = 11,
    ARG_OUT_SF_W      = 12,
    ARG_FF_W1W3_W     = 13,
    ARG_FF_W1W3_SF_W  = 14,
    ARG_FF_W2_W       = 15,
    ARG_FF_W2_SF_W    = 16,
    ARG_EMBED_W       = 17,
    ARG_EMBED_SF_W    = 18,
    ARG_RMS_ATT_W     = 19,
    ARG_RMS_FFN_W     = 20,
    ARG_RMS_FINAL_W   = 21,
    ARG_CURR_TOKEN    = 22,  // int* array, depth MODEL_SEQUENCE_LEN
    ARG_TEMPERATURE   = 23,
    ARG_COIN          = 24,
    ARG_INIT_RMS_FLAG = 25,
    ARG_PREFILL_FLAG  = 26,
    ARG_COUNT         = 27
};

} // namespace llama_hw

// ============================================================================
//  Structs kept for source compatibility with fastforward.cpp
// ============================================================================
typedef struct {
    int POS;
    int N_DIM;
    int M_DIM;
    int QKV_W;
    int QKV_sf_W;
    int Out_W;
    int Out_sf_W;
    int FF_w1w3_W;
    int FF_w1w3_sf_W;
    int FF_w2_W;
    int FF_w2_sf_W;
    int Embed_W;
    int Embed_sf_W;
    int rms_att_W;
    int rms_ffn_W;
    int rms_final_W;
    float temperature  = 0.9f;
    float coin         = 0.0f;
    bool  init_rms_flag = true;
    bool  prefill_flag  = true;
} axi_reg;

typedef struct {
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int seq_len;
} Config;

typedef struct {
    Config  config;
    int     fd;
    float*  data;
    ssize_t file_size;
} Transformer;

// ============================================================================
//  Checkpoint parsing -- shared by both engines
// ============================================================================
namespace llama_ckpt {

struct Sizes {
    size_t q_size;            // int8 blob   (embed + all layer weights)
    size_t sf_size;           // fp32 scale-factor blob
    size_t rms_size;          // fp32 rmsnorm weights
    size_t embed_float_size;  // dequantized fp32 embedding table
    size_t cache_size;        // one KV cache
    size_t curr_token_size;   // int array indexed by POS
};

inline Sizes sizes() {
    using namespace llama_hw;
    Sizes s{};
    s.q_size = (size_t)MODEL_ELEMENTS *
               ((size_t)(MODEL_ELEMENTS * 4 + MODEL_HIDDEN_DIM * 3) * MODEL_NUM_LAYERS
                + MODEL_TOKENS) * sizeof(int8_t);
    s.sf_size          = s.q_size * sizeof(float) / (sizeof(int8_t) * MODEL_SCALING_FACTOR);
    s.rms_size         = (size_t)MODEL_ELEMENTS * (MODEL_NUM_LAYERS * 2 + 1) * sizeof(float);
    s.embed_float_size = (size_t)MODEL_ELEMENTS * MODEL_TOKENS * sizeof(float);
    s.cache_size       = (size_t)MODEL_ELEMENTS * MODEL_SEQUENCE_LEN * MODEL_NUM_LAYERS * sizeof(float);
    s.curr_token_size  = (size_t)MODEL_SEQUENCE_LEN * sizeof(int);
    return s;
}

// Reads the 256-byte v2 header and sanity-checks it against the geometry the
// kernel was synthesized for. Throws on any mismatch, because every one of
// these would silently produce garbage rather than fail loudly.
inline void read_header(const std::string& path, Config& cfg,
                        int& group_size, bool& shared_classifier) {
    using namespace llama_hw;
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open checkpoint: " + path);

    uint32_t magic = 0;
    int      version = 0;
    uint8_t  shared = 0;
    int      gs = 0;

    f.read(reinterpret_cast<char*>(&magic),   sizeof(magic));
    f.read(reinterpret_cast<char*>(&version), sizeof(version));
    f.read(reinterpret_cast<char*>(&cfg),     sizeof(Config));
    f.read(reinterpret_cast<char*>(&shared),  sizeof(shared));
    f.read(reinterpret_cast<char*>(&gs),      sizeof(gs));
    if (!f) throw std::runtime_error("truncated checkpoint header: " + path);

    if (magic != 0x616b3432u) throw std::runtime_error("bad magic number in checkpoint");
    if (version != 2)         throw std::runtime_error("checkpoint version != 2 (need the runq.c export)");

    group_size        = gs;
    shared_classifier = (shared != 0);

    auto bad = [&](const char* what, long long got, long long want) {
        std::ostringstream os;
        os << "checkpoint/kernel mismatch: " << what << " = " << got
           << ", kernel was built for " << want;
        throw std::runtime_error(os.str());
    };
    if (cfg.dim        != MODEL_ELEMENTS)     bad("dim",        cfg.dim,        MODEL_ELEMENTS);
    if (cfg.hidden_dim != MODEL_HIDDEN_DIM)   bad("hidden_dim", cfg.hidden_dim, MODEL_HIDDEN_DIM);
    if (cfg.n_layers   != MODEL_NUM_LAYERS)   bad("n_layers",   cfg.n_layers,   MODEL_NUM_LAYERS);
    if (cfg.n_heads    != MODEL_NUM_HEADS)    bad("n_heads",    cfg.n_heads,    MODEL_NUM_HEADS);
    if (cfg.n_kv_heads != cfg.n_heads)        bad("n_kv_heads", cfg.n_kv_heads, cfg.n_heads);
    if (cfg.vocab_size != MODEL_TOKENS)       bad("vocab_size", cfg.vocab_size, MODEL_TOKENS);
    if (cfg.seq_len    >  MODEL_SEQUENCE_LEN) bad("seq_len",    cfg.seq_len,    MODEL_SEQUENCE_LEN);
    if (gs             != MODEL_SCALING_FACTOR) bad("group_size", gs, MODEL_SCALING_FACTOR);

    // The kernel drives the LM head from Embed_W / Embed_sf_W, i.e. it assumes
    // wcls aliases q_tokens. An unshared classifier would need its own offsets.
    if (!shared_classifier)
        throw std::runtime_error("checkpoint has an unshared classifier; the kernel "
                                 "reuses Embed_W for the LM head");
}

// Repacks the checkpoint into the three device blobs the kernel expects and
// fills in the byte offsets. Layout produced:
//   q_ptr  : [embed][L0: wq wk wv][L1: ...] ... [all wo] [L0: w1 w3] ... [all w2]
//   sf_ptr : same order, fp32 scales
//   rms_ptr: [rms_att x12][rms_ffn x12][rms_final]
inline void parse_weights(const std::string& path, char* q_ptr, char* sf_ptr,
                          char* rms_ptr, axi_reg& tt) {
    using namespace llama_hw;
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot open checkpoint: " + path);

    const size_t nn_size    = (size_t)MODEL_ELEMENTS * MODEL_ELEMENTS;
    const size_t nm_size    = (size_t)MODEL_ELEMENTS * MODEL_HIDDEN_DIM;
    const size_t nn_sf_size = nn_size * sizeof(float) / MODEL_SCALING_FACTOR;
    const size_t nm_sf_size = nm_size * sizeof(float) / MODEL_SCALING_FACTOR;

    const size_t rms_att_size   = (size_t)MODEL_ELEMENTS * MODEL_NUM_LAYERS * sizeof(float);
    const size_t rms_ffn_size   = rms_att_size;
    const size_t rms_final_size = (size_t)MODEL_ELEMENTS * sizeof(float);

    const size_t embed_size    = (size_t)MODEL_ELEMENTS * MODEL_TOKENS * sizeof(int8_t);
    const size_t embed_sf_size = (size_t)MODEL_ELEMENTS * MODEL_TOKENS * sizeof(float)
                                 / MODEL_SCALING_FACTOR;

    size_t file_ptr = 256;                       // past the v2 header
    size_t rms_idx = 0, q_idx = 0, sf_idx = 0;
    file.seekg(file_ptr, std::ios::beg);

    // ---- fp32 rmsnorm weights -------------------------------------------
    tt.rms_att_W = 0;
    file.read(rms_ptr + rms_idx, rms_att_size);   rms_idx += rms_att_size;

    tt.rms_ffn_W = tt.rms_att_W + (int)rms_att_size;
    file.read(rms_ptr + rms_idx, rms_ffn_size);   rms_idx += rms_ffn_size;

    tt.rms_final_W = tt.rms_ffn_W + (int)rms_ffn_size;
    file.read(rms_ptr + rms_idx, rms_final_size);
    file_ptr = file.tellg();

    // ---- token embedding (also the shared classifier) --------------------
    tt.Embed_W = 0;
    file.read(q_ptr + q_idx, embed_size);         q_idx  += embed_size;

    tt.Embed_sf_W = 0;
    file.read(sf_ptr + sf_idx, embed_sf_size);    sf_idx += embed_sf_size;

    // ---- wq/wk/wv, de-interleaved so each layer's QKV is contiguous ------
    tt.QKV_W    = (int)q_idx;
    tt.QKV_sf_W = (int)sf_idx;
    file_ptr = file_ptr + embed_size + embed_sf_size;

    for (int i = 0; i < MODEL_NUM_LAYERS; i++) {
        for (int j = 0; j < 3; j++) {
            file.seekg(file_ptr + (size_t)j * (nn_size + nn_sf_size) * MODEL_NUM_LAYERS,
                       std::ios::beg);
            file.read(q_ptr  + q_idx,  nn_size);     q_idx  += nn_size;
            file.read(sf_ptr + sf_idx, nn_sf_size);  sf_idx += nn_sf_size;
        }
        file_ptr += (nn_size + nn_sf_size);
    }

    // ---- wo --------------------------------------------------------------
    tt.Out_W    = (int)q_idx;
    tt.Out_sf_W = (int)sf_idx;
    for (int i = 0; i < MODEL_NUM_LAYERS; i++) {
        file.read(q_ptr  + q_idx,  nn_size);     q_idx  += nn_size;
        file.read(sf_ptr + sf_idx, nn_sf_size);  sf_idx += nn_sf_size;
    }
    file_ptr = file.tellg();

    // ---- w1 and w3 interleaved per layer, hopping over the w2 block ------
    tt.FF_w1w3_W    = (int)q_idx;
    tt.FF_w1w3_sf_W = (int)sf_idx;
    for (int i = 0; i < MODEL_NUM_LAYERS; i++) {
        for (int j = 0; j < 2; j++) {
            file.seekg(file_ptr + (size_t)j * 2 * (nm_size + nm_sf_size) * MODEL_NUM_LAYERS,
                       std::ios::beg);
            file.read(q_ptr  + q_idx,  nm_size);     q_idx  += nm_size;
            file.read(sf_ptr + sf_idx, nm_sf_size);  sf_idx += nm_sf_size;
        }
        file_ptr += (nm_size + nm_sf_size);
    }

    // ---- w2 ---------------------------------------------------------------
    tt.FF_w2_W    = (int)q_idx;
    tt.FF_w2_sf_W = (int)sf_idx;
    file.seekg(file_ptr, std::ios::beg);
    for (int i = 0; i < MODEL_NUM_LAYERS; i++) {
        file.read(q_ptr  + q_idx,  nm_size);     q_idx  += nm_size;
        file.read(sf_ptr + sf_idx, nm_sf_size);  sf_idx += nm_sf_size;
    }
    if (!file) throw std::runtime_error("checkpoint ran short while loading weights");

    tt.N_DIM = MODEL_ELEMENTS;
    tt.M_DIM = MODEL_ELEMENTS;

    const Sizes sz = sizes();
    if (q_idx != sz.q_size || sf_idx != sz.sf_size)
        throw std::runtime_error("weight repack size mismatch -- check the layout constants");
}

// Dequantizes the embedding table straight out of the device blobs that
// parse_weights() just filled. Embed_W and Embed_sf_W are both 0, so the int8
// rows and their scales are already at the front of q_ptr / sf_ptr -- no need
// to re-read 98 MB from disk into a temporary.
inline void dequantize_embeddings(const char* q_ptr, const char* sf_ptr, float* out) {
    using namespace llama_hw;
    const int8_t* q = reinterpret_cast<const int8_t*>(q_ptr);
    const float*  s = reinterpret_cast<const float*>(sf_ptr);
    const size_t  n = (size_t)MODEL_ELEMENTS * MODEL_TOKENS;
    for (size_t i = 0; i < n; i++)
        out[i] = static_cast<float>(q[i]) * s[i / MODEL_SCALING_FACTOR];
}

} // namespace llama_ckpt


// ============================================================================
//  FastForward -- bare-metal xrt::ip flow
//
//  XRT knows nothing about argument/bank connectivity here, so device
//  addresses go into the AXI-Lite offset registers by hand and the memory
//  group for every xrt::bo is chosen by the caller.
// ============================================================================
class FastForward {
public:
    // mem_group: XRT memory bank index for every buffer. Check `xrt-smi examine
    // -r memory` -- if your CMA reservation lives in high DDR you probably want
    // bank 1, not 0.
    FastForward(int device_index,
                const std::string& binaryFile,
                const std::string& checkpoint,
                int mem_group = 0,
                int alt_mem_group = -1)     // -1 => share one weight blob
        : grp_(mem_group)
        , alt_grp_(alt_mem_group < 0 ? mem_group : alt_mem_group)
        , split_banks_(alt_mem_group >= 0)
    {
        device = xrt::device(device_index);
        std::cout << "device name: " << device.get_info<xrt::info::device::name>() << "\n";
        std::cout << "device bdf:  " << device.get_info<xrt::info::device::bdf>() << "\n";
        std::cout << "compiled     " << __DATE__ << " " << __TIME__ << "\n";

        uuid = device.load_xclbin(binaryFile);
        transformer_ip = xrt::ip(device, uuid, "transformer_cu");

        llama_ckpt::read_header(checkpoint, cfg_, group_size_, shared_classifier_);
        allocate();
        load_weights(checkpoint);
        program_static_registers();

        std::cout << "FastForward (xrt::ip) ready.\n";
    }

    // ---- configuration --------------------------------------------------
    void set_temperature(float t) {
        tt.temperature = t;
        uint32_t bits; std::memcpy(&bits, &t, sizeof(bits));
        transformer_ip.write_register(XTRANSFORMER_CU_CONTROL_ADDR_TEMPERATURE_DATA, bits);
    }

    void set_rms_flag(bool x) {
        tt.init_rms_flag = x;
        transformer_ip.write_register(XTRANSFORMER_CU_CONTROL_ADDR_INIT_RMS_FLAG_DATA, x ? 1u : 0u);
    }

    // prefill_flag == 1 runs the truncated FSM (4*L-2 steps, no LM head).
    // prefill_flag == 0 runs the full 4*L+1 steps and samples.
    void enable_prefill() { set_prefill(true); }
    void enable_decode()  { set_prefill(false); }

    // ---- curr_token array -------------------------------------------------
    void seed_prompt(const int* toks, int n) {
        if (n < 0 || n > llama_hw::MODEL_SEQUENCE_LEN)
            throw std::runtime_error("prompt longer than MODEL_SEQUENCE_LEN");
        std::memcpy(curr_token_map_, toks, (size_t)n * sizeof(int));
        for (int i = n; i < llama_hw::MODEL_SEQUENCE_LEN; i++) curr_token_map_[i] = -1;
    }
    void set_token(int pos, int tok) { bounds(pos); curr_token_map_[pos] = tok; }
    int  get_token(int pos) const    { bounds(pos); return curr_token_map_[pos]; }

    // ---- execution ---------------------------------------------------------
    void startForward(int pos, float coin) {
        bounds(pos);
        curr_token_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        transformer_ip.write_register(XTRANSFORMER_CU_CONTROL_ADDR_POS_R_DATA,
                                      static_cast<uint32_t>(pos));
        uint32_t bits; std::memcpy(&bits, &coin, sizeof(bits));
        transformer_ip.write_register(XTRANSFORMER_CU_CONTROL_ADDR_COIN_DATA, bits);

        // Plain write, not read-modify-write: bit0 = ap_start, and we do not
        // want to accidentally re-arm auto_restart (bit 7).
        transformer_ip.write_register(XTRANSFORMER_CU_CONTROL_ADDR_AP_CTRL, 0x1u);
    }

    // Returns curr_token[pos+1]: the token the kernel just sampled, or (during
    // prefill, where the kernel does not write) the prompt token already there.
    int endForward(int pos) {
        bounds(pos + 1);
        wait_done();
        curr_token_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        return curr_token_map_[pos + 1];
    }

    int forward(int pos, float coin) { startForward(pos, coin); return endForward(pos); }

    // ---- info ---------------------------------------------------------------
    const Config&  config() const { return cfg_; }
    const axi_reg& regs()   const { return tt; }

private:
    static void bounds(int pos) {
        if (pos < 0 || pos >= llama_hw::MODEL_SEQUENCE_LEN)
            throw std::runtime_error("pos out of range for curr_token[]");
    }

    void set_prefill(bool x) {
        tt.prefill_flag = x;
        transformer_ip.write_register(XTRANSFORMER_CU_CONTROL_ADDR_PREFILL_FLAG_DATA, x ? 1u : 0u);
    }

    void wait_done(int timeout_ms = 60000) {
        using clk = std::chrono::steady_clock;
        const auto deadline = clk::now() + std::chrono::milliseconds(timeout_ms);
        for (;;) {
            // ap_done is clear-on-read, so break out immediately once seen.
            if (transformer_ip.read_register(XTRANSFORMER_CU_CONTROL_ADDR_AP_CTRL) & 0x2u) return;
            if (clk::now() > deadline)
                throw std::runtime_error("transformer_cu timed out waiting for ap_done");
            std::this_thread::yield();
        }
    }

    void write_bo_address(uint32_t offset, xrt::bo& bo) {
        const uint64_t addr = bo.address();
        transformer_ip.write_register(offset,     static_cast<uint32_t>(addr & 0xFFFFFFFFull));
        transformer_ip.write_register(offset + 4, static_cast<uint32_t>(addr >> 32));
    }

    void allocate() {
        const auto sz = llama_ckpt::sizes();

        embed_bo      = xrt::bo(device, sz.embed_float_size, grp_);
        parent_rms_bo = xrt::bo(device, sz.rms_size,         grp_);
        parent_w_bo   = xrt::bo(device, sz.q_size,           grp_);
        parent_sf_bo  = xrt::bo(device, sz.sf_size,          grp_);
        key_cache_bo  = xrt::bo(device, sz.cache_size,       grp_);
        value_cache_bo= xrt::bo(device, sz.cache_size,       grp_);
        curr_token_bo = xrt::bo(device, sz.curr_token_size,  grp_);

        if (split_banks_) {
            // Second physical copy of the weight blob so the two GeMV threads
            // pull from different DDR controllers instead of colliding on the
            // same banks. Costs ~116 MB.
            alt_w_bo  = xrt::bo(device, sz.q_size,  alt_grp_);
            alt_sf_bo = xrt::bo(device, sz.sf_size, alt_grp_);
        }

        curr_token_map_ = curr_token_bo.map<int*>();
        for (int i = 0; i < llama_hw::MODEL_SEQUENCE_LEN; i++) curr_token_map_[i] = -1;
    }

    void load_weights(const std::string& checkpoint) {
        char*  q_ptr   = parent_w_bo.map<char*>();
        char*  sf_ptr  = parent_sf_bo.map<char*>();
        char*  rms_ptr = parent_rms_bo.map<char*>();
        float* emb_ptr = embed_bo.map<float*>();

        llama_ckpt::parse_weights(checkpoint, q_ptr, sf_ptr, rms_ptr, tt);
        llama_ckpt::dequantize_embeddings(q_ptr, sf_ptr, emb_ptr);

        const auto sz = llama_ckpt::sizes();
        if (split_banks_) {
            std::memcpy(alt_w_bo.map<char*>(),  q_ptr,  sz.q_size);
            std::memcpy(alt_sf_bo.map<char*>(), sf_ptr, sz.sf_size);
            alt_w_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            alt_sf_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        }

        parent_rms_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        parent_sf_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        parent_w_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        embed_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        curr_token_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        // KV caches are written before they are read (mha_WAR_store_load writes
        // position POS before any later position reads it), so no init needed.
    }

    void program_static_registers() {
        write_bo_address(XTRANSFORMER_CU_CONTROL_ADDR_TOKENS_DATA,      embed_bo);
        write_bo_address(XTRANSFORMER_CU_CONTROL_ADDR_W_SF_0_DATA,      parent_sf_bo);
        write_bo_address(XTRANSFORMER_CU_CONTROL_ADDR_W_0_DATA,         parent_w_bo);
        write_bo_address(XTRANSFORMER_CU_CONTROL_ADDR_W_SF_1_DATA,
                         split_banks_ ? alt_sf_bo : parent_sf_bo);
        write_bo_address(XTRANSFORMER_CU_CONTROL_ADDR_W_1_DATA,
                         split_banks_ ? alt_w_bo  : parent_w_bo);
        write_bo_address(XTRANSFORMER_CU_CONTROL_ADDR_WEIGHTS_DATA,     parent_rms_bo);
        write_bo_address(XTRANSFORMER_CU_CONTROL_ADDR_KEY_CACHE_DATA,   key_cache_bo);
        write_bo_address(XTRANSFORMER_CU_CONTROL_ADDR_VALUE_CACHE_DATA, value_cache_bo);
        write_bo_address(XTRANSFORMER_CU_CONTROL_ADDR_CURR_TOKEN_DATA,  curr_token_bo);

        auto w32 = [&](uint32_t off, int v) {
            transformer_ip.write_register(off, static_cast<uint32_t>(v));
        };
        w32(XTRANSFORMER_CU_CONTROL_ADDR_QKV_W_DATA,        tt.QKV_W);
        w32(XTRANSFORMER_CU_CONTROL_ADDR_QKV_SF_W_DATA,     tt.QKV_sf_W);
        w32(XTRANSFORMER_CU_CONTROL_ADDR_OUT_W_DATA,        tt.Out_W);
        w32(XTRANSFORMER_CU_CONTROL_ADDR_OUT_SF_W_DATA,     tt.Out_sf_W);
        w32(XTRANSFORMER_CU_CONTROL_ADDR_FF_W1W3_W_DATA,    tt.FF_w1w3_W);
        w32(XTRANSFORMER_CU_CONTROL_ADDR_FF_W1W3_SF_W_DATA, tt.FF_w1w3_sf_W);
        w32(XTRANSFORMER_CU_CONTROL_ADDR_FF_W2_W_DATA,      tt.FF_w2_W);
        w32(XTRANSFORMER_CU_CONTROL_ADDR_FF_W2_SF_W_DATA,   tt.FF_w2_sf_W);
        w32(XTRANSFORMER_CU_CONTROL_ADDR_EMBED_W_DATA,      tt.Embed_W);
        w32(XTRANSFORMER_CU_CONTROL_ADDR_EMBED_SF_W_DATA,   tt.Embed_sf_W);
        w32(XTRANSFORMER_CU_CONTROL_ADDR_RMS_ATT_W_DATA,    tt.rms_att_W);
        w32(XTRANSFORMER_CU_CONTROL_ADDR_RMS_FFN_W_DATA,    tt.rms_ffn_W);
        w32(XTRANSFORMER_CU_CONTROL_ADDR_RMS_FINAL_W_DATA,  tt.rms_final_W);

        set_temperature(tt.temperature);
        set_rms_flag(tt.init_rms_flag);
        set_prefill(tt.prefill_flag);
    }

    xrt::device device;
    xrt::uuid   uuid;
    xrt::ip     transformer_ip;

    xrt::bo embed_bo, parent_rms_bo, parent_w_bo, parent_sf_bo;
    xrt::bo alt_w_bo, alt_sf_bo;
    xrt::bo key_cache_bo, value_cache_bo, curr_token_bo;
    int*    curr_token_map_ = nullptr;

    axi_reg tt{};
    Config  cfg_{};
    int     group_size_ = 0;
    bool    shared_classifier_ = false;

    int  grp_, alt_grp_;
    bool split_banks_;
};


// ============================================================================
//  RunForward -- managed xrt::kernel / xrt::run flow
//
//  Same API as FastForward. XRT resolves argument->bank connectivity from the
//  xclbin, so buffers land in the right DDR automatically and no register
//  offsets are hand-written.
// ============================================================================
class RunForward {
public:
    RunForward(int device_index,
               const std::string& binaryFile,
               const std::string& checkpoint)
    {
        device = xrt::device(device_index);
        std::cout << "device name: " << device.get_info<xrt::info::device::name>() << "\n";
        std::cout << "device bdf:  " << device.get_info<xrt::info::device::bdf>() << "\n";
        std::cout << "compiled     " << __DATE__ << " " << __TIME__ << "\n";

        uuid   = device.load_xclbin(binaryFile);
        kernel = xrt::kernel(device, uuid, "transformer_cu",
                             xrt::kernel::cu_access_mode::exclusive);
        run    = xrt::run(kernel);

        llama_ckpt::read_header(checkpoint, cfg_, group_size_, shared_classifier_);
        allocate();
        load_weights(checkpoint);
        bind_static_args();

        std::cout << "RunForward (xrt::kernel) ready.\n";
    }

    // ---- configuration ----------------------------------------------------
    void set_temperature(float t) {
        tt.temperature = t;
        run.set_arg(llama_hw::ARG_TEMPERATURE, t);
    }

    void set_rms_flag(bool x) {
        tt.init_rms_flag = x;
        run.set_arg(llama_hw::ARG_INIT_RMS_FLAG, x);
    }

    void enable_prefill() { set_prefill(true); }
    void enable_decode()  { set_prefill(false); }

    // ---- curr_token array --------------------------------------------------
    void seed_prompt(const int* toks, int n) {
        if (n < 0 || n > llama_hw::MODEL_SEQUENCE_LEN)
            throw std::runtime_error("prompt longer than MODEL_SEQUENCE_LEN");
        std::memcpy(curr_token_map_, toks, (size_t)n * sizeof(int));
        for (int i = n; i < llama_hw::MODEL_SEQUENCE_LEN; i++) curr_token_map_[i] = -1;
    }
    void set_token(int pos, int tok) { bounds(pos); curr_token_map_[pos] = tok; }
    int  get_token(int pos) const    { bounds(pos); return curr_token_map_[pos]; }

    // ---- execution -----------------------------------------------------------
    void startForward(int pos, float coin) {
        bounds(pos);
        curr_token_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        run.set_arg(llama_hw::ARG_POS,  pos);
        run.set_arg(llama_hw::ARG_COIN, coin);
        run.start();
    }

    int endForward(int pos) {
        bounds(pos + 1);
        auto state = run.wait(std::chrono::milliseconds(60000));
        if (state != ERT_CMD_STATE_COMPLETED)
            throw std::runtime_error("transformer_cu run did not complete (timeout or error)");
        curr_token_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        return curr_token_map_[pos + 1];
    }

    int forward(int pos, float coin) { startForward(pos, coin); return endForward(pos); }

    // ---- info -----------------------------------------------------------------
    const Config&  config() const { return cfg_; }
    const axi_reg& regs()   const { return tt; }

private:
    static void bounds(int pos) {
        if (pos < 0 || pos >= llama_hw::MODEL_SEQUENCE_LEN)
            throw std::runtime_error("pos out of range for curr_token[]");
    }

    void set_prefill(bool x) {
        tt.prefill_flag = x;
        run.set_arg(llama_hw::ARG_PREFILL_FLAG, x);
    }

    void allocate() {
        using namespace llama_hw;
        const auto sz = llama_ckpt::sizes();

        embed_bo       = xrt::bo(device, sz.embed_float_size, kernel.group_id(ARG_TOKENS));
        parent_rms_bo  = xrt::bo(device, sz.rms_size,         kernel.group_id(ARG_WEIGHTS));
        parent_w_bo    = xrt::bo(device, sz.q_size,           kernel.group_id(ARG_W_0));
        parent_sf_bo   = xrt::bo(device, sz.sf_size,          kernel.group_id(ARG_W_SF_0));
        key_cache_bo   = xrt::bo(device, sz.cache_size,       kernel.group_id(ARG_KEY_CACHE));
        value_cache_bo = xrt::bo(device, sz.cache_size,       kernel.group_id(ARG_VALUE_CACHE));
        curr_token_bo  = xrt::bo(device, sz.curr_token_size,  kernel.group_id(ARG_CURR_TOKEN));

        // w_0/w_1 and w_sf_0/w_sf_1 are separate m_axi bundles. If the platform
        // maps them to different banks we must hold a copy in each; if they land
        // on the same bank one buffer serves both arguments.
        split_w_  = (kernel.group_id(ARG_W_1)    != kernel.group_id(ARG_W_0));
        split_sf_ = (kernel.group_id(ARG_W_SF_1) != kernel.group_id(ARG_W_SF_0));
        if (split_w_)  alt_w_bo  = xrt::bo(device, sz.q_size,  kernel.group_id(ARG_W_1));
        if (split_sf_) alt_sf_bo = xrt::bo(device, sz.sf_size, kernel.group_id(ARG_W_SF_1));

        std::cout << "weight banks: w_0=" << kernel.group_id(ARG_W_0)
                  << " w_1="   << kernel.group_id(ARG_W_1)
                  << " sf_0="  << kernel.group_id(ARG_W_SF_0)
                  << " sf_1="  << kernel.group_id(ARG_W_SF_1)
                  << (split_w_ ? "  (duplicating weight blob)" : "  (shared weight blob)")
                  << "\n";

        curr_token_map_ = curr_token_bo.map<int*>();
        for (int i = 0; i < MODEL_SEQUENCE_LEN; i++) curr_token_map_[i] = -1;
    }

    void load_weights(const std::string& checkpoint) {
        char*  q_ptr   = parent_w_bo.map<char*>();
        char*  sf_ptr  = parent_sf_bo.map<char*>();
        char*  rms_ptr = parent_rms_bo.map<char*>();
        float* emb_ptr = embed_bo.map<float*>();

        llama_ckpt::parse_weights(checkpoint, q_ptr, sf_ptr, rms_ptr, tt);
        llama_ckpt::dequantize_embeddings(q_ptr, sf_ptr, emb_ptr);

        const auto sz = llama_ckpt::sizes();
        if (split_w_) {
            std::memcpy(alt_w_bo.map<char*>(), q_ptr, sz.q_size);
            alt_w_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        }
        if (split_sf_) {
            std::memcpy(alt_sf_bo.map<char*>(), sf_ptr, sz.sf_size);
            alt_sf_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        }

        parent_rms_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        parent_sf_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        parent_w_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        embed_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        curr_token_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }

    void bind_static_args() {
        using namespace llama_hw;
        run.set_arg(ARG_TOKENS,       embed_bo);
        run.set_arg(ARG_W_SF_0,       parent_sf_bo);
        run.set_arg(ARG_W_0,          parent_w_bo);
        run.set_arg(ARG_W_SF_1,       split_sf_ ? alt_sf_bo : parent_sf_bo);
        run.set_arg(ARG_W_1,          split_w_  ? alt_w_bo  : parent_w_bo);
        run.set_arg(ARG_WEIGHTS,      parent_rms_bo);
        run.set_arg(ARG_KEY_CACHE,    key_cache_bo);
        run.set_arg(ARG_VALUE_CACHE,  value_cache_bo);
        run.set_arg(ARG_CURR_TOKEN,   curr_token_bo);

        run.set_arg(ARG_QKV_W,        tt.QKV_W);
        run.set_arg(ARG_QKV_SF_W,     tt.QKV_sf_W);
        run.set_arg(ARG_OUT_W,        tt.Out_W);
        run.set_arg(ARG_OUT_SF_W,     tt.Out_sf_W);
        run.set_arg(ARG_FF_W1W3_W,    tt.FF_w1w3_W);
        run.set_arg(ARG_FF_W1W3_SF_W, tt.FF_w1w3_sf_W);
        run.set_arg(ARG_FF_W2_W,      tt.FF_w2_W);
        run.set_arg(ARG_FF_W2_SF_W,   tt.FF_w2_sf_W);
        run.set_arg(ARG_EMBED_W,      tt.Embed_W);
        run.set_arg(ARG_EMBED_SF_W,   tt.Embed_sf_W);
        run.set_arg(ARG_RMS_ATT_W,    tt.rms_att_W);
        run.set_arg(ARG_RMS_FFN_W,    tt.rms_ffn_W);
        run.set_arg(ARG_RMS_FINAL_W,  tt.rms_final_W);

        // POS and coin are rewritten every step; seed them so the first
        // start() never launches with an unset argument.
        run.set_arg(ARG_POS,  0);
        run.set_arg(ARG_COIN, 0.0f);

        set_temperature(tt.temperature);
        set_rms_flag(tt.init_rms_flag);
        set_prefill(tt.prefill_flag);
    }

    xrt::device device;
    xrt::uuid   uuid;
    xrt::kernel kernel;
    xrt::run    run;

    xrt::bo embed_bo, parent_rms_bo, parent_w_bo, parent_sf_bo;
    xrt::bo alt_w_bo, alt_sf_bo;
    xrt::bo key_cache_bo, value_cache_bo, curr_token_bo;
    int*    curr_token_map_ = nullptr;

    axi_reg tt{};
    Config  cfg_{};
    int     group_size_ = 0;
    bool    shared_classifier_ = false;
    bool    split_w_  = false;
    bool    split_sf_ = false;
};

#endif // COMPUTE_UNITS_H
