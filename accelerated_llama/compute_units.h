// ============================================================================
//  compute_units.h  --  host driver for the transformer_cu kernel (xrt::kernel)
//
//  One class, TransformerCU, owns every device buffer and the xrt::run handle.
//  All knowledge of the memory image (weight striping, rms offsets, sizes)
//  lives in llama_layout.h, which has no XRT dependency so the HLS testbench
//  can share it.
//
//  Per-step protocol (same in prefill and decode):
//      seed_prompt(tokens, n)        once: write prompt into curr_token[]
//      startForward(pos, coin)       push curr_token[], set POS/coin, launch
//      endForward(pos)               wait, pull curr_token[], return [pos+1]
//
//  In prefill (prefill_flag = 1) the kernel stops after the last layer's O
//  projection and does NOT write curr_token[pos+1], so endForward() returns
//  the prompt token the host already seeded there. The loop is identical.
// ============================================================================
#ifndef COMPUTE_UNITS_H
#define COMPUTE_UNITS_H

#include "llama_layout.h"

#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>
#include <xrt/experimental/xrt_xclbin.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

namespace llama {

// ----------------------------------------------------------------------------
//  Kernel argument indices: positional, from transformer_cu()'s C signature
//  (non-__DEBUG__ build). Cross-check against the generated driver header
//  xtransformer_cu_hw.h -- the register order there is the argument order.
// ----------------------------------------------------------------------------
enum KernelArg : int {
    kArgTokens      = 0,   // fdata_v_t*   fp32 embedding table   [vocab][dim]
    kArgW0          = 1,   // wide_t*      striped weight blob    (port 0)
    kArgW1          = 2,   // wide_t*      striped weight blob    (port 1)
    kArgRmsWeights  = 3,   // fdata_v_t*   rms weights            [att|ffn|final]
    kArgKeyCache    = 4,   // mfdata_v_t*  [layer][head][pos][head_size]
    kArgValueCache  = 5,   // mfdata_v_t*  [layer][head][pos][head_size]
    kArgPos         = 6,   // int
    kArgRmsAttOff   = 7,   // int, byte offset into rms buffer
    kArgRmsFfnOff   = 8,   // int, byte offset into rms buffer
    kArgRmsFinalOff = 9,   // int, byte offset into rms buffer
    kArgCurrToken   = 10,  // int*         [seq_len]
    kArgTemperature = 11,  // float
    kArgCoin        = 12,  // float
    kArgInitRms     = 13,  // bool         load rms weights into URAM
    kArgPrefill     = 14,  // bool         truncated FSM, no LM head
};

class TransformerCU {
public:
    TransformerCU(const std::string& xclbin_path,
                  const std::string& checkpoint_path,
                  int device_index = 0)
        : ck_(scan_checkpoint(checkpoint_path))
    {
        device_ = xrt::device(device_index);
        std::cout << "device:   " << device_.get_info<xrt::info::device::name>() << "\n";

        const auto uuid = device_.register_xclbin(xrt::xclbin(xclbin_path));
        hwctx_  = xrt::hw_context(device_, uuid);
        kernel_ = xrt::kernel(hwctx_, "transformer_cu");
        run_    = xrt::run(kernel_);

        allocate_buffers();
        load_model();
        bind_args();
        std::cout << "transformer_cu ready.\n";
    }

    // ---- per-run settings ----------------------------------------------------
    void set_temperature(float t) { run_.set_arg(kArgTemperature, t); }

    // The rms weights sit in a static URAM array inside the kernel: load them on
    // the first call, then turn this off.
    void set_rms_flag(bool on)    { run_.set_arg(kArgInitRms, static_cast<uint32_t>(on)); }

    void enable_prefill()         { run_.set_arg(kArgPrefill, static_cast<uint32_t>(1)); }
    void enable_decode()          { run_.set_arg(kArgPrefill, static_cast<uint32_t>(0)); }

    // ---- curr_token[] --------------------------------------------------------
    void seed_prompt(const int* tokens, int n) {
        if (n < 0 || n > kSeqLen) throw std::runtime_error("prompt longer than kSeqLen");
        std::memcpy(tokens_, tokens, static_cast<size_t>(n) * sizeof(int));
        for (int i = n; i < kSeqLen; i++) tokens_[i] = -1;
    }
    void set_token(int pos, int tok) { check_pos(pos); tokens_[pos] = tok; }
    int  get_token(int pos) const    { check_pos(pos); return tokens_[pos]; }

    // ---- execution -----------------------------------------------------------
    void startForward(int pos, float coin) {
        check_pos(pos);
        token_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        run_.set_arg(kArgPos,  static_cast<int32_t>(pos));
        run_.set_arg(kArgCoin, coin);
        run_.start();
    }

    // Returns curr_token[pos+1]: the sampled token in decode, or the seeded
    // prompt token in prefill.
    int endForward(int pos, std::chrono::milliseconds timeout = std::chrono::seconds(60)) {
        check_pos(pos + 1);
        if (run_.wait(timeout) != ERT_CMD_STATE_COMPLETED)
            throw std::runtime_error("transformer_cu did not complete (timeout or error)");
        token_bo_.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        return tokens_[pos + 1];
    }

    int forward(int pos, float coin) { startForward(pos, coin); return endForward(pos); }

    // ---- info ----------------------------------------------------------------
    const Checkpoint& checkpoint() const { return ck_; }
    const Config&     config()     const { return ck_.cfg; }

private:
    static void check_pos(int pos) {
        if (pos < 0 || pos >= kSeqLen) throw std::runtime_error("pos out of range for curr_token[]");
    }

    xrt::bo make_bo(size_t bytes, KernelArg arg) {
        return xrt::bo(device_, bytes, kernel_.group_id(arg));
    }

    void allocate_buffers() {
        embed_bo_  = make_bo(kEmbedF32Bytes, kArgTokens);
        w0_bo_     = make_bo(kPackedBytes,   kArgW0);
        rms_bo_    = make_bo(kRmsBytes,      kArgRmsWeights);
        kcache_bo_ = make_bo(kKvCacheBytes,  kArgKeyCache);
        vcache_bo_ = make_bo(kKvCacheBytes,  kArgValueCache);
        token_bo_  = make_bo(kTokenBytes,    kArgCurrToken);

        // w_0 and w_1 read the SAME striped blob (the kernel adds PORT*204 beats
        // itself). Only if the two bundles land in different memory banks does
        // each need its own physical copy.
        const int g0 = kernel_.group_id(kArgW0);
        const int g1 = kernel_.group_id(kArgW1);
        split_weights_ = (g0 != g1);
        if (split_weights_) w1_bo_ = make_bo(kPackedBytes, kArgW1);

        std::cout << "banks:    w_0=" << g0 << " w_1=" << g1
                  << (split_weights_ ? "  (weight blob duplicated per bank)\n"
                                     : "  (one shared weight blob)\n");

        tokens_ = token_bo_.map<int*>();
        for (int i = 0; i < kSeqLen; i++) tokens_[i] = -1;
    }

    void load_model() {
        const auto t0 = std::chrono::steady_clock::now();

        pack_weights(ck_, w0_bo_.map<void*>());
        rms_off_ = load_rms(ck_, rms_bo_.map<void*>());
        load_embedding_f32(ck_, embed_bo_.map<float*>());

        // The kernel only reads positions it has already written, but zeroed
        // caches make a bad run reproducible instead of reading stale CMA.
        std::memset(kcache_bo_.map<void*>(), 0, kKvCacheBytes);
        std::memset(vcache_bo_.map<void*>(), 0, kKvCacheBytes);

        if (split_weights_) std::memcpy(w1_bo_.map<void*>(), w0_bo_.map<void*>(), kPackedBytes);

        for (xrt::bo* bo : {&w0_bo_, &rms_bo_, &embed_bo_, &kcache_bo_, &vcache_bo_, &token_bo_})
            bo->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        if (split_weights_) w1_bo_.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() - t0).count();
        std::cout << "loaded:   " << ck_.path << " (" << kPackedBytes / (1024 * 1024)
                  << " MiB striped weights) in " << ms << " ms\n";
    }

    void bind_args() {
        run_.set_arg(kArgTokens,      embed_bo_);
        run_.set_arg(kArgW0,          w0_bo_);
        run_.set_arg(kArgW1,          split_weights_ ? w1_bo_ : w0_bo_);
        run_.set_arg(kArgRmsWeights,  rms_bo_);
        run_.set_arg(kArgKeyCache,    kcache_bo_);
        run_.set_arg(kArgValueCache,  vcache_bo_);
        run_.set_arg(kArgCurrToken,   token_bo_);

        run_.set_arg(kArgRmsAttOff,   static_cast<int32_t>(rms_off_.att));
        run_.set_arg(kArgRmsFfnOff,   static_cast<int32_t>(rms_off_.ffn));
        run_.set_arg(kArgRmsFinalOff, static_cast<int32_t>(rms_off_.final));

        // Defaults so the first start() never launches with an unset argument.
        run_.set_arg(kArgPos,  static_cast<int32_t>(0));
        run_.set_arg(kArgCoin, 0.0f);
        set_temperature(1.0f);
        set_rms_flag(true);
        enable_decode();
    }

    Checkpoint  ck_;
    RmsOffsets  rms_off_;

    xrt::device     device_;
    xrt::hw_context hwctx_;
    xrt::kernel     kernel_;
    xrt::run        run_;

    xrt::bo embed_bo_, w0_bo_, w1_bo_, rms_bo_, kcache_bo_, vcache_bo_, token_bo_;
    bool    split_weights_ = false;
    int*    tokens_ = nullptr;
};

} // namespace llama

#endif // COMPUTE_UNITS_H