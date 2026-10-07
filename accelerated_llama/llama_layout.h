// ============================================================================
//  llama_layout.h  --  memory image of the transformer_cu kernel
//
//  Everything the host needs to know about how transformer_cu expects its
//  buffers laid out, with NO dependency on XRT and NO dependency on Vitis HLS
//  types. That is what lets the same code run in three places:
//
//    accelerated_llama/compute_units.h     host, fills mapped xrt::bo memory
//    transformer_cu/testbench/*.cpp        csim/cosim, fills std::vectors
//    anywhere else                         plain g++, for offline checks
//
//  If this header is included AFTER transformer_cu/mha_forward.h (the
//  testbench does this), every constant below is static_assert-ed against the
//  kernel's own definitions, so geometry drift is a compile error rather than
//  a silent garbage token.
//
//  ---------------------------------------------------------------------------
//  Weight stream layout ("striping")
//  ---------------------------------------------------------------------------
//  All quantized weights live in ONE packed blob that is bound to both w_0 and
//  w_1. The blob is a sequence of frames:
//
//      frame = [ 12 beats of fp32 scales ][ 192 beats of int8 weights ]
//            = 204 beats x 64 B = 13056 B, covering 12288 weights / 192 scales
//
//  Each GeMV ("stage") is the row-major concatenation of its tensors, cut into
//  frames, split into two contiguous halves, and the halves interleaved:
//
//      stage memory:  f0(p0) f0(p1) f1(p0) f1(p1) ... f(K-1)(p0) f(K-1)(p1)
//      fj(p0) = logical frame j          (first half of the output rows)
//      fj(p1) = logical frame K + j      (second half of the output rows)
//
//  The kernel's mm2ds_get_data<PORT, 2> reads frame i for port PORT at
//      beat  = stage_offset + PORT * 204 + i * 408
//  so w_0 streams rows [0, M/2) and w_1 streams rows [M/2, M).
//
//  Stage order mirrors weight_fsm() in transformer_kernel.cpp:
//      for each layer:  QKV (wq|wk|wv), O (wo), gate+up (w1|w3), down (w2)
//      then once:       classifier (wcls, == token embedding when shared)
// ============================================================================
#ifndef LLAMA_LAYOUT_H
#define LLAMA_LAYOUT_H

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace llama {

// ============================================================================
//  Model geometry  (must match transformer_cu/mha_forward.h)
// ============================================================================
constexpr int kDim      = 768;     // MODEL_ELEMENTS
constexpr int kHidden   = 2048;    // MODEL_HIDDEN_DIM
constexpr int kLayers   = 12;      // MODEL_NUM_LAYERS
constexpr int kHeads    = 12;      // MODEL_NUM_HEADS
constexpr int kVocab    = 32000;   // MODEL_TOKENS
constexpr int kSeqLen   = 1024;    // MODEL_SEQUENCE_LEN
constexpr int kGroup    = 64;      // MODEL_SCALING_FACTOR (quantization group)
constexpr int kHeadSize = kDim / kHeads;

// ss_final() hardcodes top-p; the host has no way to change it.
constexpr float kKernelTopP = 0.9f;

// ============================================================================
//  Weight-stream framing  (must match SF_FRAME / QUANT_FRAME / MAX_DW)
// ============================================================================
constexpr size_t kBeatBytes   = 64;                       // MAX_DW / 8
constexpr int    kSfBeats     = 12;                       // SF_FRAME
constexpr int    kQBeats      = 192;                      // QUANT_FRAME
constexpr int    kFrameBeats  = kSfBeats + kQBeats;       // TXFR_FRAME
constexpr int    kPorts       = 2;                        // w_0, w_1

constexpr size_t kFrameSfBytes = kSfBeats * kBeatBytes;           //   768 B
constexpr size_t kFrameQBytes  = kQBeats  * kBeatBytes;           // 12288 B
constexpr size_t kFrameBytes   = kFrameSfBytes + kFrameQBytes;    // 13056 B
constexpr size_t kFrameScales  = kFrameSfBytes / sizeof(float);   //   192
constexpr size_t kFrameElems   = kFrameQBytes;                    // 12288 int8

static_assert(kFrameScales * kGroup == kFrameElems,
              "a frame's scales must cover exactly that frame's weights");

// ============================================================================
//  Device buffer sizes (bytes)
// ============================================================================
constexpr size_t kQuantElems =                     // every int8 weight in the model
    (size_t)kDim * ((size_t)(kDim * 4 + kHidden * 3) * kLayers + kVocab);
constexpr size_t kPackedBytes   = kQuantElems + kQuantElems / kGroup * sizeof(float);
constexpr size_t kPackedBeats   = kPackedBytes / kBeatBytes;
constexpr size_t kRmsBytes      = (size_t)kDim * (kLayers * 2 + 1) * sizeof(float);
constexpr size_t kEmbedF32Bytes = (size_t)kVocab * kDim * sizeof(float);
constexpr size_t kKvCacheBytes  = (size_t)kLayers * kSeqLen * kDim * sizeof(float);  // ONE cache
constexpr size_t kTokenBytes    = (size_t)kSeqLen * sizeof(int32_t);

static_assert(kPackedBytes % kFrameBytes == 0, "packed blob must be whole frames");

// ---------------------------------------------------------------------------
//  Cross-check against the kernel when compiled next to it.
//  MARK_FORWARD is mha_forward.h's include guard.
// ---------------------------------------------------------------------------
#ifdef MARK_FORWARD
static_assert(kDim     == MODEL_ELEMENTS,       "kDim != MODEL_ELEMENTS");
static_assert(kHidden  == MODEL_HIDDEN_DIM,     "kHidden != MODEL_HIDDEN_DIM");
static_assert(kLayers  == MODEL_NUM_LAYERS,     "kLayers != MODEL_NUM_LAYERS");
static_assert(kHeads   == MODEL_NUM_HEADS,      "kHeads != MODEL_NUM_HEADS");
static_assert(kVocab   == MODEL_TOKENS,         "kVocab != MODEL_TOKENS");
static_assert(kSeqLen  == MODEL_SEQUENCE_LEN,   "kSeqLen != MODEL_SEQUENCE_LEN");
static_assert(kGroup   == MODEL_SCALING_FACTOR, "kGroup != MODEL_SCALING_FACTOR");
static_assert(kBeatBytes  == MAX_DW / 8,        "kBeatBytes != MAX_DW/8");
static_assert(kSfBeats    == SF_FRAME,          "kSfBeats != SF_FRAME");
static_assert(kQBeats     == QUANT_FRAME,       "kQBeats != QUANT_FRAME");
static_assert(kFrameBeats == TXFR_FRAME,        "kFrameBeats != TXFR_FRAME");
#endif

// ============================================================================
//  Checkpoint (llama2.c runq.c "version 2" export)
// ============================================================================
struct Config {                    // byte-for-byte the on-disk header struct
    int dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len;
};

struct Tensor {                    // one quantized tensor inside the file
    size_t q_off  = 0;             // file offset of its int8 block
    size_t sf_off = 0;             // file offset of its fp32 scale block
    size_t n      = 0;             // element count
};

struct Checkpoint {
    std::string path;
    size_t      file_size = 0;
    Config      cfg{};
    int         group_size = 0;
    bool        shared_classifier = false;

    size_t rms_att_off   = 0;      // fp32 [layers][dim]
    size_t rms_ffn_off   = 0;      // fp32 [layers][dim]
    size_t rms_final_off = 0;      // fp32 [dim]

    Tensor embed;                  // q_tokens [vocab][dim]
    Tensor wq[kLayers], wk[kLayers], wv[kLayers], wo[kLayers];
    Tensor w1[kLayers], w2[kLayers], w3[kLayers];
    Tensor wcls;                   // == embed when shared_classifier
};

namespace detail {

class File {
public:
    explicit File(const std::string& path) : path_(path), f_(path, std::ios::binary) {
        if (!f_) throw std::runtime_error("cannot open " + path);
    }

    void read_at(size_t off, void* dst, size_t n, const char* what) {
        f_.clear();
        f_.seekg(static_cast<std::streamoff>(off), std::ios::beg);
        f_.read(static_cast<char*>(dst), static_cast<std::streamsize>(n));
        if (static_cast<size_t>(f_.gcount()) != n) {
            std::ostringstream os;
            os << path_ << ": short read on " << what << " (wanted " << n
               << " B at offset " << off << ", got " << f_.gcount() << ")";
            throw std::runtime_error(os.str());
        }
    }

    size_t size() {
        f_.clear();
        f_.seekg(0, std::ios::end);
        return static_cast<size_t>(f_.tellg());
    }

private:
    std::string   path_;
    std::ifstream f_;
};

// Lays out one tensor at `off` and advances `off` past it: [n int8][n/GS fp32].
inline Tensor take_tensor(size_t& off, size_t n) {
    Tensor t;
    t.q_off  = off;
    t.sf_off = off + n;
    t.n      = n;
    off += n + (n / kGroup) * sizeof(float);
    return t;
}

} // namespace detail

// Parses and validates the header, then computes the file offset of every
// tensor. Throws on any mismatch with the geometry the kernel was built for.
inline Checkpoint scan_checkpoint(const std::string& path) {
    detail::File f(path);
    Checkpoint ck;
    ck.path      = path;
    ck.file_size = f.size();

    uint32_t magic = 0;
    int32_t  version = 0;
    uint8_t  shared = 0;
    int32_t  gs = 0;
    f.read_at(0, &magic,   sizeof(magic),   "header magic");
    f.read_at(4, &version, sizeof(version), "header version");
    f.read_at(8, &ck.cfg,  sizeof(Config),  "header config");
    f.read_at(8 + sizeof(Config),     &shared, sizeof(shared), "header shared flag");
    f.read_at(8 + sizeof(Config) + 1, &gs,     sizeof(gs),     "header group size");

    if (magic != 0x616b3432u) throw std::runtime_error(path + ": bad magic number");
    if (version != 2)         throw std::runtime_error(path + ": need a version-2 (runq.c) export");

    ck.group_size        = gs;
    ck.shared_classifier = (shared != 0);

    auto expect = [&](const char* what, long long got, long long want) {
        if (got == want) return;
        std::ostringstream os;
        os << path << ": " << what << " = " << got << ", kernel was built for " << want;
        throw std::runtime_error(os.str());
    };
    expect("dim",        ck.cfg.dim,        kDim);
    expect("hidden_dim", ck.cfg.hidden_dim, kHidden);
    expect("n_layers",   ck.cfg.n_layers,   kLayers);
    expect("n_heads",    ck.cfg.n_heads,    kHeads);
    expect("n_kv_heads", ck.cfg.n_kv_heads, kHeads);
    expect("vocab_size", ck.cfg.vocab_size, kVocab);
    expect("group_size", gs,                kGroup);
    if (ck.cfg.seq_len > kSeqLen) expect("seq_len (max)", ck.cfg.seq_len, kSeqLen);

    // fp32 section: rms_att[L][dim], rms_ffn[L][dim], rms_final[dim]
    size_t off = 256;
    ck.rms_att_off   = off;  off += (size_t)kLayers * kDim * sizeof(float);
    ck.rms_ffn_off   = off;  off += (size_t)kLayers * kDim * sizeof(float);
    ck.rms_final_off = off;  off += (size_t)kDim * sizeof(float);

    // quantized section, in file order: q_tokens, wq, wk, wv, wo, w1, w2, w3, [wcls]
    const size_t nn = (size_t)kDim * kDim;
    const size_t nh = (size_t)kDim * kHidden;
    ck.embed = detail::take_tensor(off, (size_t)kVocab * kDim);
    for (auto& t : ck.wq) t = detail::take_tensor(off, nn);
    for (auto& t : ck.wk) t = detail::take_tensor(off, nn);
    for (auto& t : ck.wv) t = detail::take_tensor(off, nn);
    for (auto& t : ck.wo) t = detail::take_tensor(off, nn);
    for (auto& t : ck.w1) t = detail::take_tensor(off, nh);
    for (auto& t : ck.w2) t = detail::take_tensor(off, nh);
    for (auto& t : ck.w3) t = detail::take_tensor(off, nh);
    ck.wcls = ck.shared_classifier ? ck.embed
                                   : detail::take_tensor(off, (size_t)kVocab * kDim);

    if (off != ck.file_size) {
        std::ostringstream os;
        os << path << ": layout ends at " << off << " B but file is " << ck.file_size << " B";
        throw std::runtime_error(os.str());
    }
    return ck;
}

// ============================================================================
//  GeMV stage plan  (mirror of weight_fsm() in transformer_kernel.cpp)
// ============================================================================
enum class StageKind { QKV, O, GateUp, Down, Classifier };

inline const char* stage_name(StageKind k) {
    switch (k) {
        case StageKind::QKV:        return "QKV";
        case StageKind::O:          return "O";
        case StageKind::GateUp:     return "gate+up";
        case StageKind::Down:       return "down";
        case StageKind::Classifier: return "classifier";
    }
    return "?";
}

struct Stage {
    StageKind kind;
    int       layer;               // -1 for the classifier
    int       n, m;                // GeMV is (m x n) * (n): n inputs, m output rows
    size_t    beat_offset;         // == the kernel's r.OFFSET for this step
    size_t    frames_per_port;     // == the kernel's r.N_FRAMES for this step

    size_t bytes()      const { return frames_per_port * kPorts * kFrameBytes; }
    size_t rows_per_port() const { return (size_t)m / kPorts; }
};

// CTRL_CNT in transformer_cu(): prefill stops after the last layer's O
// projection (it only needs the KV cache); decode runs everything.
inline int num_steps(bool prefill) { return prefill ? kLayers * 4 - 2 : kLayers * 4 + 1; }

inline std::vector<Stage> stage_plan(bool prefill = false) {
    std::vector<Stage> plan;
    size_t offset = 0;
    auto push = [&](StageKind k, int layer, int n, int m) {
        // Same integer maths as weight_fsm(): N*M*17/16 bytes, split over 2 ports.
        const size_t bytes = (size_t)n * m * 17 / 16;
        if (bytes % (kFrameBytes * kPorts) != 0) {
            std::ostringstream os;
            os << "stage " << stage_name(k) << " (" << n << "x" << m << ") is " << bytes
               << " B, not a whole number of frame pairs";
            throw std::logic_error(os.str());
        }
        plan.push_back({k, layer, n, m, offset, bytes / (kFrameBytes * kPorts)});
        offset += bytes / kBeatBytes;
    };
    for (int l = 0; l < kLayers; l++) {
        push(StageKind::QKV,    l, kDim,    kDim * 3);
        push(StageKind::O,      l, kDim,    kDim);
        push(StageKind::GateUp, l, kDim,    kHidden * 2);
        push(StageKind::Down,   l, kHidden, kDim);
    }
    push(StageKind::Classifier, -1, kDim, kVocab);

    if (offset != kPackedBeats) throw std::logic_error("stage plan does not cover the packed blob");
    plan.resize(static_cast<size_t>(num_steps(prefill)));
    return plan;
}

// Tensors concatenated (row-major, in this order) to form a stage's matrix.
inline std::vector<Tensor> stage_tensors(const Checkpoint& ck, const Stage& s) {
    const int l = s.layer;
    switch (s.kind) {
        case StageKind::QKV:        return {ck.wq[l], ck.wk[l], ck.wv[l]};
        case StageKind::O:          return {ck.wo[l]};
        case StageKind::GateUp:     return {ck.w1[l], ck.w3[l]};
        case StageKind::Down:       return {ck.w2[l]};
        case StageKind::Classifier: return {ck.wcls};
    }
    return {};
}

// Beat address the kernel reads for frame `i` of `port` (mm2ds_get_data).
inline size_t port_frame_beat(const Stage& s, int port, size_t i) {
    return s.beat_offset + (size_t)port * kFrameBeats + i * (size_t)kFrameBeats * kPorts;
}

// Which logical frame of the stage matrix that address is supposed to hold.
inline size_t logical_frame(const Stage& s, int port, size_t i) {
    return (size_t)port * s.frames_per_port + i;
}

// ============================================================================
//  Loaders -- write straight into caller-owned memory (mapped BO or vector)
// ============================================================================

// Byte offsets into the rms buffer; the kernel divides by sizeof(fdata_v_t).
struct RmsOffsets {
    int att   = 0;
    int ffn   = 0;
    int final = 0;
};

// rms buffer = [rms_att x L][rms_ffn x L][rms_final], kRmsBytes total.
inline RmsOffsets load_rms(const Checkpoint& ck, void* dst) {
    detail::File f(ck.path);
    char* p = static_cast<char*>(dst);
    const size_t layer_bytes = (size_t)kLayers * kDim * sizeof(float);

    RmsOffsets r;
    r.att   = 0;
    r.ffn   = static_cast<int>(layer_bytes);
    r.final = static_cast<int>(layer_bytes * 2);
    f.read_at(ck.rms_att_off,   p + r.att,   layer_bytes,                 "rms_att");
    f.read_at(ck.rms_ffn_off,   p + r.ffn,   layer_bytes,                 "rms_ffn");
    f.read_at(ck.rms_final_off, p + r.final, (size_t)kDim * sizeof(float), "rms_final");
    return r;
}

// Dequantized fp32 token embedding table [vocab][dim], kEmbedF32Bytes total.
inline void load_embedding_f32(const Checkpoint& ck, float* dst) {
    detail::File f(ck.path);
    const size_t n = ck.embed.n;
    std::vector<int8_t> q(n);
    std::vector<float>  s(n / kGroup);
    f.read_at(ck.embed.q_off,  q.data(), n,                         "embedding int8");
    f.read_at(ck.embed.sf_off, s.data(), s.size() * sizeof(float), "embedding scales");
    for (size_t i = 0; i < n; i++)
        dst[i] = static_cast<float>(q[i]) * s[i / kGroup];
}

// Builds the striped weight blob (kPackedBytes) consumed through w_0 and w_1.
inline void pack_weights(const Checkpoint& ck, void* dst) {
    detail::File f(ck.path);
    char* out = static_cast<char*>(dst);

    std::vector<int8_t> q;           // one stage, staged contiguously
    std::vector<float>  s;

    for (const Stage& st : stage_plan(false)) {
        const auto ts = stage_tensors(ck, st);

        size_t n_total = 0;
        for (const auto& t : ts) n_total += t.n;
        if (n_total != (size_t)st.n * st.m)
            throw std::logic_error("stage tensors do not match stage dimensions");

        q.resize(n_total);
        s.resize(n_total / kGroup);
        size_t qe = 0, se = 0;
        for (const auto& t : ts) {
            f.read_at(t.q_off,  q.data() + qe, t.n,                            "weights");
            f.read_at(t.sf_off, s.data() + se, t.n / kGroup * sizeof(float),   "scales");
            qe += t.n;
            se += t.n / kGroup;
        }

        // Emit f0(p0) f0(p1) f1(p0) f1(p1) ... -- see the header comment.
        char* p = out + st.beat_offset * kBeatBytes;
        for (size_t j = 0; j < st.frames_per_port; j++) {
            for (int port = 0; port < kPorts; port++) {
                const size_t fr = logical_frame(st, port, j);
                std::memcpy(p, s.data() + fr * kFrameScales, kFrameSfBytes);  p += kFrameSfBytes;
                std::memcpy(p, q.data() + fr * kFrameElems,  kFrameQBytes);   p += kFrameQBytes;
            }
        }
    }
}

// ============================================================================
//  Verification helpers
// ============================================================================

// Fetches the CORRECT contents of any (stage, port, frame) straight from the
// checkpoint, without going through pack_weights(). Used as the reference.
class FrameSource {
public:
    explicit FrameSource(const Checkpoint& ck) : ck_(ck), f_(ck.path) {}

    // scales: kFrameScales floats, q: kFrameElems int8
    void fetch(const Stage& s, int port, size_t i, float* scales, int8_t* q) {
        size_t e = logical_frame(s, port, i) * kFrameElems;   // element offset in stage
        for (const auto& t : stage_tensors(ck_, s)) {
            if (e < t.n) {
                if (e + kFrameElems > t.n)
                    throw std::logic_error("frame straddles two tensors");
                f_.read_at(t.q_off + e, q, kFrameElems, "reference weights");
                f_.read_at(t.sf_off + e / kGroup * sizeof(float), scales,
                           kFrameSfBytes, "reference scales");
                return;
            }
            e -= t.n;
        }
        throw std::out_of_range("frame past end of stage");
    }

private:
    const Checkpoint& ck_;
    detail::File      f_;
};

struct StripeReport {
    size_t frames_checked = 0;
    size_t bad_frames     = 0;
    bool ok() const { return frames_checked > 0 && bad_frames == 0; }
};

// Walks every frame of every stage using the KERNEL's address formula and
// checks it holds the rows that port is supposed to compute. Independent of
// how pack_weights() emitted the blob, so it catches packer bugs too.
inline StripeReport verify_striping(const Checkpoint& ck, const void* packed,
                                    std::ostream* log = nullptr, int max_errors = 10) {
    const char* blob = static_cast<const char*>(packed);
    FrameSource ref(ck);
    std::vector<float>  sf(kFrameScales);
    std::vector<int8_t> q(kFrameElems);
    StripeReport rep;

    for (const Stage& st : stage_plan(false)) {
        size_t stage_bad = 0;

        // The port split must fall on an output-row boundary, otherwise one
        // port would compute a partial row.
        const size_t rows_from_frames = st.frames_per_port * kFrameElems / st.n;
        const bool   row_aligned = (st.frames_per_port * kFrameElems) % st.n == 0
                                && rows_from_frames == st.rows_per_port();

        for (int port = 0; port < kPorts; port++) {
            for (size_t i = 0; i < st.frames_per_port; i++) {
                ref.fetch(st, port, i, sf.data(), q.data());
                const char* frame = blob + port_frame_beat(st, port, i) * kBeatBytes;
                const bool good = std::memcmp(frame, sf.data(), kFrameSfBytes) == 0
                               && std::memcmp(frame + kFrameSfBytes, q.data(), kFrameQBytes) == 0;
                rep.frames_checked++;
                if (good) continue;
                rep.bad_frames++;
                if (stage_bad++ < (size_t)max_errors && log)
                    *log << "  MISMATCH stage " << stage_name(st.kind) << " L" << st.layer
                         << " port " << port << " frame " << i
                         << " (beat " << port_frame_beat(st, port, i) << ")\n";
            }
        }
        if (!row_aligned) rep.bad_frames++;

        if (log) {
            *log << "  " << (st.layer < 0 ? std::string("  ") : (st.layer < 10 ? "L0" : "L")
                             + std::to_string(st.layer))
                 << " " << std::left << std::setw(10) << stage_name(st.kind) << std::right
                 << std::setw(5) << st.n << "x" << std::left << std::setw(6) << st.m << std::right
                 << "  frames/port " << std::setw(4) << st.frames_per_port
                 << "  w_0 rows [0," << st.rows_per_port() << ")"
                 << "  w_1 rows [" << st.rows_per_port() << "," << st.m << ")"
                 << (row_aligned ? "" : "  ROW SPLIT MISALIGNED")
                 << (stage_bad ? "  FAIL" : "  ok") << "\n";
        }
    }
    return rep;
}

} // namespace llama

#endif // LLAMA_LAYOUT_H
