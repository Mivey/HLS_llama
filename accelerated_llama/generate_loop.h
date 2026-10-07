// ============================================================================
//  generate_loop.h  --  token generation loop (runq.c generate(), on the FPGA)
//
//  Templated on the engine so the same loop can drive llama::TransformerCU on
//  hardware or the testbench's SimTransformerCU in csim. Any engine with
//  seed_prompt / enable_prefill / enable_decode / set_rms_flag /
//  startForward / endForward works.
//
//  #include AFTER Tokenizer, Sampler, encode(), decode(), safe_printf(),
//  random_f32() and time_in_ms() are declared.
// ============================================================================
#ifndef GENERATE_LOOP_H
#define GENERATE_LOOP_H

#include "llama_layout.h"

//   pos <  n_prompt-1  ->  prefill: kernel fills the KV cache only; endForward()
//                          hands back the prompt token already in curr_token[]
//   pos >= n_prompt-1  ->  decode:  kernel runs the LM head and writes
//                          curr_token[pos+1]
template <typename Engine>
void newgen(Tokenizer* tokenizer, Sampler* sampler, char* prompt, int steps, Engine& f)
{
    char empty[] = "";
    if (prompt == NULL) prompt = empty;

    int  n_prompt = 0;
    int* prompt_tokens = (int*)malloc((strlen(prompt) + 3) * sizeof(int));
    encode(tokenizer, prompt, 1, 0, prompt_tokens, &n_prompt);
    if (n_prompt < 1) {
        fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
        exit(EXIT_FAILURE);
    }

    // The kernel writes curr_token[pos+1], so the last usable pos is kSeqLen-2.
    const int max_steps = llama::kSeqLen - 1;
    if (steps > max_steps) steps = max_steps;
    if (n_prompt > max_steps) {
        fprintf(stderr, "prompt is longer than the KV cache (%d tokens)\n", max_steps);
        exit(EXIT_FAILURE);
    }

    f.seed_prompt(prompt_tokens, n_prompt);
    f.set_rms_flag(true);                  // first call loads rms weights into URAM

    bool prefill = (n_prompt > 1);
    if (prefill) f.enable_prefill(); else f.enable_decode();

    long start = 0;
    int  pos   = 0;
    int  token = prompt_tokens[0];

    while (pos < steps) {
        const bool want_prefill = (pos < n_prompt - 1);
        if (want_prefill != prefill) {
            prefill = want_prefill;
            if (prefill) f.enable_prefill(); else f.enable_decode();
        }

        f.startForward(pos, random_f32(&sampler->rng_state));
        const int next = f.endForward(pos);

        if (pos == 0) {
            f.set_rms_flag(false);
            start = time_in_ms();
        }

        if (next == 1) { pos++; break; }  // BOS terminates, as in runq.c
        if (next < 0 || next >= tokenizer->vocab_size) {
            fprintf(stderr, "\n[host] kernel returned out-of-range token %d at pos %d\n", next, pos);
            break;
        }

        safe_printf(decode(tokenizer, token, next));
        fflush(stdout);

        token = next;
        pos++;
    }
    printf("\n");

    if (pos > 1 && start != 0) {
        const long end = time_in_ms();
        fprintf(stderr, "achieved tok/s: %f\n", (pos - 1) / (double)(end - start) * 1000);
    }
    free(prompt_tokens);
}

#endif // GENERATE_LOOP_H