#ifndef ABEL_INFERENCE_H
#define ABEL_INFERENCE_H

#ifdef __cplusplus
extern "C" {
#endif

struct Inference;

struct Inference* inference_create(const char* model_path, const char* tokenizer_path);
void inference_free(struct Inference* infer);

char* inference_generate(
    struct Inference* infer,
    const char* prompt,
    int max_new_tokens,
    float temperature,
    float top_p,
    float repetition_penalty,
    const float* logit_bias,
    int logit_bias_len);

void inference_free_result(char* str);

#ifdef __cplusplus
}
#endif

#endif // ABEL_INFERENCE_H
