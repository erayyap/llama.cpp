#include "ngram-mod.h"
#include "speculative.h"

#include <cstdlib>
#include <iostream>
#include <vector>

using tokens = std::vector<int32_t>;

static void require(bool condition) {
    if (!condition) {
        std::abort();
    }
}

static tokens block(int32_t base, int32_t size) {
    tokens result;
    result.reserve(size);
    for (int32_t i = 0; i < size; ++i) {
        result.push_back(base + i);
    }
    return result;
}

static tokens reliable_prompt(const tokens & source) {
    tokens prompt = source;
    prompt.push_back(900001);
    prompt.insert(prompt.end(), source.begin(), source.end());
    prompt.push_back(900002);
    prompt.insert(prompt.end(), source.begin(), source.begin() + 23);
    return prompt;
}

static void append_verified(tokens & prompt, int32_t sampled, const tokens & draft, int32_t accepted) {
    prompt.push_back(sampled);
    prompt.insert(prompt.end(), draft.begin(), draft.begin() + accepted);
}

int main() {
    const auto source = block(1000, 220);
    const common_ngram_mod_router_config config { 24, 48, 64, 73728 };

    {
        common_ngram_mod_router router(config);
        const auto prompt = block(5000, 160);
        tokens draft;
        router.begin(prompt);
        const auto route = router.draft(prompt, 42, 64, draft);
        require(!route.selected);
        require(draft.empty());
    }

    {
        common_ngram_mod_router router(config);
        auto prompt = reliable_prompt(source);
        tokens draft;
        router.begin(prompt);

        auto route = router.draft(prompt, source[23], 64, draft);
        require(route.selected);
        require(route.width == 48);
        require(route.agreement >= route.threshold);
        append_verified(prompt, source[23], draft, 48);
        router.accept(48);

        draft.clear();
        route = router.draft(prompt, source[72], 64, draft);
        require(route.selected);
        require(route.width == 48);
        append_verified(prompt, source[72], draft, 48);
        router.accept(48);

        draft.clear();
        route = router.draft(prompt, source[121], 64, draft);
        require(route.selected);
        require(route.width == 64);
    }

    {
        common_ngram_mod_router router(config);
        const auto prompt = reliable_prompt(source);
        tokens draft;
        router.begin(prompt);
        auto route = router.draft(prompt, source[23], 64, draft);
        require(route.selected);
        router.accept(8);

        for (int32_t i = 0; i < 4; ++i) {
            draft.clear();
            route = router.draft(prompt, source[23], 64, draft);
            require(!route.selected);
        }
        draft.clear();
        require(router.draft(prompt, source[23], 64, draft).selected);
    }

    {
        common_ngram_mod_router router(config);
        const auto prompt_a = reliable_prompt(source);
        tokens draft;
        router.begin(prompt_a);
        require(router.draft(prompt_a, source[23], 64, draft).selected);
        router.accept(4);
        router.begin(prompt_a);
        draft.clear();
        require(router.draft(prompt_a, source[23], 64, draft).selected);

        const auto prompt_b = block(7000, 180);
        router.begin(prompt_b);
        draft.clear();
        require(!router.draft(prompt_b, source[23], 64, draft).selected);
    }

    {
        auto cutoff_config = config;
        cutoff_config.n_ctx_max = 400;
        common_ngram_mod_router router(cutoff_config);
        const auto prompt = reliable_prompt(source);
        tokens draft;
        router.begin(prompt);
        require(!router.draft(prompt, source[23], 64, draft).selected);
    }

    {
        common_ngram_mod_router router(config);
        const auto prompt = reliable_prompt(source);
        tokens draft;
        router.begin(prompt);
        require(router.draft(prompt, source[23], 64, draft).selected);
        router.accept(12);
        draft.clear();
        require(!router.draft(prompt, source[23], 64, draft).selected);
    }

    {
        common_params_speculative params;
        params.types = { COMMON_SPECULATIVE_TYPE_NGRAM_MOD, COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK };
        params.ngram_mod.n_match = config.n_match;
        params.ngram_mod.n_min = config.n_min;
        params.ngram_mod.n_max = config.n_max;
        params.draft.n_ctx_max = config.n_ctx_max;
        common_speculative_ptr spec(common_speculative_init(params, 1));

        const auto prompt = reliable_prompt(source);
        tokens draft;
        common_speculative_begin(spec.get(), 0, prompt);
        common_speculative_get_draft_params(spec.get(), 0) = {
            true, 64, (llama_pos) prompt.size(), source[23], &prompt, &draft, nullptr,
        };
        common_speculative_draft(spec.get());
        require(!draft.empty());
        require(common_speculative_last_type(spec.get(), 0) == COMMON_SPECULATIVE_TYPE_NGRAM_MOD);

        common_speculative_draft(spec.get());
        require(common_speculative_last_type(spec.get(), 0) == COMMON_SPECULATIVE_TYPE_NGRAM_MOD);
        common_speculative_accept(spec.get(), 0, 12);
    }

    std::cout << "adaptive ngram router tests passed\n";
    return 0;
}
