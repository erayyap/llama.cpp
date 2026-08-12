#pragma once

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

//
// common_ngram_mod
// ref: https://github.com/ggml-org/llama.cpp/pull/19164
//

// basic n-gram hasher
struct common_ngram_mod {
    using entry_t = int32_t;

    static constexpr entry_t EMPTY = -1;

    common_ngram_mod(uint16_t n, size_t size);

    size_t  idx(const entry_t * tokens) const;
    void    add(const entry_t * tokens);
    entry_t get(const entry_t * tokens) const; // return -1 if not found

    void reset();

    size_t get_n()    const;
    size_t get_used() const;

    size_t size()       const;
    size_t size_bytes() const;

private:
    size_t n; // ngram size to hash

    size_t used;

    std::vector<entry_t> entries;
};

struct common_ngram_mod_router_config {
    int32_t n_match;
    int32_t n_min;
    int32_t n_max;
    int32_t n_ctx_max;
};

struct common_ngram_mod_route {
    bool selected = false;
    int32_t width = 0;
    int32_t threshold = 0;
    int32_t occurrences = 0;
    int32_t agreement = 0;
    int32_t copy_run = 0;
    bool from_prompt = false;
};

// The index stores positions only and is cleared at every begin(), so tokens and learning never cross requests.
struct common_ngram_mod_router {
    using entry_t = common_ngram_mod::entry_t;

    static constexpr uint64_t GENERATED_REGION = UINT64_C(1) << 63;
    static constexpr uint64_t INVALID_REGION = UINT64_MAX;

    explicit common_ngram_mod_router(common_ngram_mod_router_config config);

    void begin(const std::vector<entry_t> & prompt);
    common_ngram_mod_route draft(
            const std::vector<entry_t> & prompt,
            entry_t sampled,
            int32_t n_max,
            std::vector<entry_t> & result);
    void accept(int32_t n_accepted);

private:
    struct key_info {
        uint32_t first = 0;
        uint32_t last = 0;
        uint32_t occurrences = 0;
        bool collision = false;
    };

    struct region_info {
        int32_t successes = 0;
        int32_t last_accepted = 0;
    };

    struct pending_info {
        bool active = false;
        uint64_t region = INVALID_REGION;
        int32_t width = 0;
        int32_t threshold = 0;
    };

    common_ngram_mod_router_config config;
    size_t prompt_size = 0;
    size_t i_next = 0;
    int32_t cooldown_left = 0;
    std::unordered_map<size_t, key_info> keys;
    std::unordered_map<uint64_t, region_info> regions;
    pending_info pending;

    size_t hash(const entry_t * tokens) const;
    uint64_t region_for(bool from_prompt, size_t current_start, size_t source) const;
    void update_index(const std::vector<entry_t> & tokens);
    int32_t threshold(int32_t context, int32_t width) const;
};
