#include "ngram-mod.h"

#include <algorithm>
#include <cmath>
#include <limits>

//
// common_ngram_mod
//

common_ngram_mod::common_ngram_mod(uint16_t n, size_t size) : n(n), used(0) {
    entries.resize(size);

    reset();
}

size_t common_ngram_mod::idx(const entry_t * tokens) const {
    size_t res = 0;

    for (size_t i = 0; i < n; ++i) {
        res = res*6364136223846793005ULL + tokens[i];
    }

    res = res % entries.size();

    return res;
}

void common_ngram_mod::add(const entry_t * tokens) {
    const size_t i = idx(tokens);

    if (entries[i] == EMPTY) {
        used++;
    }

    entries[i] = tokens[n];
}

common_ngram_mod::entry_t common_ngram_mod::get(const entry_t * tokens) const {
    const size_t i = idx(tokens);

    return entries[i];
}

void common_ngram_mod::reset() {
    std::fill(entries.begin(), entries.end(), EMPTY);
    used = 0;
}

size_t common_ngram_mod::get_n() const {
    return n;
}

size_t common_ngram_mod::get_used() const {
    return used;
}

size_t common_ngram_mod::size() const {
    return entries.size();
}

size_t common_ngram_mod::size_bytes() const {
    return entries.size() * sizeof(entries[0]);
}

//
// common_ngram_mod_router
//

common_ngram_mod_router::common_ngram_mod_router(common_ngram_mod_router_config config) : config(config) {
}

size_t common_ngram_mod_router::hash(const entry_t * tokens) const {
    size_t result = 0;
    for (int32_t i = 0; i < config.n_match; ++i) {
        result = result*6364136223846793005ULL + tokens[i];
    }
    return result;
}

uint64_t common_ngram_mod_router::region_for(bool from_prompt, size_t current_start, size_t source) const {
    if (current_start < source) {
        return INVALID_REGION;
    }
    return (from_prompt ? 0 : GENERATED_REGION) | (current_start - source);
}

void common_ngram_mod_router::update_index(const std::vector<entry_t> & tokens) {
    const size_t n = config.n_match;
    const size_t n_min = config.n_min;

    while (i_next + n + n_min <= tokens.size()) {
        const size_t key_hash = hash(tokens.data() + i_next);
        auto [it, inserted] = keys.emplace(key_hash, key_info {});
        auto & info = it->second;

        if (inserted) {
            info.first = i_next;
            info.last = i_next;
            info.occurrences = 1;
        } else if (!std::equal(
                       tokens.begin() + info.first,
                       tokens.begin() + info.first + n,
                       tokens.begin() + i_next)) {
            info.collision = true;
        } else {
            info.last = i_next;
            info.occurrences++;
        }

        i_next++;
    }
}

void common_ngram_mod_router::begin(const std::vector<entry_t> & prompt) {
    prompt_size = prompt.size();
    i_next = 0;
    cooldown_left = 0;
    keys.clear();
    regions.clear();
    pending = {};
    keys.reserve(prompt.size());
    update_index(prompt);
}

int32_t common_ngram_mod_router::threshold(int32_t context, int32_t width) const {
    constexpr int32_t context_lo = 16000;
    constexpr int32_t context_hi = 64000;

    const double scale = std::clamp(
            (double) (context - context_lo)/(context_hi - context_lo), 0.0, 1.0);

    int32_t break_even;
    if (width <= 32) {
        break_even = (int32_t) std::lround(25.0 + 3.0*scale);
    } else if (width <= 48) {
        break_even = (int32_t) std::lround(27.0 + 4.0*scale);
    } else {
        break_even = 35;
    }

    return break_even + 4;
}

common_ngram_mod_route common_ngram_mod_router::draft(
        const std::vector<entry_t> & prompt,
        entry_t sampled,
        int32_t n_max,
        std::vector<entry_t> & result) {
    common_ngram_mod_route route;
    pending = {};
    result.clear();

    if (cooldown_left > 0) {
        cooldown_left--;
        return route;
    }

    const size_t n = config.n_match;
    const size_t cur_len = prompt.size();
    if (config.n_min <= 0 || config.n_max < config.n_min || n_max < config.n_min || cur_len + 1 < n) {
        return route;
    }
    if (config.n_ctx_max > 0 && cur_len >= (size_t) config.n_ctx_max) {
        return route;
    }

    update_index(prompt);

    std::vector<entry_t> key(n);
    std::copy(prompt.end() - (n - 1), prompt.end(), key.begin());
    key.back() = sampled;

    const auto it = keys.find(hash(key.data()));
    if (it == keys.end() || it->second.collision) {
        return route;
    }

    const auto & info = it->second;
    auto key_matches = [&](size_t pos) {
        return pos + n <= prompt.size() &&
            std::equal(key.begin(), key.end(), prompt.begin() + pos);
    };

    if (!key_matches(info.first) || !key_matches(info.last)) {
        return route;
    }

    const size_t current_start = cur_len - n + 1;
    auto copy_run = [&](size_t pos) {
        int32_t length = config.n_match;
        size_t i_cur = current_start;
        size_t i_src = pos;
        while (i_cur > 0 && i_src > 0 && prompt[i_cur - 1] == prompt[i_src - 1] && length < 256) {
            i_cur--;
            i_src--;
            length++;
        }
        return length;
    };

    size_t source = info.last;
    int32_t n_copy = copy_run(source);
    const int32_t first_copy = copy_run(info.first);
    if (first_copy > n_copy) {
        source = info.first;
        n_copy = first_copy;
    }

    const bool from_prompt = source + n < prompt_size;
    uint64_t region = region_for(from_prompt, current_start, source);
    region_info default_region;
    auto * region_state = region == INVALID_REGION ? &default_region : &regions[region];

    int32_t width = config.n_min;
    if (region_state->successes >= 2) {
        width = config.n_max;
    }
    width = std::min(width, n_max);

    auto has_width = [&](size_t pos) {
        return pos + n + width <= prompt.size();
    };
    if (!has_width(source)) {
        const size_t other = source == info.first ? info.last : info.first;
        if (!key_matches(other) || !has_width(other)) {
            return route;
        }
        source = other;
        n_copy = copy_run(source);
    }

    int32_t agreement = 0;
    if (info.occurrences >= 2 && has_width(info.first) && has_width(info.last)) {
        while (agreement < width &&
               prompt[info.first + n + agreement] == prompt[info.last + n + agreement]) {
            agreement++;
        }
    }

    const int32_t required = threshold(cur_len, width);
    const bool reliable =
        agreement >= required ||
        n_copy >= required + 12 ||
        (region_state->successes > 0 && region_state->last_accepted >= required);
    if (!reliable) {
        return route;
    }

    result.insert(
            result.end(),
            prompt.begin() + source + n,
            prompt.begin() + source + n + width);

    pending = { true, region, width, required };
    route.selected = true;
    route.width = width;
    route.threshold = required;
    route.occurrences = std::min<uint32_t>(info.occurrences, std::numeric_limits<int32_t>::max());
    route.agreement = agreement;
    route.copy_run = n_copy;
    route.from_prompt = source + n + width <= prompt_size;
    return route;
}

void common_ngram_mod_router::accept(int32_t n_accepted) {
    if (!pending.active) {
        return;
    }

    if (pending.region != INVALID_REGION) {
        auto & region = regions[pending.region];
        if (n_accepted < pending.threshold) {
            region.successes = 0;
            region.last_accepted = n_accepted;
        } else {
            region.successes++;
            region.last_accepted = n_accepted;
        }
    }
    if (n_accepted < pending.threshold) {
        cooldown_left = 4;
    }
    pending = {};
}
