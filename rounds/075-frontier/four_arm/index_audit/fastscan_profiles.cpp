#include <cstddef>
#include <cstdint>
#include <unordered_set>

static bool lower(char c) { return c >= 'a' && c <= 'z'; }
static bool upper(char c) { return c >= 'A' && c <= 'Z'; }
static bool is_digit(char c) { return c >= '0' && c <= '9'; }

static void count_cno(char atom, int *counts) {
    if (atom == 'C' || atom == 'c') ++counts[0];
    else if (atom == 'N' || atom == 'n') ++counts[1];
    else if (atom == 'O' || atom == 'o') ++counts[2];
}

// This only rejects definite C/N/O-count mismatches. Other elements do not
// affect this prefilter. Unfamiliar syntax is left for the full RDKit check.
static bool count_profile(const char *cursor, const char *end, int *counts) {
    while (cursor < end) {
        const char token = *cursor++;
        if (token == '[') {
            while (cursor < end && is_digit(*cursor)) ++cursor;
            if (cursor == end) return false;
            const char atom = *cursor++;
            if (upper(atom)) {
                if (cursor < end && lower(*cursor)) ++cursor;
                else count_cno(atom, counts);
            } else if (atom == 'c' || atom == 'n' || atom == 'o') {
                if (cursor < end && lower(*cursor)) return false;
                count_cno(atom, counts);
            } else if (atom != '*' && atom != 'b' && atom != 'p' && atom != 's' && atom != 'a') {
                return false;
            }
            while (cursor < end && *cursor != ']') ++cursor;
            if (cursor == end) return false;
            ++cursor;
        } else if ((token == 'C' && cursor < end && *cursor == 'l') ||
                   (token == 'B' && cursor < end && *cursor == 'r')) {
            ++cursor;
        } else {
            switch (token) {
                case 'C': case 'N': case 'O': case 'c': case 'n': case 'o':
                    count_cno(token, counts);
                    break;
                case 'B': case 'P': case 'S': case 'F': case 'I': case 'b': case 'p': case 's':
                case '-': case '=': case '#': case '$': case '+': case ':':
                case '/': case '\\': case '.': case '(': case ')': case '@':
                case '%': case '*': case '0': case '1': case '2': case '3': case '4':
                case '5': case '6': case '7': case '8': case '9':
                    break;
                default:
                    return false;
            }
        }
    }
    return true;
}

static std::uint64_t profile_key(const int *counts) {
    return (std::uint64_t(counts[0]) << 42) |
           (std::uint64_t(counts[1]) << 21) |
           std::uint64_t(counts[2]);
}

extern "C" std::size_t select_profiles(const char *data, const std::int64_t *offsets,
                                       std::size_t n, const int *profiles,
                                       std::size_t n_profiles, std::int64_t *out) {
    std::unordered_set<std::uint64_t> wanted;
    for (std::size_t i = 0; i < n_profiles; ++i) wanted.insert(profile_key(profiles + 3 * i));
    std::size_t found = 0;
    for (std::size_t i = 0; i < n; ++i) {
        int counts[3] = {};
        if (!count_profile(data + offsets[i], data + offsets[i + 1], counts) ||
            wanted.find(profile_key(counts)) != wanted.end()) {
            out[found++] = i;
        }
    }
    return found;
}
