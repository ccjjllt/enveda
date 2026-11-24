#include <cstdint>
#include <cstddef>

// Return C/N/O heavy atom counts for ordinary SMILES. Anything not understood
// is sent to RDKit instead of being discarded.  A recognized non-C/N/O
// atom makes a row ineligible under the competition's heavy-atom rule.
static int composition(const char *s, const char *end, int *counts) {
    while (s < end) {
        char atom = *s++;
        if (atom == '[') {
            while (s < end && '0' <= *s && *s <= '9') ++s;
            if (s == end) return -1;
            atom = *s++;
            if (atom == 'C' || atom == 'N' || atom == 'O') {
                if (s < end && 'a' <= *s && *s <= 'z') return 0;
                ++counts[atom == 'C' ? 0 : atom == 'N' ? 1 : 2];
            } else if (atom == 'c' || atom == 'n' || atom == 'o') {
                ++counts[atom == 'c' ? 0 : atom == 'n' ? 1 : 2];
            } else if (atom == 'H') {
                if (s < end && 'a' <= *s && *s <= 'z') return 0;
            } else if (atom == 'D' || atom == 'T') {
                if (s < end && 'a' <= *s && *s <= 'z') return 0;
            } else if (atom != '*') {
                if (('A' <= atom && atom <= 'Z') || ('a' <= atom && atom <= 'z')) return 0;
                return -1;
            }
            while (s < end && *s != ']') ++s;
            if (s == end) return -1;
            ++s;
            continue;
        }
        switch (atom) {
            case 'C':
                if (s < end && *s == 'l') return 0;
                [[fallthrough]];
            case 'c': ++counts[0]; break;
            case 'N': case 'n': ++counts[1]; break;
            case 'O': case 'o': ++counts[2]; break;
            case 'B': case 'F': case 'I': case 'P': case 'S':
            case 'b': case 'p': case 's': return 0;
            case '-': case '=': case '#': case '$': case '+': case ':':
            case '/': case '\\': case '.': case '(': case ')': case '@':
            case '%': case '*':
            case '0': case '1': case '2': case '3': case '4':
            case '5': case '6': case '7': case '8': case '9': break;
            default: return -1;
        }
    }
    return 1;
}

extern "C" std::size_t select_profiles(const char *data, const std::int64_t *off,
                                      std::size_t n, const int *profiles,
                                      std::size_t n_profiles, std::int64_t *out) {
    std::size_t found = 0;
    for (std::size_t i = 0; i < n; ++i) {
        int counts[3] = {};
        const auto result = composition(data + off[i], data + off[i+1], counts);
        if (result < 0) {
            out[found++] = i;
        } else if (result == 1) {
            for (std::size_t p = 0; p < n_profiles; ++p) {
                if (counts[0] == profiles[3*p] && counts[1] == profiles[3*p+1] &&
                    counts[2] == profiles[3*p+2]) {
                    out[found++] = i;
                    break;
                }
            }
        }
    }
    return found;
}
