package store

import (
	"regexp"
	"strings"
)

// stopwords is a set of common English stopwords and agent-specific filler
// that should be removed from FTS queries. These words dilute BM25 ranking
// because they match broadly across all indexed symbols.
var stopwords = map[string]struct{}{
	"a": {}, "about": {}, "above": {}, "after": {}, "again": {}, "against": {},
	"all": {}, "am": {}, "an": {}, "and": {}, "any": {}, "are": {}, "as": {},
	"at": {}, "be": {}, "because": {}, "been": {}, "before": {}, "being": {},
	"below": {}, "between": {}, "both": {}, "but": {}, "by": {}, "can": {},
	"cannot": {}, "could": {}, "did": {}, "do": {}, "does": {}, "doing": {},
	"don't": {}, "down": {}, "during": {}, "each": {}, "few": {}, "for": {},
	"from": {}, "further": {}, "get": {}, "got": {}, "had": {}, "has": {},
	"have": {}, "having": {}, "he": {}, "her": {}, "here": {}, "hers": {},
	"herself": {}, "him": {}, "himself": {}, "his": {}, "how": {}, "i": {},
	"if": {}, "in": {}, "into": {}, "is": {}, "it": {}, "its": {}, "itself": {},
	"just": {}, "let": {}, "like": {}, "me": {}, "might": {}, "more": {},
	"most": {}, "my": {}, "myself": {}, "need": {}, "no": {}, "nor": {},
	"not": {}, "now": {}, "of": {}, "off": {}, "on": {}, "once": {}, "only": {},
	"or": {}, "other": {}, "ought": {}, "our": {}, "ours": {}, "ourselves": {},
	"out": {}, "over": {}, "own": {}, "please": {}, "same": {}, "she": {},
	"should": {}, "so": {}, "some": {}, "such": {}, "than": {}, "that": {},
	"the": {}, "their": {}, "theirs": {}, "them": {}, "themselves": {},
	"then": {}, "there": {}, "these": {}, "they": {}, "this": {}, "those": {},
	"through": {}, "to": {}, "too": {}, "under": {}, "until": {}, "up": {},
	"us": {}, "very": {}, "want": {}, "was": {}, "we": {}, "were": {},
	"what": {}, "when": {}, "where": {}, "which": {}, "while": {}, "who": {},
	"whom": {}, "why": {}, "will": {}, "with": {}, "won't": {}, "would": {},
	"you": {}, "your": {}, "yours": {}, "yourself": {}, "yourselves": {},
	// Agent-specific filler
	"help": {}, "file": {}, "files": {}, "code": {}, "use": {}, "using": {},
	"make": {}, "way": {}, "thing": {}, "something": {},
}

// tokenRe matches lowercase alphanumeric tokens, including hyphenated,
// dotted, and underscored compound tokens (e.g., "pod-affinity", "net.http").
var tokenRe = regexp.MustCompile(`[a-z0-9]+(?:[-_.][a-z0-9]+)*`)

const (
	minTokenLen = 3
	maxTokenLen = 30
	maxTerms    = 12
)

// ExtractKeywords extracts discriminative keywords from a natural-language
// query for use in FTS5 MATCH clauses. It lowercases the input, removes
// stopwords, deduplicates tokens, and filters by length. The result is a
// space-separated string of terms.
func ExtractKeywords(query string) string {
	lower := strings.ToLower(query)
	tokens := tokenRe.FindAllString(lower, -1)

	seen := make(map[string]struct{}, len(tokens))
	result := make([]string, 0, len(tokens))

	for _, tok := range tokens {
		if len(tok) < minTokenLen || len(tok) > maxTokenLen {
			continue
		}
		if _, stop := stopwords[tok]; stop {
			continue
		}
		if _, dup := seen[tok]; dup {
			continue
		}
		seen[tok] = struct{}{}
		result = append(result, tok)
		if len(result) >= maxTerms {
			break
		}
	}

	return strings.Join(result, " ")
}
