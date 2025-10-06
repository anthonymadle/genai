# app/bigram_model.py
from collections import defaultdict, Counter
import random
import re

def simple_tokenizer(text, frequency_threshold=5):
    """Simple tokenizer that splits text into words."""
    tokens = re.findall(r"\b\w+\b", text.lower())
    if not frequency_threshold:
        return tokens
    word_counts = Counter(tokens)
    filtered_tokens = [t for t in tokens if word_counts[t] >= frequency_threshold]
    return filtered_tokens

def analyze_bigrams(text, frequency_threshold=None):
    """Analyze text to compute bigram probabilities."""
    words = simple_tokenizer(text, frequency_threshold)
    bigrams = list(zip(words[:-1], words[1:]))
    bigram_counts = Counter(bigrams)
    unigram_counts = Counter(words)
    bigram_probs = defaultdict(dict)
    for (w1, w2), count in bigram_counts.items():
        bigram_probs[w1][w2] = count / unigram_counts[w1]
    return list(unigram_counts.keys()), bigram_probs

def generate_text(bigram_probs, start_word, num_words=20):
    """Generate text based on bigram probabilities."""
    current = start_word.lower()
    generated = [current]
    for _ in range(num_words - 1):
        nexts = bigram_probs.get(current)
        if not nexts:
            break
        next_word = random.choices(list(nexts.keys()), weights=nexts.values())[0]
        generated.append(next_word)
        current = next_word
    return " ".join(generated)

def print_bigram_probs_matrix_python(vocab, bigram_probs):
    """Pretty-print bigram probabilities in a matrix (for debugging)."""
    print(f"{'':<15}", end="")
    for word in vocab:
        print(f"{word:<15}", end="")
    print("\n" + "-" * (15 * (len(vocab) + 1)))
    for w1 in vocab:
        print(f"{w1:<15}", end="")
        for w2 in vocab:
            prob = bigram_probs.get(w1, {}).get(w2, 0)
            print(f"{prob:<15.2f}", end="")
        print()

# NOTE: No top-level demo code here. If you need to run a demo, do it under:
# if __name__ == "__main__":
#     ...
