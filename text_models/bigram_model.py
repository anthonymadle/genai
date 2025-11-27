# text_models/bigram_model.py

import random
from collections import defaultdict


class BigramModel:
    """
    Very simple bigram language model.
    Trains on a piece of text and generates new text
    word-by-word using bigram probabilities.
    """

    def __init__(self, text: str = "this is a simple default training text"):
        self.counts = defaultdict(lambda: defaultdict(int))
        self._train(text)

    def _train(self, text: str):
        words = text.split()
        for w1, w2 in zip(words, words[1:]):
            self.counts[w1][w2] += 1

    def generate(self, prompt: str, max_words: int = 20) -> str:
        words = prompt.split()

        if not words:
            words = ["this"]

        current = words[-1]

        for _ in range(max_words):
            next_words = self.counts.get(current)
            if not next_words:
                break

            choices = list(next_words.keys())
            weights = [next_words[w] for w in choices]

            current = random.choices(choices, weights=weights)[0]
            words.append(current)

        return " ".join(words)
