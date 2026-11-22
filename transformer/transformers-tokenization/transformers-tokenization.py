class SimpleTokenizer:
    def __init__(self):
        self.word_to_id = {}
        self.id_to_word = {}
        self.vocab_size = 0
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

    def build_vocab(self, texts: list[str]) -> None:
        """
        Builds the vocabulary in place.
        """
        
        self.word_to_id[self.pad_token] = 0
        self.word_to_id[self.unk_token] = 1
        self.word_to_id[self.bos_token] = 2
        self.word_to_id[self.eos_token] = 3

        self.id_to_word[0] = self.pad_token
        self.id_to_word[1] = self.unk_token
        self.id_to_word[2] = self.bos_token
        self.id_to_word[3] = self.eos_token

        unique_set = set()

        for text in texts:
            tokens = text.split(" ")

            for idx, ch in enumerate(tokens):
                unique_set.add(ch.lower())

        unique_set = sorted(unique_set)
        for idx, ch in enumerate(unique_set, start=4):
            self.word_to_id[ch.lower()] = idx
            self.id_to_word[idx] = ch.lower()
        
        self.vocab_size = len(unique_set) + 4

    def encode(self, text: str) -> list[int]:
        """
        Returns token IDs for the input text.
        """
        if text is None or text is "":
            return []
        
        result = list()
        tokens = text.split(" ")

        # result.append(self.word_to_id[self.bos_token])

        for idx, ch in enumerate(tokens):
            if self.word_to_id.get(ch.lower()):
                result.append(self.word_to_id.get(ch.lower()))
            else:
                result.append(self.word_to_id.get(self.unk_token))

        # result.append(self.word_to_id[self.eos_token])

        return result

    def decode(self, ids: list[int]) -> str:
        """
        Returns the decoded, space-separated text.
        """
        result = []

        for id in ids:
            if self.id_to_word.get(id):
                result.append(self.id_to_word.get(id))
            else:
                result.append(self.unk_token)

        if len(result) > 0:
            return " ".join(result)

        return ""
            