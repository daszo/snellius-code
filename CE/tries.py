class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False


class DocIDTrie:
    def __init__(self, tokenizer):
        self.root = TrieNode()
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id

    def insert(self, token_ids: list):
        """Inserts a sequence of token_ids (a doc_id) into the trie."""
        node = self.root
        for token in token_ids:
            if token not in node.children:
                node.children[token] = TrieNode()
            node = node.children[token]
        node.is_end_of_word = True
        # Explicitly allow EOS at the end of a valid doc_id
        if self.eos_token_id not in node.children:
            node.children[self.eos_token_id] = TrieNode()

    def get_next_allowed_tokens(self, current_sequence: list):
        """
        Traverses the trie based on current_sequence and returns
        list of allowed next token_ids.
        """
        node = self.root

        # Traverse the trie to the current state
        for token in current_sequence:
            # Skip pad tokens usually found at start of decoder sequences
            if token == self.pad_token_id:
                continue

            if token in node.children:
                node = node.children[token]
            else:
                # If we wandered off the trie (shouldn't happen with forced decoding),
                # we force EOS or return empty list to stop generation.
                return [self.eos_token_id]

        return list(node.children.keys())
