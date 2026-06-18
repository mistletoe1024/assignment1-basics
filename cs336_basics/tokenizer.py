import regex as re
from typing import Iterable, Iterator

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

        bytes_to_jd = {}
        for id, b in self.vocab.items():
            assert b not in bytes_to_jd
            bytes_to_jd[b] = id
        self.bytes_to_id = bytes_to_jd
        self.id_to_bytes = vocab

    def encode(self, text: str) -> list[int]:
        target_str_list = []
        if self.special_tokens:
            escaped = "|".join(
                re.escape(token) 
                for token in sorted(self.special_tokens, key=len, reverse=True)
            )
            pattern = re.compile(f"({escaped})")
            parts = pattern.split(text)
            target_str_list = [p for p in parts if p != ""]
        else :
            target_str_list = [text]
        
        id_list = []
        
        for target_str in target_str_list:
            if self.special_tokens and target_str in self.special_tokens:
                id_list.append(self.bytes_to_id[target_str.encode("utf-8")])
                continue

            matches = re.finditer(PAT, target_str)
            for match in matches:
                key = match.group(0).encode("utf-8")
                pieces = [bytes([k]) for k in key]
                if self.merges:
                    for merge in self.merges:
                        i = 0
                        while i < len(pieces) - 1:
                            if pieces[i] == merge[0] and pieces[i+1] == merge[1]:
                                pieces[i] = merge[0] + merge[1]
                                del pieces[i+1]
                            i += 1
                    id_list.extend([self.bytes_to_id[piece] for piece in pieces])
        return id_list

    def decode(self, ids: list[int]) -> str:
        # 先拼接bytes,再整体decode,用 replace 处理无效序列
        all_bytes = b"".join(self.id_to_bytes[i] for i in ids)
        return all_bytes.decode("utf-8", errors="replace")
    

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        if not self.special_tokens:
            for chunk in iterable:
                yield from self.encode(chunk)
            return
        else:
            max_special_len = max(len(special_token) for special_token in self.special_tokens)

        read_buffer = ""
        holdback = max_special_len - 1
        while True:
            chunk = next(iterable, None)
            if chunk is None:
                break
            text = read_buffer + chunk

            safe_len = len(text) - holdback

            if len(text) > holdback:
                idx = 0
                for match in re.finditer(PAT, text):
                    if match.end() <= safe_len:
                        idx = match.end()

                for sp_tok in self.special_tokens:
                    # TODO: 优化查找特殊token的位置 find(sp_tok, start+1
                    pos = text.find(sp_tok)
                    if pos < idx and idx < pos + len(sp_tok):
                        idx = pos
                    elif pos + len(sp_tok) > safe_len:
                        idx = min(idx, pos)

                yield from self.encode(text[:idx])
                read_buffer = text[idx:]
            else :
                read_buffer = text

        if read_buffer:
            yield from self.encode(read_buffer)



