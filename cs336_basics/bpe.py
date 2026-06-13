from typing import Any


import regex as re

data_dir = "/Users/pl/project/assignment1-basics/tests/fixtures/corpus.en"

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# 匹配字符串得,按照special_tokens分割,得到match_str
def get_split_target_str(data_dir, special_tokens) -> list[str]:
    target_str_list = []
    try:
        with open(data_dir, "r", encoding="utf-8") as file:
            target_str = file.read()
            if not special_tokens:
                target_str_list.append(target_str)
                return target_str_list
            escaped = "|".join(re.escape(token) for token in special_tokens)
            pattern = re.compile(f"{escaped}")
            parts = pattern.split(target_str)
            target_str_list = [p for p in parts if p != ""]
            return target_str_list
    except FileNotFoundError:
        print(f"File {data_dir} not found")
        exit(1)




# 获取pair的次数
def get_pair_counts(pretoken_seq, pretoken_counts) -> dict[tuple[int, int], int]:
    pair_counts = {}

    for key, val in pretoken_seq.items():
        for i in range(len(val) - 1):
            pair = (val[i], val[i+1])
            pair_counts[pair] = pair_counts.get(pair, 0) + pretoken_counts[key]

    return pair_counts

# 获取频数最高的pair
def get_best_pair(pair_counts, vocab) -> tuple[tuple[int, int], int]:
    return max(pair_counts.items(), key=lambda item: (item[1], vocab[item[0][0]], vocab[item[0][1]]))

# merge pair，为更新pretoken_seq做准备
def merge_one_pair(list, best_pair, new_id) -> list[int]:
    new_list = []
    i = 0
    while i < len(list):
        if i < len(list) - 1 and list[i] == best_pair[0] and list[i+1] == best_pair[1]:
            new_list.append(new_id)
            i += 2
        else:
            new_list.append(list[i])
            i += 1
    return new_list

# 更新 pretoken_seq
def update_pretoken_seq(pretoken_seq, best_pair, new_id) -> dict[bytes, list[int]]:
    for key, seq in pretoken_seq.items():
        seq = merge_one_pair(seq, best_pair, new_id)
        pretoken_seq[key] = seq
    return pretoken_seq

# 更新 pair_counts
def update_pair_counts(pretoken_seq, pretoken_counts) -> dict[tuple[int, int], int]:
    return get_pair_counts(pretoken_seq, pretoken_counts)


def train_bpe(data_dir, special_tokens, vocab_size) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    target_str_list = get_split_target_str(data_dir, special_tokens)

    pretoken_counts = {}
    pretoken_seq = {}
    pair_counts = {}
    
    vocab = {i: bytes([i]) for i in range(256)} 

    # 处理special_tokens
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")
    merges = []
    
    for target_str in target_str_list:
        matches = re.finditer(PAT, target_str)

        for match in matches:
            key = match.group(0).encode("utf-8")
            pretoken_counts[key] = pretoken_counts.get(key, 0) + 1
            pretoken_seq[key] = list[int](key)
    
    pair_counts = get_pair_counts(pretoken_seq, pretoken_counts)


    while len(vocab) < vocab_size:
        best_pair, best_count = get_best_pair(pair_counts, vocab)
        
        merges.append((vocab[best_pair[0]], vocab[best_pair[1]]))

        new_id = len(vocab)

        vocab[new_id] = vocab[best_pair[0]] + vocab[best_pair[1]]

        update_pretoken_seq(pretoken_seq, best_pair, new_id)

        pair_counts = update_pair_counts(pretoken_seq, pretoken_counts)

    return vocab, merges

def main():
    special_tokens=["<|endoftext|>"]

    vocab, merges = train_bpe(data_dir, special_tokens, vocab_size=500)

    print(f"vocab_size: {len(vocab)}")
    print(f"merges: {len(merges)}")

    return vocab, merges

if __name__ == "__main__":
    main()

