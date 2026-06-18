import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# 匹配字符串得,按照special_tokens分割,得到match_str
def get_split_target_str(data_dir : str, special_tokens : list[str]) -> list[str]:
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

# 获取pair的次数,只在初始化调用一次
def get_pair_counts(pretoken_seq : dict[bytes, list[int]], pretoken_counts : dict[bytes, int]) -> dict[tuple[int, int], int]:
    pair_counts = {}
    for key, val in pretoken_seq.items():
        for i in range(len(val) - 1):
            pair = (val[i], val[i+1])
            pair_counts[pair] = pair_counts.get(pair, 0) + pretoken_counts[key]

    return pair_counts

# 获取频数最高的pair
def get_best_pair(pair_counts : dict[tuple[int, int], int], vocab : dict[int, bytes]) -> tuple[tuple[int, int], int]:
    return max(pair_counts.items(), key=lambda item: (item[1], vocab[item[0][0]], vocab[item[0][1]]))

# 更新 pretoken_seq
def update_pretoken_seq(pretoken_seq : dict[bytes, list[int]], pretoken_counts : dict[bytes, int], best_pair : tuple[int, int], new_id : int) -> tuple[dict[bytes, list[int]], dict[tuple[int, int], int]]:
    new_pair_counts : dict[tuple[int, int], int] = {}
    for key, seq in pretoken_seq.items():
        for i in range(len(seq) - 2, -1, -1):
            if seq[i] == best_pair[0] and seq[i+1] == best_pair[1]:
                # seq = merge_one_pair(seq, best_pair, new_id)
                j = 0
                while j < len(seq):
                    if j < len(seq) - 1 and seq[j] == best_pair[0] and seq[j+1] == best_pair[1]:
                        if (j > 0): 
                            # reduce pair_counts for the pair that is merged
                            pair_reduce = (seq[j-1], seq[j])
                            new_pair_counts[pair_reduce] = new_pair_counts.get(pair_reduce, 0) - pretoken_counts[key]

                            # add pair_counts for the new pair
                            pair_add = (seq[j-1], new_id)
                            new_pair_counts[pair_add] = new_pair_counts.get(pair_add, 0) + pretoken_counts[key]

                        if (j < len(seq) - 2):
                            # reduce pair_counts for the pair that is merged
                            pair_reduce = (seq[j+1], seq[j+2])
                            new_pair_counts[pair_reduce] = new_pair_counts.get(pair_reduce, 0) - pretoken_counts[key]

                            # add pair_counts for the new pair
                            pair_add = (new_id, seq[j+2])
                            new_pair_counts[pair_add] = new_pair_counts.get(pair_add, 0) + pretoken_counts[key]
                        
                        # update pretoken_seq
                        seq[j] = new_id
                        del seq[j+1]
                    j += 1
                pretoken_seq[key] = seq
                break
    return pretoken_seq, new_pair_counts

# 更新 pair_counts
def update_pair_counts(best_pair : tuple[int, int], pair_counts : dict[tuple[int, int], int], new_pair_counts : dict[tuple[int, int], int]) -> dict[tuple[int, int], int]:
    del pair_counts[best_pair]
    for pair, count in new_pair_counts.items():
        pair_counts[pair] = pair_counts.get(pair, 0) + count
        if pair_counts[pair] == 0:
            del pair_counts[pair]
    return pair_counts

def train_bpe(data_dir : str, special_tokens : list[str], vocab_size : int) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
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

        pretoken_seq , new_pair_counts= update_pretoken_seq(pretoken_seq, pretoken_counts, best_pair, new_id)

        pair_counts = update_pair_counts(best_pair, pair_counts, new_pair_counts)

    return vocab, merges


