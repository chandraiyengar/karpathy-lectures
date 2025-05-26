words = open("names.txt", "r").read().splitlines()
b = {}
for word in words[:1]:
    chs = ["<S>"] + list(word) + ["<E>"]
    for ch1, ch2 in zip(chs, chs[1:]):
        bigram = (ch1, ch2)
        b[bigram] = b.get(bigram, 0) + 1

sorted(b.items(), key=lambda x: x[1], reverse=True)

print(b)

import torch

a = torch.zeros((3, 5), dtype=torch.int32)
print(a)
