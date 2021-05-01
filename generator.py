import random
import numpy as np
import json



bag_of_words = [["the", "at", "my", "a", "how", "when", "is", "it", "with"], ["element", "science", "chemistry", "planet", "electricity", "physics", "voltage", "scientists", "research", "innovation", "hydrogen"], ["nation", "border", "economy", "power", "president", "strength", "government", "power"], ["java", "computer", "compile", "tech", "code", "editor", "runtime", "search", "graph", "tree"]]
bag_of_words = np.array([np.array(i) for i in bag_of_words], dtype=object)
def gen_doc(distribution, length):
    samples = np.random.multinomial(length, distribution)
    out = [ bag_of_words[i][random.choices(range(len(bag_of_words[i])), k=samples[i])]  for i in range(len(bag_of_words)) ]
    return np.concatenate([i for i in out]).tolist()

arr = []
for i in range(200):
    arr.append(gen_doc([0.5, 0.45, 0.025, 0.025], 155))
    arr.append(gen_doc([0.5, 0.025, 0.45, 0.025], 155))
    arr.append(gen_doc([0.5, 0.025, 0.025, 0.4], 155))

with open("out.json", 'w') as outfile:
    json.dump(arr, outfile)