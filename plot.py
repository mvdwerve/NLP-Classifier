from matplotlib import pyplot as plt
from collections import Counter

# make the dictionaries, tags and words
pos_dict = {}
pos_tags = {}
sentences = []
target_words = []

with open('ted_dict_words_pos.txt') as f:
    content = f.readlines()

    for line in content:
        line = line.strip()
        words = line.split(' ')
        for word in words:
            tagged_word = word.split('/')
            if len(tagged_word) > 1:
                if pos_tags.get(tagged_word[1]) == None:
                    pos_tags[tagged_word[1]] = len(pos_tags)
                pos_dict[tagged_word[0]] = tagged_word[1]

def get_pos_tag(index):
    return [k for (k, v) in list(pos_tags.items()) if v == index]

with open('ted_lm/to_run/data/train.txt') as f:
    content = f.readlines()

    for line in content:
        if len(line.split(' ')) > 2:
            sentences.append(' '.join(line.split(' ')[:-2]))
            target_words.append(line.split(' ')[-2])


co = 8
count = Counter(pos_dict.values())
items = list(sorted(count.items(), key=lambda x: -x[1]))
labels, sizes = zip(*items[:co])
labels = labels + ('Other',)
sizes = sizes+ (sum([x[1] for x in items[co:]]),)


# Plot
plt.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
plt.title('Frequency of POS text in training set')

plt.axis('equal')
plt.show()
