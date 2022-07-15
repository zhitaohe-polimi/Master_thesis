g = open('kg_final.txt_post', 'w')

with open('kg_final.txt') as f:
    for line in f:
        th = line.split('\t')
        g.write(' '.join(th))

g.close()

from collections import defaultdict
interlist = defaultdict(list)
with open('ratings_final.txt') as f:
    for line in f:
        th = line.split('\t')
        if int(th[2]) == 1:
            interlist[th[0]].append(th[1])
import random
g1 = open('train.txt', 'w')
g2 = open('test.txt', 'w')

for k in interlist:
    #print(interlist[k])
    random.shuffle(interlist[k])
    l = len(interlist[k])
    sp = int(l*0.8)
    tr = sorted(interlist[k][:sp])
    te = sorted(interlist[k][sp:])
    g1.write(k)
    for x in tr:
        g1.write(' '+x)
    g1.write('\n')
    g2.write(k)
    for x in te:
        g2.write(' '+x)
    g2.write('\n')

g1.close()
g2.close()
