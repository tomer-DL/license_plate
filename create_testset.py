import random

indices = random.choices(range(1,666), k=25)
print(indices)

f = open("Israeli license plates.json", "r")
out1 = open("license plates-main.json", "w")
out2 = open("license plates-test.json", "w")
a = 1
for line in f:
    if a in indices:
        out2.write(line)
    else:
        out1.write(line)
    a += 1
out2.close()
out1.close()
f.close()
