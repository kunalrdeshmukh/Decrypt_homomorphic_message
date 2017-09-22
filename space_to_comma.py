f1 = open('Zodiac.txt', 'r')
f2 = open('Zodiacs.txt.tmp', 'w')
for line in f1:
    f2.write(line.replace(' ', ','))
f1.close()
f2.close()