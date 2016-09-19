__author__ = 'ngot0008'

fname_t_train = ['cls-acl10-processed/de/books/train.processed',
                 'cls-acl10-processed/de/dvd/train.processed',
                 'cls-acl10-processed/de/music/train.processed']
lines = [100,400]
with open('target_test.processed','w') as output:
    for f in range(0,len(fname_t_train)):
        with open(fname_t_train[f], 'r') as input:
            count = 0
            for line in input:
                count +=1
                if count<= lines[0]:
                    continue
                elif count > lines[1]:
                    break
                else:
                    tokens = line.rstrip().split(' ')
                    s, label = tokens[-1].split(':')
                    assert s == "#label#"
                    tokens[-1] = "#label#:"+str(f)
                    output.write(' '.join(tokens)+'\n')
