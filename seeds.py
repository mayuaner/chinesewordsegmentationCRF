import os, csv

class Seed(object):

    def __init__(self, path):
        self.path=path
        self.doclist = list(os.walk(path))[0][2]

    def get_seed_2tag(self, seed_number=3): #default to choose Peking University
        """this method transforms the training data from SIGHAN to 2-tag system files
        in this project, I use the training data from Peking University"""
        
        filepath = os.path.join(self.path, self.doclist[seed_number])
        file = open(filepath,'r', encoding = 'utf8')
        with open('seed_pku_2','w', encoding = 'utf8', newline = '') as fp:
            writer = csv.writer(fp, delimiter = ' ')
            for line in file:
                tl = []
                for i in range(len(line)):
                    if line[i]!= ' ' and line[i] != '\n':
                        if line[i] == '\u3002':
                            tl.append((line[i], 'O'))
                        elif i == 0 or line[i-1] == ' ' or line[i-1] == '\n':
                            tl.append((line[i], 'B'))
                        else:
                            tl.append((line[i], 'I'))
                writer.writerows(tl)

    def get_seed_4tag(self, seed_number = 3):   #default to choose Peking University
        """this method transforms the training data from SIGHAN to 4-tag system
        data from Peking University"""
        filepath = os.path.join(self.path, self.doclist[seed_number])
        file = open(filepath,'r', encoding = 'utf8')
        with open('seed_pku_4','w', encoding = 'utf8', newline = '') as fp:
            writer = csv.writer(fp, delimiter = ' ')
            for line in file:
                tl = []
                for i in range(len(line)):
                    if line[i]!= ' ' and line[i] != '\n':
                        if line[i] == '\u3002':
                            tl.append((line[i], 'O'))
                        elif (i == 0 or line[i-1] == ' ' or line[i-1] == '\n') and (line[i+1] == ' ' or line[i+1] == '\n'):
                            tl.append((line[i], 'S'))
                        elif i == 0 or line[i-1] == ' ' or line[i-1] == '\n':
                            tl.append((line[i], 'B'))
                        elif line[i+1] == ' ' or line[i+1] == '\n':
                            tl.append((line[i], 'E'))
                        else:
                            tl.append((line[i], 'M'))
                writer.writerows(tl)

if __name__ == '__main__':
        seed = Seed('training')
        seed.get_seed_4tag()
