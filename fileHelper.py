import csv


def read(fileName):
    with open(fileName, 'rb') as f:
        reader = csv.reader(f, delimiter=' ')
        lines = list(reader)

    return [map(float, line) for line in lines]


def writeToFile(lst, fileName):
    with open(fileName, 'wb') as f:
        for line in lst:
            f.write('%s\n' % ' '.join(line))
