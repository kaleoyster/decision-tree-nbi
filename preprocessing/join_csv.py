import pandas as pd

def combine(filename1, filename2, column):
    df1 = pd.read_csv(filename1)
    df2 = pd.read_csv(filename2)

    df1[column] = ['paved']*len(df1)
    df2[column] = ['gravel']*len(df2)

    df = pd.concat([df1, df2])
    return df


def main():
    file1 = 'nebraska_paved.csv'
    file2 = 'nebraska_gravel.csv'
    df = combine(file1, file2, 'deckSurface')
    df.to_csv('nebraska_combined.csv')

main()
