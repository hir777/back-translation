import sys

if __name__ == '__main__':
    for x in sys.stdin:
        x = ''.join(x.split()).replace('▁', ' ')
        x = x.strip()
        print(x)
