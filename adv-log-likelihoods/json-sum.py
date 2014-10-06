import argparse
import sys
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('var')
    var = parser.parse_args().var
    print sum(json.loads(sys.stdin.read())[var])
