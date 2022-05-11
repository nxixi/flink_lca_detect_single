import json


def load(path):

    jsonStr = ""
    with open(path,'r') as file:
        for line in file.readlines():
            jsonStr = jsonStr + line

    return json.loads(jsonStr)

if __name__ == '__main__':
    path = '/Users/zhouwenyang/Desktop/alg_data/kernel.json'

    result = load(path)
    print(result['pre_value'])
