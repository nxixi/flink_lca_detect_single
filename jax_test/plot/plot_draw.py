import json
from matplotlib import pyplot as plt

if __name__ == '__main__':
    x = []
    value = []
    upper = []
    lower = []

    i = 0


    with open("/Users/zhouwenyang/Desktop/result.txt","r") as f:
        for line in f:
            if i>2880:
                user_dic = json.loads(line)
                print(user_dic)
                x.append(user_dic["@timestamp"])
                value.append(user_dic["pre_value"])
                upper.append(user_dic["upper"])
                lower.append(user_dic["lower"])

            i = i+1

    plt.plot(x, value, ".-b")
    plt.plot(x, upper, ".-g")
    plt.plot(x, lower, ".-r")
    plt.show()