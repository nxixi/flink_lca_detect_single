from matplotlib import pyplot as plt

def draw(data:dict):
    x = data['time']
    value = data['pre_value']
    upper = data['upper']
    lower = data['lower']

    plt.plot(x, value, ".-b")
    plt.plot(x, upper, ".-g")
    plt.plot(x, lower, ".-r")
    plt.show()


