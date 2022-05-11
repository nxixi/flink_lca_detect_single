from ..action.flink_action import Transporter
from ..statistic.Statistic import Statistic

if __name__ == '__main__':

    s = Statistic()
    conf = {
        "time_field":"time",
        "value_field":"value"
    }

    transporter = Transporter()
    transporter.transport(conf,s)