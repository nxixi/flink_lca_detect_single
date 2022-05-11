from abc import abstractmethod


class AbstractAbnormalDetect(object):
    """
    原子异常检测算法必须是实现该类，才能与spark集成
    """
    @abstractmethod
    def version(self):
        """
        返回该算法的版本，使用字符串表示的3位点分版本号。如"3.2.1"
        """
        pass

    @abstractmethod
    def configure(self, dict):
        """
        configure the class once
        :param dic: dict that contains string-string like configuration
        :return: no need to return
        """
        pass

    @abstractmethod
    def transform(self, *df):
        """
        transform方法接收至少一个pandas.DataFrame作为数据参数，params是一个表示参数的对象。输出为一个pandas.DataFrame。
        :param df: 至少一个的pandas.DataFrame。使用df[0]获取第一个数据集，绝大多数情况下只有一个数据集，除非算法需要多个不同的数据集一起参与计算。
        df至少需要两列：时间列和值列。时间列字段名约定为'@timestamp', 值列字段名约定为'@value'（之所以用@开头是为了避免与其他业务字段冲突）。时间列与值列的位置不确定，请不要使用列索引来检索。
        除时间列和值列外，可能还会有其他业务字段列。如算法需要进行数据填补，请将业务字段列填充为null
        :return: 一个pandas.DataFrame，以及一个二进制模型，组成的二元组
        输出结果只能在源df的右侧添加新的字段，不能将新字段插入到中间或者最左边。不能删除源df中的任何一个字段（包括业务字段），也不能移动源df中字段的位置。
        """
        pass

    @abstractmethod
    def fields(self):
        """
        返回一组元组列表，说明算法将添加的列，及其数据类型。元组列表的顺序应该与实际添加的字段从左到右保持一致。
        如果mode_append是False的话。返回算法会输出的所有列，系统将完全抛弃原始的数据列，只保留算法输出的列
        字段的类型包括int, float, double, boolean, long, str
        :return: 元组列表。
        例如：[('abnormal', 'float'),('is_high', 'boolean')]意味着算法将在源df的右侧依次添加float类型的abnormal字段、boolean类型的is_high字段
        """
        pass

    @abstractmethod
    def mode_append(self):
        """
        算法输出结果是否是增加字段模式的，默认是True
        对于类似根因定位、融合算法应属于全量列模式，应返回False
        """
        return True