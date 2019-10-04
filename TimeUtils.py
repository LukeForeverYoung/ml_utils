import time
class TimeCoster():

    def __init__(self):
        self.__time_tick=time.time()
        self.__time_start=time.time()
    def start(self):
        self.__time_tick = time.time()

    def tock(self,print_s=False):
        time_cost=time.time() - self.__time_tick
        self.__time_tick=time.time()
        if print_s:
            self.print_time(time_cost)
        return time_cost

    def cost_from_start(self,print_s=False):
        time_cost=time.time() - self.__time_start
        if print_s:
            self.print_time(time_cost)
        return time_cost

    def print_time(self,s):
        print('cost {:.2g} s'.format(s))
if __name__ == '__main__':
    coster=TimeCoster()
    time.sleep(5)
    coster.tock(print_s=True)
    time.sleep(5)
    coster.tock(print_s=True)
    coster.cost_from_start(print_s=True)
