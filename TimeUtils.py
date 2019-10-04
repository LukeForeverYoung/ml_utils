import time
class TimeCoster():

    def __init__(self):
        self.__time_tick=time.time()
        self.__time_start=time.time()
    def start(self):
        self.__time_tick = time.time()

    def tock(self):
        time_cost=time.time() - self.__time_tick
        self.__time_tick=time.time()
        return time_cost

    def cost_from_start(self):
        return time.time() - self.__time_start
if __name__ == '__main__':
    coster=TimeCoster()
    time.sleep(5)
    print(coster.tock())
    time.sleep(5)
    print(coster.tock())
    print(coster.cost_from_start())
