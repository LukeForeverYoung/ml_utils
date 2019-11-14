class BatchDataLoader():
    def __init__(self,data,batch_size):
        self.__data=list(data)
        self.__batch_size=batch_size
        assert batch_size<=len(self.__data)
        self.__reset()

    def __reset(self):
        from random import shuffle
        self.__batch_pos=0
        self.__stop_flag=False
        shuffle(self.__data)

    def __iter__(self):
        return self

    def __next__(self):
        from random import sample
        if self.__stop_flag==True:
            self.__reset()
            raise StopIteration
        self.__batch_pos+=1
        if self.__batch_pos*self.__batch_size<=len(self.__data):
            return self.__data[(self.__batch_pos-1)*self.__batch_size:self.__batch_pos*self.__batch_size]
        else:
            tmp=self.__data[(self.__batch_pos-1)*self.__batch_size:]
            lake_num=self.__batch_pos*self.__batch_size-len(self.__data)
            tmp.extend(sample(self.__data,lake_num))
            self.__stop_flag=True
            return tmp

if __name__=='__main__':
    data=[i for i in range(100)]
    loader=BatchDataLoader(data,32)
    for batch in loader:
        print(batch)
    for batch in loader:
        print(batch)

