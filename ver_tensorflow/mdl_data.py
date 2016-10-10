__author__ = 'jhlee'

import cPickle
import numpy as np
import csv
import sys
import time
import os.path

EVENT = {'Ev101': 1, 'Ev102': 2, 'Ev103': 3, 'Ev104': 4, 'Ev105': 5, 'Ev106': 6, 'Ev107': 7, 'Ev108': 8, 'Ev109': 9, 'Ev110': 0}
RATIO = {'Training': 0, 'Test': 1}

class YLIMED():
    def __init__(self, pathInfo, pathAud, pathImg):
        self.pathInfo = pathInfo
        self.pathAud = pathAud
        self.pathImg = pathImg
        self.VID = []
        self.LABEL = []
        self.SET = []
        self.TIME = []
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self.__summary_data()
        
        self._X = []
        self._Y = []
        if not (os.path.isfile('YLIMED_info.tmp')):
            self.__initial_data_info()
        else:
            self.__resummary_data()

    def __summary_data(self):
        # read from csv file and save VID, LABEL, SET(tr or te) data
        f = open(self.pathInfo, 'rb')
        inforeader = csv.reader(f)
        next(inforeader)

        trteratio = np.zeros(2)
        labelset = np.zeros(10)
        trainingset = np.zeros(10)
        testset = np.zeros(10)

        for info in inforeader:
            self.VID.append(info[0])
            self.LABEL.append(info[7])
            self.SET.append(info[13])

            labelset[EVENT[info[7]]] += 1
            trteratio[RATIO[info[13]]] += 1

            if info[13] == 'Training':
                trainingset[EVENT[info[7]]] += 1
            else:
                testset[EVENT[info[7]]] += 1
        f.close()
            
        print '-------------------------------------SUMMARY-------------------------------------'
        print 'Total            =', int(sum(labelset))
        print 'Tr / Te ratio    =', trteratio
        print 'Label set        =', labelset
        print 'Training set     =', trainingset
        print 'Test set         =', testset
        print '---------------------------------------------------------------------------------'
        
    def __resummary_data(self):
        # read from tmp file
        del(self.VID)
        del(self.LABEL)
        del(self.SET)
        self.VID = []
        self.LABEL = []
        self.SET = []

        tmpinfo = open('YLIMED_info.tmp', 'r')
        tmpdata = tmpinfo.readlines()

        trteratio = np.zeros(2)
        labelset = np.zeros(10)
        trainingset = np.zeros(10)
        testset = np.zeros(10)

        for tmp in tmpdata:
            tmp = tmp.split()
            if tmp[1] == 'MISS':
                continue
            
            self.VID.append(tmp[0])
            self.TIME.append(tmp[1])
            self.LABEL.append(tmp[2])
            self.SET.append(tmp[3])
            
            labelset[EVENT[tmp[2]]] += 1
            trteratio[RATIO[tmp[3]]] += 1

            if tmp[3] == 'Training':
                trainingset[EVENT[tmp[2]]] += 1
            else:
                testset[EVENT[tmp[2]]] += 1
        tmpinfo.close()
            
        print '-------------------------------------RESUMMARY-------------------------------------'
        print 'Total            =', int(sum(labelset))
        print 'Tr / Te ratio    =', trteratio
        print 'Label set        =', labelset
        print 'Training set     =', trainingset
        print 'Test set         =', testset
        print '---------------------------------------------------------------------------------'
        
    def __initial_data_info(self):
        print 'Initial data info...'
        f = open('YLIMED_info.tmp', 'w')

        starttime = time.time()
        total = len(self.VID)
        count = 0

        for i in range(0, len(self.VID)):
            #check both file exist
            temp_aud_file = self.pathAud + '/' + self.VID[i] + '.mfcc20.ascii'
            temp_img_file = self.pathImg + '/' + self.VID[i] + '.fc7.txt'

            #set progress bar
            count += 1
            progress = int(float(count) / float(total) * float(100))
            sys.stdout.write('\r')
            sys.stdout.write("[%-20s] %d%% %d sec" % ('='*(progress/5), progress, time.time() - starttime))
            sys.stdout.flush()
            
            #if exist missing file
            if not (os.path.isfile(temp_aud_file) and os.path.isfile(temp_img_file)):
                f.write(self.VID[i] + ' ' + 'MISS' + ' ' + self.LABEL[i] + ' ' + self.SET[i] + '\n')
                continue
            #open file
            audFile = open(temp_aud_file, 'r')
            audData = audFile.readlines()
            imgFile = open(temp_img_file, 'r')
            imgData = imgFile.readlines()

            #split sequence by shorter data
            if(cmp(len(imgData), len(audData)/100)>0):
                range_len=len(audData)/100
            else:
                range_len=len(imgData)

            f.write(self.VID[i] + ' ' + str(range_len) + ' ' + self.LABEL[i] + ' ' + self.SET[i] + '\n')
            imgFile.close()

        f.close()
        
    # -- get frame data for global randomly
    def __get_part_data(self, Aud_Img_Lab, tr_or_te):
        tmpinfo = open('YLIMED_info.tmp', 'r')
        tmpdata = tmpinfo.readlines()
        starttime = time.time()
        total = len(tmpdata)
        count = 0
        output = []
        for line in tmpdata:
            count += 1
            progress = int(float(count) / float(total) * float(100))
            sys.stdout.write('\r')
            sys.stdout.write("[%-20s] %d%% %d sec" % ('='*(progress/5), progress, time.time() - starttime))
            sys.stdout.flush()

            line = line.split()
            tVID = line[0]
            range_len = line[1]
            if range_len == 'MISS':
                continue
            set = self.SET[self.VID.index(tVID)]

            if set != tr_or_te:
                continue

            if Aud_Img_Lab == 'Lab':
                for i in range(int(range_len)):
                    label = EVENT[self.LABEL[self.VID.index(tVID)]]
                    output.append(label)
            else:
                if Aud_Img_Lab == 'Aud':
                    temp_file = self.pathAud + '/' + tVID + '.mfcc20.ascii'
                    f = open(temp_file, 'r')
                    data = f.readlines()
                    f.close()
                elif Aud_Img_Lab == 'Img':
                    temp_file = self.pathImg + '/' + tVID + '.fc7.txt'
                    f = open(temp_file, 'r')
                    data = f.readlines()
                    f.close()

                for i in range(int(range_len)):
                    add = []
                    if Aud_Img_Lab == 'Aud':
                        for j in range(100):
                            add += [float(x) for x in data[i*100+j].split()]
                    elif Aud_Img_Lab == 'Img':
                        add = [float(x) for x in data[i].split()]

                    output.append(add)
        tmpinfo.close()
        output = np.asarray(output)
        print ', finish'
        return output
    
    # -- get frame data per vid
    def get_vid_data(self, line, Aud_Img_Lab):
        output = []
        line = line.split()
        tVID = line[0]
        range_len = line[1]
        
        if Aud_Img_Lab == 'Aud':
            temp_file = self.pathAud + '/' + tVID + '.mfcc20.ascii'
            f = open(temp_file, 'r')
            data = f.readlines()
            f.close()
        elif Aud_Img_Lab == 'Img':
            temp_file = self.pathImg + '/' + tVID + '.fc7.txt'
            f = open(temp_file, 'r')
            data = f.readlines()
            f.close()
        else:
            print "Error in get_vid_data"
            raise SystemExit

        for i in range(int(range_len)):
            add = []
            if Aud_Img_Lab == 'Aud':
                for j in range(100):
                    add += [float(x) for x in data[i*100+j].split()]
            elif Aud_Img_Lab == 'Img':
                add = [float(x) for x in data[i].split()]
            output.append(add)
        output = np.asarray(output)
        return output
        
    #Source reference: https://github.com/aymericdamien/TensorFlow-Examples.git/input_data.py
    def next_batch(self, X, Y, batch_size, total_size):
        """Return the next `batch_size` examples from this data set."""
        finish = False
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > total_size:
            # Finished epoch
            self._epochs_completed += 1
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= total_size
            finish = True
        end = self._index_in_epoch
        return X[start:end], Y[start:end], finish
    
    def next_batch_multi(self, X_1, X_2, Y, batch_size, total_size):
        """Return the next `batch_size` examples from this data set."""
        finish = False
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > total_size:
            # Finished epoch
            self._epochs_completed += 1
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= total_size
            finish = True
        end = self._index_in_epoch
        return X_1[start:end], X_2[start:end], Y[start:end], finish

    def get_aud_X_train(self):
        print 'Load Training Audio Data'
        return self.__get_part_data('Aud', 'Training')
    def get_aud_X_test(self):
        print 'Load Test Audio Data'
        return self.__get_part_data('Aud', 'Test')
    def get_img_X_train(self):
        print 'Load Training Image Data'
        return self.__get_part_data('Img', 'Training')
    def get_img_X_test(self):
        print 'Load Test Image Data'
        return self.__get_part_data('Img', 'Test')
    def get_y_train(self):
        print 'Load Training Label Data'
        return self.__get_part_data('Lab', 'Training')
    def get_y_test(self):
        print 'Load Test Label Data'
        return self.__get_part_data('Lab', 'Test')

    def get_vid_info(self, tr_or_te):
        f = open('YLIMED_info.tmp', 'r')
        lines = f.readlines()
        out = []
        for tmp in lines:
            tmp = tmp.split()
            if tmp[1] == 'MISS':
                continue
            if tmp[3] == tr_or_te:
                tmp[2] = str(EVENT[tmp[2]])
                
                line = tmp[0] + ' ' + tmp[1] + ' ' + tmp[2] + ' ' + tmp[3]
                out.append(line)
        out = np.asarray(out)
        return out
        
if __name__ == '__main__':
    data = YLIMED('YLIMED_info.csv', '/DATA/YLIMED150924/audio/mfcc20', '/DATA/YLIMED150924/keyframe/fc7')
    
    vtr = data.get_vid_info('Training')
    vte = data.get_vid_info('Test')
    print vtr[1]
    #test = data.get_vid_data(vtr[1], 'aud')
    
    #print test.shape
    
    '''
    y_train = data.get_y_train()
    X_img_train = np.zeros(len(y_train))
    X_aud_train = np.zeros(len(y_train))

    y_test = data.get_y_test()
    X_img_test = np.zeros(len(y_test))
    X_aud_test = np.zeros(len(y_test))
    
    total_batch = int(len(y_train)/32)
    for i in range(total_batch):
        batch_x_aud, batch_x_img, batch_ys, finish = data.next_batch_multi(X_aud_train, X_img_train, y_train, 32, len(y_train))
    for i in range(int(len(y_test))):
        batch_x_aud, batch_x_img, batch_ys, finish = data.next_batch_multi(X_aud_test, X_img_test, y_test, 1, len(y_test))
    '''