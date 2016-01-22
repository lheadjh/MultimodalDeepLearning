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
        self.__summary_data()

    def __summary_data(self):
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

        print '-------------------------------------SUMMARY-------------------------------------'
        print 'Total            =', int(sum(labelset))
        print 'Tr / Te ratio    =', trteratio
        print 'Label set        =', labelset
        print 'Training set     =', trainingset
        print 'Test set         =', testset
        print '---------------------------------------------------------------------------------'

        f.close()

    def __initial_data_info(self):
        print 'Initial data info...'
        f = open('YLIMED_info.tmp', 'w')

        starttime = time.time()
        total = len(self.VID)
        count = 0

        for tVID in self.VID:
            #check both file exist
            temp_aud_file = self.pathAud + '/' + tVID + '.mfcc20.ascii'
            temp_img_file = self.pathImg + '/' + tVID + '.fc7.txt'
            if not (os.path.isfile(temp_aud_file) and os.path.isfile(temp_img_file)):
                total -= 1
                continue
            #set progress bar
            count += 1
            progress = int(float(count) / float(total) * float(100))
            sys.stdout.write('\r')
            sys.stdout.write("[%-20s] %d%% %d sec" % ('='*(progress/5), progress, time.time() - starttime))
            sys.stdout.flush()
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

            f.write(tVID + ' ' + str(range_len) + '\n')
            imgFile.close()

        f.close()

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

#TODO
        # print 'Pickling...'
        #
        # y_train = np.asarray(y_train)
        # y_test = np.asarray(y_test)
        # print 'y_train num:     %d' % len(y_train), y_train.shape
        # print 'y_test num:      %d' % len(y_test), y_test.shape
        # f = open('YLDMED_y.pkl', 'wb')
        # temp = y_train, y_test
        # cPickle.dump(temp, f)
        # f.close()
        #
        # aud_X_train = np.asarray(aud_X_train)
        # aud_X_test = np.asarray(aud_X_test)
        # print 'aud_X_train num: %d' % len(aud_X_train), aud_X_train.shape
        # print 'aud_X_test num:  %d' % len(aud_X_test), aud_X_test.shape
        # f = open('YLDMED_aud_X.pkl', 'wb')
        # temp = aud_X_train, aud_X_test
        # cPickle.dump(temp, f)
        # f.close()
        #
        # img_X_train = np.asarray(img_X_train)
        # img_X_test = np.asarray(img_X_test)
        # print 'img_X_train num: %d' % len(img_X_train), img_X_train.shape
        # print 'img_X_test num:  %d' % len(img_X_test), img_X_test.shape
        # f = open('YLDMED_img_X.pkl', 'wb')
        # temp = img_X_train, img_X_test
        # cPickle.dump(temp, f)
        # f.close()

if __name__ == '__main__':
    data = YLIMED('YLIMED_info.csv', '../YLIMED150924/audio/mfcc20', '../YLIMED150924/keyframe/fc7')
    print data.get_aud_X_test().shape
