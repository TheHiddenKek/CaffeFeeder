import re, csv, colorama, caffe, time, os
import itertools as it
import numpy as np
from caffe.proto import caffe_pb2
from google.protobuf import text_format
from PIL import Image
import matplotlib.pyplot as plt


def loadSolver(fileName):
    solverFile = 'solver-mnist.prototxt'
    solverParam = caffe_pb2.SolverParameter()
    text_format.Merge(open(solverFile).read(), solverParam)
    # net parameter
    netFile = solverParam.train_net
    netParam = caffe_pb2.NetParameter()
    text_format.Merge(open(netFile).read(), netParam)

    # model storage
    outDir = solverParam.snapshot_prefix
    if not os.path.isdir(outDir):
        os.mkdir(outDir)

    return solverParam, netParam



def MaxValue(filename):
    file = open(filename)

    splitter = re.compile(r",")

    Max = 0

    for i in file:
        LineMax = max([float(i) for i in splitter.split(i)[:-1]])
        if Max < LineMax:
            Max = LineMax

    file.close()

    return Max



def Batches(filename, dim, batch_size):
    Max = MaxValue(filename)

    file = open(filename)

    splitter = re.compile(r',')

    lines = sum(1 for i in file)

    file.seek(0)

    if lines % batch_size != 0:
        print(colorama.Fore.YELLOW + "Num of lines is not multiple of batch size" + colorama.Style.RESET_ALL)

    for i in range(lines // batch_size):
        inputs, labels = [], []
        for j in it.islice(file, 0, batch_size):

            Element = np.array(splitter.split(j), dtype=np.float32)

            inputs.append(Element[:-1] / Max)

            labels.append(Element[-1])

        yield (np.array(inputs).reshape(dim), np.array(labels))

    file.close()


def Train(solverFile, trainFile, testFile=None, maxEpoch=10000, mode="gpu"):

    if mode == "gpu":
        caffe.set_mode_gpu()
    elif mode == "cpu":
        caffe.set_mode_cpu()
    else:
        raise Exception("Unknown mode")

    solver = caffe.get_solver(solverFile)

    solverParam, netParam = loadSolver(solverFile)

    if netParam.layer[0].type == 'MemoryData':
        batch_size = netParam.layer[0].memory_data_param.batch_size
        channels = netParam.layer[0].memory_data_param.channels
        height = netParam.layer[0].memory_data_param.height
        width = netParam.layer[0].memory_data_param.width
        inputDim = [batch_size, channels, height, width]


    iterations = 0

    losses = []
    accuracies = []

    for epoch in range(maxEpoch):

        if solverParam.max_iter <= iterations:
            break

        print("[Train]:{0} epoch".format(epoch))

        epoch_loss = []

        start = time.clock()

        for batch in Batches(trainFile, inputDim, batch_size):

            if solverParam.max_iter <= iterations:
                break

            solver.net.set_input_arrays(batch[0], batch[1])

            solver.step(1)

            loss = float(solver.net.blobs["loss"].data)
            accuracy = float(solver.net.blobs["accuracy"].data)

            epoch_loss.append(loss)

            losses.append(loss)

            accuracies.append(accuracy)

            iterations += 1

        stop = time.clock() - start

        print("[Train]: Epoch:{0}\nTrain time {1:d}:{2:d}\nLoss:{3}".format(epoch, int(stop // 60), int(stop % 60), np.mean(epoch_loss)))

        if testFile != None:
            testAcc = []
            testLoss = []

            for batch in Batches(testFile, inputDim, batch_size):
                solver.net.set_input_arrays(batch[0], batch[1])

                loss = float(solver.net.blobs["loss"].data)
                accuracy = float(solver.net.blobs["accuracy"].data)

                testLoss.append(loss)
                testAcc.append(accuracy)

            print("[Test]: Test accuracy:{0} Test loss:{1}".format(np.mean(testAcc), np.mean(testLoss)))

    plt.plot(losses)
    plt.plot(accuracies)
    plt.show()
    return solver



TestSolver = Train("solver-mnist.prototxt", "train.csv", "test.csv", maxEpoch=100)

