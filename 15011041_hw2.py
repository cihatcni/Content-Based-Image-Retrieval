import cv2
import numpy as np
import pickle


class Image:
    def __init__(self, imageName, blue, green, red, lbp):
        self.imageName = imageName
        self.red = red
        self.green = green
        self.blue = blue
        self.lbp = lbp


class Difference:
    def __init__(self, imageName, diff):
        self.imageName = imageName
        self.diff = diff

    def __lt__(self, other):
        return self.diff < other.diff


class DifferenceList:
    def __init__(self, colorList: list, lbpList: list, colorLBPList: list):
        colorList.sort()
        lbpList.sort()
        colorLBPList.sort()
        self.colorList = colorList[0:5]
        self.lbpList = lbpList[0:5]
        self.colorLBPList = colorLBPList[0:5]


# Renk histogramını oluşturur.
def colorHistogramCreator(image):
    col = np.size(image[0]) // 3
    row = np.size(image) // (col * 3)
    hist = np.zeros(shape=(3, 256))
    print("IMAGE SIZE : ", row, "x", col)
    for i in range(0, row):
        for j in range(0, col):
            pixel = image[i, j]
            hist[0, pixel[0]] += 1
            hist[1, pixel[1]] += 1
            hist[2, pixel[2]] += 1
    return hist


# Renk histogramını 0-1 arası normalize eder.
def colorHistogramNormalizer(hist):
    minValue = np.zeros(shape=3)
    maxValue = np.zeros(shape=3)
    # RGB değerleri için min max bulunur.
    for i in range(0, 3):
        minValue[i] = np.amin(hist[i])
        maxValue[i] = np.amax(hist[i])

    for i in range(0, 3):
        for j in range(0, 256):
            hist[i, j] = (hist[i, j] - minValue[i]) / (maxValue[i] - minValue[i])
    return hist


# Doku histogramını oluşturur.
def textureHistogramCreatorLBP(image):
    col = np.size(image[0]) // 3
    row = np.size(image) // (col * 3)
    hist = np.zeros(shape=256)
    # Görseldeki çerçeve pikseller haricinde her piksel için
    # hesaplama yapılır.
    for i in range(1, row - 1):
        for j in range(1, col - 1):
            color = sum(image[i][j])
            power = 7
            total = 0
            for x in range(i - 1, i + 2):
                for y in range(j - 1, j + 2):
                    if x != i or y != j:
                        colorTemp = sum(image[x][y])
                        if color < colorTemp:
                            total += pow(2, power)
                        power -= 1
            hist[total] += 1
    return hist


# Doku histogramını 0-1 arası normalize eder.
def textureHistogramNormalizer(hist):
    minValue = np.amin(hist)
    maxValue = np.amax(hist)
    for i in range(0, 256):
        hist[i] = (hist[i] - minValue) / (maxValue - minValue)
    return hist


# İki resim arasındaki farkı bulur.
def diffBetweenTwoImages(image1: Image, image2: Image):
    diffR = diffG = diffB = diffLBP = 0
    for i in range(0, 256):
        diffB += abs(image1.blue[i] - image2.blue[i])
        diffG += abs(image1.green[i] - image2.green[i])
        diffR += abs(image1.red[i] - image2.red[i])
        diffLBP += abs(image1.lbp[i] - image2.lbp[i])
    diffRGB = Difference(image2.imageName, diffB + diffR + diffG)
    diffLBP = Difference(image2.imageName, diffLBP)
    return diffRGB, diffLBP


# Dosya yolu verilen görseli okur. Histogramlarını oluşturur.
def readImage(imgPath):
    print("READING IMAGE : " + imgPath)
    image = cv2.imread(imgPath, cv2.IMREAD_COLOR)
    colorHist = colorHistogramCreator(image)
    colorHist = colorHistogramNormalizer(colorHist)
    textureHist = textureHistogramCreatorLBP(image)
    textureHist = textureHistogramNormalizer(textureHist)
    if imgPath[5] != 't':  # train image ise
        return Image(imgPath[6:], colorHist[0], colorHist[1], colorHist[2], textureHist)
    else:  # test iamge ise
        return Image(imgPath[5:], colorHist[0], colorHist[1], colorHist[2], textureHist)


# Train için kullanılacak görselleri okur.
def readTrainImages():
    imageList = []
    for i in range(1, 71):
        if i < 10:
            imageNum = "0" + str(i)
        else:
            imageNum = str(i)
        image = readImage("train/train (" + imageNum + ").jpg")
        imageList.append(image)

    return imageList


# Test görseli ile diğer görseller arasındaki en benzeri bulur.
def findMinDiffImage(testImage: Image, trainImageList: list):
    diffColorList = []
    diffLBPList = []
    diffColorLBPList = []
    for i in range(0, len(trainImageList)):
        diffRGB, diffLBP = diffBetweenTwoImages(testImage, trainImageList[i])
        diffColorLBP = Difference(diffRGB.imageName, diffRGB.diff + diffLBP.diff)
        diffColorList.append(diffRGB)
        diffLBPList.append(diffLBP)
        diffColorLBPList.append(diffColorLBP)

    differenceList = DifferenceList(diffColorList, diffLBPList, diffColorLBPList)
    return differenceList


# Objeyi belirtilen dosyaya kaydeder.
def saveObjectToFile(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


# Objeyi belirtilen dosyadan okur.
def readObjectFromFile(filename):
    try:
        with open(filename, 'rb') as inputs:
            imageList = pickle.load(inputs)
            print("Dosyadan okuma başarılı.")
    except FileNotFoundError:
        imageList = []
    return imageList


def printResults(testImage, differenceList):
    testImageType = testImage.imageName[6]
    if testImage.imageName[7] == '0':  # 1-10 arası aynı tip görsel
        testImageType = str(int(testImageType) - 1)
    print("Main Image : " + testImage.imageName)
    getResult("RGB", differenceList.colorList, testImageType)
    getResult("LBP", differenceList.lbpList, testImageType)
    getResult("RGB AND LBP", differenceList.colorLBPList, testImageType)


def getResult(compareName: str, compareList: list, testImageType):
    success = 0
    print(compareName + " COMPARE")
    for i in range(0, len(compareList)):
        print(str(i + 1) + ".Image : " + compareList[i].imageName)
        compareImageType = compareList[i].imageName[7]
        if compareList[i].imageName[8] == '0':  # 1-10 arası aynı tip görsel
            compareImageType = str(int(compareImageType) - 1)
        if testImageType == compareImageType:
            success += 1
    success = (100 * success) // len(compareList)
    print(compareName + " SUCCESS : " + str(success))
    if success > 0:
        print(compareName + " İLE BENZERLİK BAŞARILI.")
    else:
        print(compareName + " İLE BENZERLİK BAŞARISIZ.")


def makeTests(trainImages):
    for i in range(1, 71):
        if i < 10:
            imageNum = "0" + str(i)
        else:
            imageNum = str(i)
        testImage = readImage("test/test (" + imageNum + ").jpg")
        differenceList = findMinDiffImage(testImage, trainImages)
        printResults(testImage, differenceList)


def main():
    trainImages: list = readObjectFromFile('traindataset.pkl')
    if len(trainImages) == 0:
        print("Dosya bulunamadı.")
        trainImages = readTrainImages()
        saveObjectToFile(trainImages, 'traindataset.pkl')

    for i in range(0, trainImages.__len__()):
        print(trainImages[i].imageName)

    makeTests(trainImages)
    exit(0)


if __name__ == '__main__':
    main()
