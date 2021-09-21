from PIL import Image
import cv2
import numpy as np
def loadImage(imagePath, scaleFactor = 1):
    img = Image.open(imagePath)
    imgGray = img.convert('L')
    newSize = np.array(img.size)*scaleFactor
    newSize = [round(newSize[0]), round(newSize[1])]
    imgGray = imgGray.resize(newSize)
    imgMatrix = np.array(imgGray)
    return imgMatrix

def createImage(imageMatrix, curvePixels, curveColor = (0, 255, 0), widenCurve = False):
    rgbImage = cv2.cvtColor(imageMatrix, cv2.COLOR_GRAY2RGB)    
    
    if widenCurve:
        for y, x in curvePixels:
            for a,b in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,0),(0,1),(1,-1),(1,0),(1,1)]:
                try:
                    rgbImage[y+a][x+b] = curveColor
                except:
                    pass
    else:
        for y, x in curvePixels:
            rgbImage[y][x] = curveColor

    return rgbImage
    
def saveImage(imageMatrix):
    im = Image.fromarray(imageMatrix)
    im.save('test.png')

def displayImage(imageMatrix):
    im = Image.fromarray(imageMatrix)
    im.show()

if __name__ == "__main__":
    img = loadImage("004_demo_curv.tiff")
    pixels = [[x, i] for i in range(200) for x in range(20, 40)]
    newImg = createImage(img, pixels)
    displayImage(newImg)