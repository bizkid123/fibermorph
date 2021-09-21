import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.measure import CircleModel
import time
from curveDrawing import createImage, loadImage, displayImage

class CurvesImage:
    def __init__(self, imagePath, imageWidth, imageHeight):
        self.imageWidth, self.imageHeight = imageWidth, imageHeight
        self.imagePath = imagePath
        img = Image.open(imagePath)
        self.xPixels, self.yPixels = img.size
        img.thumbnail((600, 600))

        self.scaleFactor = img.size[0]/self.xPixels
        imgGray = img.convert('L')

        imageMatrix = np.array(imgGray)
        binarizedImage = ((256 - imageMatrix) // 70)
        self.analysis(binarizedImage)
        
    def analysis(self, imageMatrix):
        labels, numFeatures = ndimage.label(imageMatrix, structure = np.ones((3,3)))

        # Method from https://stackoverflow.com/questions/32748950/fast-way-to-turn-a-labeled-image-into-a-dictionary-of-label-coordinates
        # @ Warren Weckesser, Saves ~ 12 ms over naiive solution with np.where
        nz = np.nonzero(labels)
        coords = np.column_stack(nz)
        nzvals = labels[nz[0], nz[1]]
        self.curves = [Curve(coords[nzvals == i], i) for i in range(1, numFeatures + 1)]

class Curve:
    def __init__(self, points, labelNum):
        self.points = points
        self.labelNum = labelNum
        self.generateCircle()
        self.getAngles()
        self.getAccuracy()
        
    def generateCircle(self):
        self.model = CircleModel()
        self.model.estimate(self.points)

    def getAngles(self):
        normalizedPoints = self.points - self.model.params[:2]
        angles = np.arctan2(normalizedPoints[:,1], normalizedPoints[:,0])
        self.angles = np.where(angles > 0, angles, 2*np.pi + angles)
    
    def getAccuracy(self):
        predictedPoints = self.model.predict_xy(self.angles)
        distances = np.linalg.norm(predictedPoints - self.points)
        return np.average(distances**2) / (len(self.points))**1/(1.5) # Arbitrary
    
        
if __name__ == "__main__":
    s = time.perf_counter()
    x = None
    for i in range(100):
        x = CurvesImage("027_demo_nocurv.tiff", 10, 10)
    e = time.perf_counter()
    print(e - s)
    
    # # Color every point detected
    # i = createImage(loadImage(x.imagePath,x.scaleFactor), np.concatenate([curve.points for curve in x.curves]).astype(int))
    # displayImage(i)
    
    # # Color every calculated points
    # i = createImage(loadImage(x.imagePath,x.scaleFactor), np.concatenate([curve.model.predict_xy(curve.angles) for curve in x.curves]).astype(int), widenCurve = True)
    # displayImage(i)
    
    
    # Color only calculated points on "good fits"
    i = createImage(loadImage(x.imagePath,x.scaleFactor), np.concatenate([curve.model.predict_xy(curve.angles) for curve in x.curves if len(curve.points) > 50 and curve.getAccuracy() < 50]).astype(int), widenCurve = True)
    displayImage(i)