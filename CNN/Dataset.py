
import gzip, cPickle
from sklearn.preprocessing import LabelBinarizer
import numpy 
import pylab
from PIL import Image
import glob
import pylab

def ImageProcess(jpgfile):
    # open random image of dimensions 162x162
    img = Image.open(jpgfile)
    img=img.resize((64,64),Image.BILINEAR)
    img=img.convert("L")
    img = numpy.asarray(img, dtype='float64') / 256.
    img = numpy.reshape(img,(-1,64*64))
    #pylab.subplot(1, 1, 1); pylab.axis('off'); pylab.imshow(img)
    #pylab.gray();
    #pylab.show()
    
    return img[0]
####
imageList=[]
labels=[]
pathname="1/*.jpg"
pathname2="2/*.jpg"
pathname3="3/*.jpg"
pathname4="4/*.jpg"
for jpgfile in glob.glob(pathname):
    imageList.append(ImageProcess(jpgfile))
    labels.append(0)
for jpgfile in glob.glob(pathname2):
    imageList.append(ImageProcess(jpgfile))
    labels.append(1)
for jpgfile in glob.glob(pathname3):
    imageList.append(ImageProcess(jpgfile))
    labels.append(2)
for jpgfile in glob.glob(pathname4):
    imageList.append(ImageProcess(jpgfile))
    labels.append(3)
    
x=numpy.asarray(imageList)
y=numpy.asarray(labels)
y=LabelBinarizer().fit_transform(y)
print x.shape



dataset=[x,y]

f = gzip.open('data_64.pkl.gz','wb')
cPickle.dump(dataset, f, protocol=2)
f.close()
