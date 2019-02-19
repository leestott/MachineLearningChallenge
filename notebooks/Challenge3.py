#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'notebooks'))
	print(os.getcwd())
except:
	pass
#%% [markdown]
# First download the gear images using; 
# 
# * curl -O https://challenge.blob.core.windows.net/challengefiles/gear_images.zip
# * curl -O https://storagetimscarfe.blob.core.windows.net/openhack/gear_images_test.zip
# * unzip *.zip
# 
# Or in windows powershell;
# 
# * wget "https://challenge.blob.core.windows.net/challengefiles/gear_images.zip" -OutFile gear_images.zip
# * wget "https://storagetimscarfe.blob.core.windows.net/openhack/gear_images_test.zip" -OutFile gear_images_test.zip
# * unzip gear_images.zip
#%% [markdown]
# <img src="results.png" style="width:75%" />

#%%
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.pyplot import imread, imshow, show, subplot, title, get_cmap, hist
import numpy as np
from PIL import Image, ImageOps, ImageChops
import cv2
import os
import pickle
from pipe import Pipe
from functools import reduce
import operator
from pipe import select,take,as_list
from sklearn import tree
from glob import glob
from PIL import Image
import numpy as np
import re

#Inline Matplot graphics into the notebook
get_ipython().magic(u'matplotlib inline')

def resizeImageToSquare(image, desiredSize):
    # old_size[0] is in (width, height) format
    oldSize = image.size

    ratio = float(desiredSize)/max(oldSize)
    newSize = tuple([int(x*ratio) for x in oldSize])

    image = image.resize(newSize, Image.BILINEAR)

    # create a new image and paste the resized on it
    resized = Image.new("L", (desiredSize, desiredSize), color=(255))
    resized.paste(image, ((desiredSize-newSize[0])//2,
                        (desiredSize-newSize[1])//2))
    return resized

#%% [markdown]
# * Load all the gear_images 
# * Resize them to 128^2
# * Make black and white (we don't want to learn particular colours of tents etc)
# * Scale image numbers $\in [0,1]$
# 
# Note that we have two separate populations of images:
# 
# * The original open hack challenge set
# * the set prepared by Tim with about 30 images per class. This set is slightly more "real world" i.e. avoiding the canonical white background shots where possible. Caveats being that this was hard to achieve in some cases, especially with crampons. Also as this is a classification problem (not multilabel) we tried to find images where there is only one class or one dominant class in the image. 

#%%
get_ipython().run_cell_magic(u'time', u'', u'\n@Pipe\ndef as_npy(iterable):\n    return np.array(iterable)\n\ndef getImageTypeFromPath(imagePath):\n    return re.search(\'[^\\w](\\w+)[^\\w]\', imagePath)[1]\n\n# note I have obfuscated the names so that people can not google the solutions\nimageTypes = {\n    \'axes\': 0, \n    \'boots\': 1, \n    \'carabiners\': 2,\n    \'crampons\': 3, \n    \'gloves\': 4, \n    \'hardshell_jackets\': 5, \n    \'harnesses\': 6, \n    \'helmets\': 7, \n    \'insulated_jackets\': 8, \n    \'pulleys\': 9, \n    \'rope\': 10, \n    \'tents\': 11, \n}\n\nimageTypesInverted = {v: k for k, v in imageTypes.items()}\n\ndef replaceBlack(t):\n    t2 = t[1]\n    t2[t2==0]=255\n    return (t[0], t2)\n\ndef GetDataByFolder(path="gear_images/**/*"):\n    \n    xy = ( glob(path)  \n        | select(lambda path: (imageTypes[getImageTypeFromPath(path)], Image.open(path)))     \n        # make black and white\n        | select(lambda t: (t[0], t[1].convert(\'L\') ) )     \n        # make square\n        | select(lambda t: (t[0], resizeImageToSquare(t[1], 128)) )     \n        | select(lambda t: (t[0], np.array(t[1]))) \n        # replace blacks with white, about 1/50 images load with a black background\n        # I have absolutley no idea why...\n        | select(replaceBlack)   \n        # scale to [0,1]\n        | select(lambda t: (t[0], t[1] / 255))     \n        | as_list() )\n\n    X = xy | select(lambda e: e[1]) | as_list() | as_npy()\n\n    # flattened version for classical learning\n    Xf = X.reshape( X.shape[0], reduce(operator.mul, X.shape[1:], 1)  )\n    y = xy | select(lambda e: e[0]) | as_list() | as_npy()\n\n    return X,y,Xf\n\nX,y,Xf = GetDataByFolder()\n\nX_test,y_test,Xf_test = GetDataByFolder("gear_images_test/**/*")\n\nprint("STANDARD: X", X.shape, "y", y.shape)\nprint("TEST: X", X_test.shape, "y", y_test.shape)')

#%% [markdown]
# Let's plot some of the images so we can see what we are working with

#%%
def plotSome(x, y, name='axes'):
    fig = plt.figure(figsize=(8, 6))
    # plot several images
    for i in range(15):
        ax = fig.add_subplot(3, 5, i + 1, xticks=[], yticks=[])
        ax.imshow(x[y==imageTypes[name]][i], cmap=plt.cm.bone)

from random import randint
        
def plot16(x, rand=False, title=""):
    fig = plt.figure(figsize=(8, 6))
    plt.title(title)
    plt.xticks([]),plt.yticks([])
    plt.axis('off')
    # plot several images
    for i in range(15):
        
        img = i
        if rand==True:
            img = randint(0,x.shape[0])
        
        ax = fig.add_subplot(3, 5, i + 1, xticks=[], yticks=[])
        
        
        ax.imshow(x[img], cmap=plt.cm.bone)
        ax.set_xlabel(img)

#%% [markdown]
# Let's visually inspect our dataset by running this a few times, this is great for sanity checking and looking for bugs in how we parsed the images. I already noticed that some of the images were loading in with a black background, I have no idea why! I put a hacky line of code in the loader to fix this i.e. setting all 0=>255

#%%
plot16(X, rand=True, title="random images from original population")
plot16(X_test, rand=True, title="random images from test/external/real world population")


#%%
plotSome(X,y,'carabiners')

#%% [markdown]
# Let's try plotting one image for every class

#%%
def PlotOneFromEachClass(x,y, title=""):
    
    fig = plt.figure(figsize=(8, 6))
    plt.title(title)
    plt.xticks([]),plt.yticks([])
    plt.axis('off')
    
    # plot several images
    i = 0
    for key in imageTypes.keys():
        ax = fig.add_subplot(3, 4, i + 1, xticks=[], yticks=[])
        
        ax.imshow(x[y==imageTypes[key]][0], cmap=plt.cm.bone)
        ax.set_xlabel(key)
        i = i + 1
    
    plt.show()
    
PlotOneFromEachClass(X,y, title="Training Image Population")
PlotOneFromEachClass(X_test,y_test, title="External Test Image Population")

#%% [markdown]
# Let's get a bit more clever now and plot an average image for each class. The result is pretty instructive;
# 
# * We can see that roughly half of the boots are inverted
# * In most cases there is a clear "signature" to the class, this is part of why this challenge is quite contrived, similar to the MNIST challenge, it should be possible to guess what the class is merely by choosing values for pixels in certain areas. 
# 
# We also plot an average image for the external "real world" set of images, as you can see, these lose their regularity possibly apart from insulated jackets, crampons and harnesses -- these were the classes which I found it quite difficult to find likey real-world images which were not dominated by other classes (and we are modelling this as a classification problem)

#%%
def plotAverage(x,y, title):
    fig = plt.figure(figsize=(8, 6))
    plt.title(title)
    # plot several images
    i = 0
    for key in imageTypes.keys():
        ax = fig.add_subplot(3, 4, i + 1, xticks=[], yticks=[])
        ax.imshow(x[y==imageTypes[key]].mean(axis=0), cmap=plt.cm.bone)
        ax.set_xlabel(key)
        i = i + 1
        
plotAverage(X,y, "average for the standard image population")
plotAverage(X_test,y_test, "average for the external/test image population")

#%% [markdown]
# Let's split the data into a test and train set

#%%
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X,
        y, random_state=0)

print(X_train.shape, X_val.shape)

#%% [markdown]
# Helper function to convert label indexes i.e. {0,1,2...} into labels i.e. {"axes","carabiners"...}

#%%
def ConvertIndexToLabel(indexes):
    return list( indexes ) | select( lambda i: imageTypesInverted[i] ) | as_list 

#%% [markdown]
# Here is a function which let's us plot a nice confusion matrix

#%%
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        

    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def plotConfusion(yy, p, title='Confusion matrix'):
    cnf_matrix = metrics.confusion_matrix(yy, p)
    np.set_printoptions(precision=2)

    cm_labels = list(set(imageTypes.keys()))
    cm_labels.sort()

    # Plot non-normalized confusion matrix
    plt.figure(figsize=(6, 6))
    plot_confusion_matrix(cnf_matrix, classes=cm_labels,
                          title=title)

#%% [markdown]
# What is the average image over everything?

#%%
def PlotAverageImage(D):
    x = X.mean(axis=0).flatten()
    plt.imshow(D.mean(axis=0), cmap=plt.cm.bone)
    u = plt.xlabel( f"Average of all images" ) 

PlotAverageImage(X)

#%% [markdown]
# Jackets seem to be dominating, could this be because;
# 
# * The intensity values are higher in jackets
# * There are more jackets overall (compounded by insulated jackets + normal jackets)
# 
# Note that the test set is balanced, or as near as makes no difference balanced 
# 

#%%
u = plt.hist(y, bins=12)
u = plt.xticks(range(12),imageTypes.keys(), rotation='vertical')

#%% [markdown]
# It does indeed appear that our hardshell jackets are dominating due to there being many more of them, also the jackets are around 200 and will have a similar profile which will compound the problem
#%% [markdown]
# Let's balance the distribution of classes by supersampling in the *train* split. It's important we don't do this on the test split too, as there would be many duplicate images in the test set and it would be trivial to memorise what they are. 

#%%
from pipe import chain, as_dict, Pipe, select, take, as_list

@Pipe
def pcycle(iterable):
    while True:
        for it in iterable:
            yield it
        
@Pipe
def as_npy(iterable):
    return np.array(iterable)

def balancedSignalGenerator(X,y,num_classes=12):
    
    class_map = {}
    for c in range(num_classes):
        class_map[c] = list(np.where( y==c)[0])
                            
    D = range(num_classes)         | select( lambda c: class_map[c] | pcycle | select( lambda i: (c,X[i]) )  )         | as_list

    while True:
        for c in D:
            yield next(c)[0], next(c)[1]

        
data = balancedSignalGenerator(X_train,y_train) | take(4200) | as_list

# note that we need as_list on the data
X_train_bal = data | select(lambda el: el[1])  | as_list | as_npy
y_train_bal = data | select(lambda el: el[0])  | as_list | as_npy


#%%
print( X_train_bal.shape, y_train_bal.shape )

#%% [markdown]
# And we are now balanced!

#%%
def labelDist(y, title):
    plt.figure()
    plt.hist(y,bins=12)
    plt.title(title)
    plt.xticks(range(12),imageTypes.keys(), rotation='vertical')
    
labelDist(y_train_bal, title="label distribution on train set")
labelDist(y_test, title="label distribution on external/test set")

#%% [markdown]
# Now, let's plot the average image again. This looks a little bit more reasonable i.e. there is a consistent "contribution" to the mixture from all 12 classes. If anything it's now the rope which is slightly dominating, I assume because its being "boosted" from mixing with the jackets which entirely intersect. 

#%%
PlotAverageImage(X_train_bal)

#%% [markdown]
# Let's look at the intensity histogram for each class average on the train set. This is interesting;
# 

#%%
fig = plt.figure(figsize=(8, 6))
# plot several images
i = 0
for key in imageTypes.keys():
    ax = fig.add_subplot(3, 4, i + 1, xticks=[], yticks=[])
    x = X[y==imageTypes[key]].mean(axis=0).flatten()
    ax.hist(x)
    ax.set_xlabel( f"{key} {x.mean():.2f}/{x.std():.2f}" ) 
    i = i + 1

#%% [markdown]
# One of the key goals in vision, is transforming the data out of the space domain. In the olden days of computer vision, the onus was on the data scientists themselves to transform the data out of the space domain using some feature. This histogram technique is the most basic version of that. 
# 
# 
# * Classical algorithms can not learn any spatial relationship or local pattern between the pixels and the assumption is that images are typically taken from a variety of angles/environments which *should* render pixel-by-pixel mapping methods unworkable. 
# * Images are sparse and statistically undescriptive
# * We experience the "curse of dimensionality" problem working with images on classical methods
# 
# 
# Let's look at these histograms; while there are some differences between the classes, clearly the previous (2d spatial) information is more instructive to seperate the class. We can tell this just with visual inspection. Let's close the loop on this one though and build a classifier using these histograms as feature vectors. 

#%%
def getHistFeatures(D, bins=20):
    
    n = D.shape[0]
    
    Xf = D.reshape(n,128*128)
    
    x = np.zeros( (n, bins))
    
    for i in range( n ):
        hist, _ = np.histogram(Xf[i], bins=bins)
        
        x[i] = hist
        
    x = x / np.max(np.max(x))
    
    return x

bins = 20
Xh_train_bal = getHistFeatures(X_train_bal, bins=bins)
Xh_val = getHistFeatures(X_val, bins=bins)
Xh_test = getHistFeatures(X_test, bins=bins)

print(Xh_train_bal.shape, Xh_val.shape, Xh_test.shape)

#%% [markdown]
# Basic Trees algorithm (ID3) function

#%%
from sklearn import tree
from sklearn import metrics

def flattenImage(I, dim=128):
    if len(I.shape)>3:
        return I.reshape(I.shape[0], dim*dim, 3)
    else:
        return I.reshape(I.shape[0], dim*dim)

# expects flattened data going in
def ID3(xtr,xte,ytr,yte, title="confusion"):
    print(title)
    classifier = tree.DecisionTreeClassifier()
    clf = classifier.fit(xtr, ytr)
    p = clf.predict(xte)
    print(
        metrics.classification_report(
            ConvertIndexToLabel(yte), 
            ConvertIndexToLabel(p)
        ))
    plotConfusion(yte, p, title=title)
    return p, clf

#%% [markdown]
# So we have now transformed our (1591, 128, 128) image data into (1591, 20) i.e. instead of $128^2$ pixel values, we have $20$ discrete bins of intensity values. We have also scaled the data $\in [0,1]$ Let's run ID3 (trees) and see what we get

#%%
get_ipython().run_cell_magic(u'time', u'', u"ID3( Xh_train_bal, Xh_val, y_train_bal, y_val, title='Trees intensity histogram features (validation)\\n' )\nID3( Xh_train_bal, Xh_test, y_train_bal, y_test, title='Trees intensity histogram features (test)\\n' )")

#%% [markdown]
# Surprisingly, this isn't too bad on the validation set! Clearly though we lose a lot of information by transforming out of the space domain in this way. On the test set i.e. the real world dataset, the results are shockingly poor
#%% [markdown]
# Let's run a basic decision tree algorithm on the images (balanced train set) on the **raw pixels** i.e. ID3 algorithm

#%%
get_ipython().run_cell_magic(u'time', u'', u"ID3( \n    flattenImage(X_train_bal), \n    flattenImage(X_val), \n    y_train_bal, \n    y_val, \n    title='Trees on balanced train set, unbalanced validation set' )\n\nID3( \n    flattenImage(X_train_bal), \n    flattenImage(X_test), \n    y_train_bal, \n    y_test, \n    title='Trees on balanced train set, test set' )")

#%% [markdown]
# We get about ~.61 f-1 on validation which to be clear is training on the supersampled train split and tested on the original validation and test splits. The confusion matrix; clearly the biggest problem is confusing the two jacket classes (validation). But surprisingly, this isn't too bad either. It is better to model the raw pixels right now than the histogram features. On the real world test split however, the results are shockingly bad. We are getting slightly better than guessing and it's predicting everything as ropes!
# 
# Take home message should be developing here; using classical algorithms on image data isn't a smart thing to do -- especially if you are modelling the raw pixels!
#%% [markdown]
# Other approaches work to try and transform out of the larger, sparse space domain while retaining some spatial information. An example of this is histogram of gradients (HoG) 

#%%
from skimage.feature import hog

fd, hogimage = hog(X_train_bal[30], block_norm='L2-Hys', visualise=True, cells_per_block=(10,10), pixels_per_cell=(10,10))

plt.imshow(hogimage)

print(fd.shape)


#%%
get_ipython().run_cell_magic(u'time', u'', u"\ndef computeHogFeatures(x):\n    n = x.shape[0]\n    \n    Xhog = np.zeros( (n, 8100) )\n\n    for i in range(n):\n        Xhog[i] = hog(x[i], block_norm='L2-Hys', cells_per_block=(10,10), pixels_per_cell=(10,10))\n        \n    return Xhog\n\nX_train_bal_hog = computeHogFeatures(X_train_bal)\nX_test_bal_hog = computeHogFeatures(X_test)")

#%% [markdown]
# SVM reusable function;

#%%
from sklearn import svm
    
def SVM(xtr, xte, ytr, yte, title="confusion matrix", C=10., gamma=0.001):
    print(title)
    clf = svm.SVC(C=C, kernel='rbf', gamma=gamma)
    clf.fit(xtr, ytr)
    y_pred = clf.predict(xte)
    print(metrics.classification_report(ConvertIndexToLabel(yte), ConvertIndexToLabel(y_pred)))
    plotConfusion(yte, y_pred, title=title)
    return y_pred, clf

#%% [markdown]
# Now let's run an SVM on the hog features (turns out this is actually really bad, PCA didn't help)
# 
# Note that "Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples." means that some examples didn't even get a prediction at all

#%%
get_ipython().run_cell_magic(u'time', u'', u"y_pred = SVM(X_train_bal_hog, X_test_bal_hog, y_train_bal, y_test, title='SVM HOG test\\n', C=25, gamma=0.001)")

#%% [markdown]
# PCA helper function

#%%
from sklearn import decomposition

def PCA(xtr, xte, n_components=50, flat_size = 128*128):
    
    pca = decomposition.PCA(n_components=n_components)
    pca.fit(xtr.reshape(xtr.shape[0],flat_size))

    xtr_pca = pca.transform(xtr.reshape(xtr.shape[0],flat_size))
    xte_pca = pca.transform(xte.reshape(xte.shape[0],flat_size))
    
    return xtr_pca, xte_pca, pca
    

#%% [markdown]
# Let's try PCA on the raw pixels + SVM

#%%
get_ipython().run_cell_magic(u'time', u'', u"\nX_train_bal_pca, X_val_bal_pca, _ = PCA(X_train_bal, X_val, n_components=40)\n\ny_pred, model = SVM(X_train_bal_pca, \n                    X_val_bal_pca, \n                    y_train_bal, \n                    y_val, \n                    title='SVM PCA (validation)', \n                    C=20, \n                    gamma=0.001)\n\nX_train_bal_pca, X_test_bal_pca, _ = PCA(X_train_bal, X_test, n_components=40)\n\ny_pred, model = SVM(X_train_bal_pca, \n                    X_test_bal_pca, \n                    y_train_bal, \n                    y_test, \n                    title='SVM PCA (test)', \n                    C=20, \n                    gamma=0.001)\n")

#%% [markdown]
# This is instructive. On validation it looks like we just build an amazing classifier (f1 0.9), but in the real-world test dataset it performs shockingly badly (0.18).
# 
# On the real-world datasets it seems to love classifying everything as rope! Let's take a look at that class and see what's going on?

#%%
plotSome(X_test, y_test, name="rope")

#%% [markdown]
# Now let's add some white noise to the backgrounds, the hypothesis on the table is that it's just learning where the white pixels are to perform so well on the validation set

#%%
get_ipython().run_cell_magic(u'time', u'', u'\nfrom random import uniform\nfrom pipe import take, select, as_list, Pipe\n\nimport random\n\n@Pipe\ndef pshuffle(l):\n    random.shuffle(l)\n    return l\n\ndef noiseImages( images ):\n    \n    images_whitenoise = images.copy()\n\n    mask = images_whitenoise>0.95\n    vals = range(np.count_nonzero(mask)) | select( lambda n: uniform(0, 1)) | as_list()\n\n    # Assign back into X\n    images_whitenoise[mask] = vals\n    return images_whitenoise\n\nX_val_whitenoise = noiseImages( X_val )\nX_train_bal_whitenoise = noiseImages( X_train_bal )')

#%% [markdown]
# And let's plot 16 test images to see if the backgrounds have been noised

#%%
plot16(X_val_whitenoise, rand=True)

#%% [markdown]
# Now let's run trees (ID3) on noisy images, we are expecting the accuracy to plummet

#%%
print(X_train_bal_whitenoise.shape, X_val_whitenoise.shape, y_train_bal.shape, y_test.shape)


#%%
get_ipython().run_cell_magic(u'time', u'', u"\npreds = ID3(flattenImage(X_train_bal_whitenoise), \n             flattenImage(X_val_whitenoise), \n             y_train_bal, y_val, title='Confusion on unbal val split with white noise added')")

#%% [markdown]
# And indeed, the accuracy plumets to ~.34 average f-1 score!! Ouch.
# 
# It's now shockingly bad on axes, helmets and pulleys, let's look at the confusion matrix. This means we can surmise to some extent that the previous model was overfitting the whitespace
#%% [markdown]
# Pretty bad across the board but what we do see is that hardshell jackets dominate in the confusion matrix! This is pretty much because we have more hardshell jackets than any other class in the validation set which means that the confusion matrix is not instructive. Note that the f-1 scores are normalised by the "rate" i.e. are agnostic to class imbalance.  We can;
# 
# * Supersample the validation set so that the label distribution is balanced like the training set, this would result in many duplicate images
# * Subsample the val set so that the label distribution is balanced, this would result in a much smaller set
# * Supersample the val set with augmentation i.e. random translations, scale/rotation transforms (maybe even affine). 
# 
# There are tradeoffs here. The first option will effectively be testing over many of the same images. The second option would result in a much smaller sample. The third option is safer but would destroy classical methods even faster which perform pixel mapping. We want to do augmentation anyway soon to demonstrate this. 
#%% [markdown]
# First, let's supersample the val set and renoise, and see what we get

#%%
get_ipython().run_cell_magic(u'time', u'', u'data = balancedSignalGenerator(X_val,y_val) | take(1200) | as_list\n\nX_val_bal = data | select(lambda el: el[1])  | as_list | as_npy\ny_val_bal = data | select(lambda el: el[0])  | as_list | as_npy\n\nX_val_bal_whitenoise = noiseImages( X_val_bal )\n\ny_pred = ID3(flattenImage(X_train_bal_whitenoise), \n                     flattenImage(X_val_bal_whitenoise), y_train_bal, y_val_bal, title="ID3 raw pixels noise, balanced val")')

#%% [markdown]
# This is interesting, so the F-1 score is similar but the results are way more balanced i.e. the confusion matrix is telling a much better story and we can actually intepret what is being confused with what and suffice to say, there is a lot of confusion going on ;)
#%% [markdown]
# There is a problem in classical ML approaches called the *Curse of Dimensionality* where classifiers tend to overfit the training data in high dimensional spaces, especially when there are more dimensions as there are examples. Deep learning doesn't suffer from this problem because the data is usually larger, and that deep learning algorithms learn a lower dimensional intermediate representation as part of the training process which might have different activations for sparse patterns in the input data and also learn a spatial manifold to transform the data. 
# 
# SVM transforms the data into an implied higher dimensional (infinite in the case of RBF kernel i.e. kernel hilbert space see https://en.wikipedia.org/wiki/Mercer%27s_theorem) space by computing the $N \times N$ $XX'$ matrix i.e. where $N$ is the number of examples. This means that the space complexity is a function of $N$, not $L$ (the number of attributes). Of course in this situation, $L$ is still $~2200$ which means we have about $5$ million bytes. You might think that SVMs would be more prone to the curse of dimensionality, but actually this is not true. 
# 
# "The SVM is an approximate implementation of a bound on the generalization error, that depends on the margin (essentially the distance from the decision boundary to the nearest pattern from each class), but is independent of the dimensionality of the feature space." https://stats.stackexchange.com/questions/35276/svm-overfitting-curse-of-dimensionality
# 
# In classical ML, the more features you add past a point, the worse the performance gets;
# 
# <img src="attachment:cursedimen.PNG" width="400">
# 
# (image from http://www.visiondummy.com/2014/05/feature-extraction-using-pca/#PCA_pitfalls)
# 
# Of course, in this application every single pixel is a feature, which means we have $128^2$ features which is $\gt10000$! It means we have significantly more features than we do signals. 
# 
# An approach people use in these circumstances with classical algorithms is to use PCA to reduce the dimensionality first. 
# 
# Signals are often redundant because they are statistically dependent (correlated). PCA transforms the data into a set of uncorrelated components, allowing you to focus on the most significant components without losing any information about them. 
# 
# PCA computes the eigen vectors of the covariance matrix of $X$ and then projects the data onto those vectors. This still makes some assumption about the absolute pixel positions of images i.e. if they are all shifted, there would be less correlation between the dimensions. PCA is good because it;
# 
# * Captures the correlation if there is any and projects into decorrelated features
# * Removes noise
# * Helps with the curse of dimensionality
# 
# <img src="attachment:pca_bad.PNG" width="400">
# 
# 
# 
# 
# In this case, other dimensionality reduction methods might be of interest, such as Linear Discriminant Analysis (LDA) which tries to find the projection vector that optimally separates the two classes.
# 
# PCA would clearly help here because all the images are scaled and frontalised the same. So in that sense it actually helps "transforming the data out of the space domain" by learning correlations between pixels fixed in space. If the images were "all over the place" and subject to random transformations (augmentations); PCA would only help in this respect where there was some accidental consistency of overlap.

#%%
X_train_bal_whitenoise_pca, X_val_bal_whitenoise_pca, pca = PCA(X_train_bal_whitenoise, X_val_bal_whitenoise)

#%% [markdown]
# Let's visualise the first principal components

#%%
fig = plt.figure(figsize=(16, 6))
for i in range(50):
    ax = fig.add_subplot(5, 10, i + 1, xticks=[], yticks=[])
    ax.imshow(pca.components_[i].reshape(128,128),
              cmap=plt.cm.bone)

#%% [markdown]
# What we observe here is that the principal component do not magically transform the data out of the space domain -- the x/y dimensions are still highly relevant. But it does show us something about the variance between pixels accross classes i.e. we can start to see the shapes of different items
#%% [markdown]
# Now that we have only 50 features, it's computationally tractable to run something like an SVM, let's try that
#%% [markdown]
# Let's try running an SVM on the PCA data
# 
# Let's use the PCA projection vectors learned from the whitenoise train split to transform the external test data and see if we get any traction on that.

#%%
get_ipython().run_cell_magic(u'time', u'', u'y_pred = SVM(X_train_bal_whitenoise_pca, \n             X_val_bal_whitenoise_pca, \n             y_train_bal, \n             y_val_bal, \n             title="white noise + PCA + balanced validation")\n\nX_test_pca = pca.transform(X_test.reshape(359, 128*128))\n\ny_pred = SVM(X_train_bal_whitenoise_pca, \n             X_test_pca, \n             y_train_bal, \n             y_test, \n             title="white noise + PCA + balanced test/external")\n')

#%% [markdown]
# The performance drops from .9 to .68, the whitespace was clearly helping the classifier before. Lots of things seem to be classified as axes incorrectly
# 
# When we project the external test data using the same PCA projection vectors, the results are shocking -- clearly this is an approach that participants might think is a good idea but is a disaster in practice 
#%% [markdown]
# The hypothesis on the table is that a CNN would work better because it can model local spatial patterns which are somewhat invariant to translation
# 
# First to use a CNN we need to add an explicit colour channel back in, even if it's only 1, also we need to one-hot encode the labels

#%%
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping

def addColourDimension(x):
    return x.reshape(x.shape[0],128,128,1)

def plotHistory(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()
    
    
def runCNNModel( xtr, xte, ytr, yte, epochs=5, batchsize=100, dropout=0.4 ):
    
    xtr_1 = addColourDimension(xtr)
    xte_1 = addColourDimension(xte)
    ytr_1 = to_categorical(ytr, num_classes=12)
    yte_1 = to_categorical(yte, num_classes=12)
    
    model = Sequential()
    model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(128,128,1)))
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Flatten())
    model.add(Dropout(dropout))
    model.add(Dense(12, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    callbacks = [ EarlyStopping(
        monitor='val_acc', 
        min_delta=0, 
        patience=10, 
        verbose=0, 
        mode='auto', 
        baseline=None, 
        # this will restore the best weights back to the model
        restore_best_weights=True) ]
    
    #train the model
    history = model.fit(xtr_1, ytr_1, 
                        epochs=epochs, batch_size=batchsize, 
                        validation_data=(xte_1, yte_1),
                        callbacks=callbacks, verbose=1)
    
    plotHistory(history)
    
    return history, model


#%%
history, model_noise = runCNNModel(
    X_train_bal_whitenoise,
    X_val_bal_whitenoise,
    y_train_bal,
    y_val_bal, epochs=30, batchsize=200, dropout=0.6)


#%%
from numpy import argmax

def CNNEvaluate(model, xte,yte,title=''):
    y_pred = model.predict(addColourDimension(xte))
    print(metrics.classification_report(ConvertIndexToLabel(yte), ConvertIndexToLabel(argmax(y_pred,axis=1))))
    plotConfusion(yte, argmax(y_pred,axis=1), title=title)

CNNEvaluate(model_noise,X_val_bal_whitenoise,y_val_bal, title='CNN whitenoise balanced train+nonoise val')

#%% [markdown]
# Wow! On the validation data -- the CNN is getting a pretty much perfect score -- even with the noise added! 
#%% [markdown]
# The CNN on the balanced noisy data works very well indeed - almost a perfect result, demonstrating the power of being able to learn invariant spatial features between pixels. The best classical approach was .69f1 and this improves it significantly to .94f1
#%% [markdown]
# And if we run it against the external test population?

#%%
CNNEvaluate(model_noise,X_test,y_test, title='CNN NOISE balanced train+nonoise external test')

#%% [markdown]
# Oh dear! This should be a clue of something, with the noise the CNN doesn't generalise AT ALL to test!
#%% [markdown]
# What if we run the CNN on the pre-noise data?

#%%
history, model_nonoise = runCNNModel(
    X_train_bal,
    X_val_bal,
    y_train_bal,
    y_val_bal, epochs=30, batchsize=200, dropout=0.6)


#%%
CNNEvaluate(model_nonoise,X_test,y_test, title='CNN NO NOISE balanced train+nonoise test')

#%% [markdown]
# What's interesting here is that the white noise **hardly affected CNN** at all on validation performance, but totally destroyed it on the test performance. This is a clue that the CNN was actually learning the noise which is the opposite of what we wanted it to do. 
#%% [markdown]
# Now we are going to simulate a real-world scenario and apply random mutations to the images before splitting and training
# 
# First we use some boilerplate code to handle the augmentation
# 
# We will select out 20000 images which is the upper limit of what you could expect to use on a classical algorithm but enough to allow the CNN to learn well in respect of the augmentation

#%%
get_ipython().run_cell_magic(u'time', u'', u'\nimport random\n\nfrom keras.preprocessing.image import ImageDataGenerator\n\ndata_gen_args = dict(rotation_range=0.5,\n                     width_shift_range=0.3,\n                     height_shift_range=0.3,\n                     zoom_range=0.3,\n                     horizontal_flip=True,\n                     fill_mode="constant",\n                     cval=1)\n\nimage_datagen = ImageDataGenerator(**data_gen_args)\n\n@Pipe\ndef augmentation(flow):\n\n    for x in flow:\n        trans = image_datagen.get_random_transform( (128,128) )\n        yield ( x[0], image_datagen\n                       .apply_transform(\n                           x[1].reshape(128,128,1), trans\n                        ).reshape(128,128) )\n\naugmented_flow = balancedSignalGenerator(X,y) | augmentation | take(20000) | as_list\n\nX_bal_aug = augmented_flow | select(lambda el: el[1]) | as_list | as_npy\ny_bal_aug = augmented_flow | select(lambda el: el[0]) | as_list | as_npy\n\nX_bal_aug_noise = noiseImages(X_bal_aug)')

#%% [markdown]
# Let's see what the results are

#%%
plotSome(X_bal_aug_noise, y_bal_aug, name="pulleys")


#%%
plot16(X_bal_aug, rand=True)

#%% [markdown]
# OK, we have a pretty cool dataset here now - which resembles real life conditions. The hypothesis on the table is that the trees algorithm gets near to zero accuracy, Let's see.
#%% [markdown]
# First let's split the data. Note that now we are augmenting, it's no longer cheating to split *after* the augmenting because we can think of them as being unique images i.e. the same thing from a different angle

#%%
get_ipython().run_cell_magic(u'time', u'', u'\nfrom sklearn.model_selection import train_test_split\n\nX_train_bal_aug_noise, X_test_bal_aug_noise, y_train_bal_aug_noise, y_test_bal_aug_noise \\\n    = train_test_split(X_bal_aug_noise, y_bal_aug, random_state=0)\n\nX_train_bal_aug, X_test_bal_aug, y_train_bal_aug, y_test_bal_aug \\\n    = train_test_split(X_bal_aug, y_bal_aug, random_state=0)\n\nprint(X_train_bal_aug_noise.shape, X_test_bal_aug_noise.shape)\nprint(X_train_bal_aug.shape, X_test_bal_aug.shape)')


#%%
get_ipython().run_cell_magic(u'time', u'', u"y_pred = ID3(\n            flattenImage(X_train_bal_aug_noise), \n            flattenImage(X_test_bal_aug_noise), \n            y_train_bal_aug_noise, \n            y_test_bal_aug_noise\n            )\n\nplotConfusion(y_test_bal_aug_noise, y_pred, title='ID3 Augmented+Noise')")

#%% [markdown]
# The next assertion is that PCA will no longer help us as the images are not aligned in space let's see

#%%
get_ipython().run_cell_magic(u'time', u'', u'\nfrom sklearn import decomposition\n\nX_train_bal_aug_hog = computeHogFeatures(X_train_bal_aug)\nX_test_bal_aug_hog = computeHogFeatures(X_test_bal_aug)\n\nX_train_bal_aug_hog_pca, X_test_bal_aug_hog_pca, pca_bal_aug_hog = \\\n    PCA(X_train_bal_aug_hog, X_test_bal_aug_hog, flat_size=8100)')

#%% [markdown]
# Let's run SVM on the PCA transform from augmented+PCA (note we tried noise separatley and it was worse)

#%%
X_train_bal_aug_pca, X_test_bal_aug_pca, pca_bal_aug  =     PCA(X_train_bal_aug, X_test_bal_aug)

preds, model = SVM(
    X_train_bal_aug_pca,
    X_test_bal_aug_pca,
    y_train_bal_aug,
    y_test_bal_aug, 
    title='SVM Augmented+PCA'  )

#%% [markdown]
# And svm/pca on the test set

#%%
y_preds = model.predict( pca_bal_aug.transform( X_test.reshape(359, 128*128) ) )

print(metrics.classification_report(ConvertIndexToLabel(y_test), ConvertIndexToLabel(y_preds)))
plotConfusion(y_test, y_pred[0], title="PCA Noise Aug 20K")

#%% [markdown]
# Let's run SVM on the PCA transform from augmented+noise+hog

#%%
get_ipython().run_cell_magic(u'time', u'', u"\npreds, model = SVM(\n    X_train_bal_aug_hog_pca,\n    X_test_bal_aug_hog_pca,\n    y_train_bal_aug,\n    y_test_bal_aug, \n    title='SVM Augmented+HOG+PCA'  )")

#%% [markdown]
# And on the test set (with hog)

#%%
y_preds = model.predict( pca_bal_aug_hog.transform( computeHogFeatures( X_test ) ) )

print(metrics.classification_report(ConvertIndexToLabel(y_test), ConvertIndexToLabel(y_preds)))
plotConfusion(y_test, y_pred[0], title="PCA HOG Aug 20K")

#%% [markdown]
# PCA is not helping us like it did before, originally we went from about 0.28 on ID3 (trees) with balanced training labels and noise added to about .67 with PCA+SVM. Here we jump just .1 on average f-1 and as you can see from the confusion matrix it seems to think everything is axes and isn't doing well at all! It is doing well on gloves and boots and this might just be because they are not augmenting well?
#%% [markdown]
# Now let's try the CNN again on the augmented data, we are expecting it to be significantly better

#%%
history, model_cnn_aug_noise = runCNNModel(
    X_train_bal_aug_noise,
    X_val_bal_whitenoise,
    y_train_bal_aug_noise,
    y_val_bal, epochs=40, batchsize=200, dropout=0.3)

plotHistory(history)


#%%
CNNEvaluate(model_cnn_aug_noise, 
            X_val_bal_whitenoise,
            y_val_bal, 
            title='CNN whitenoise balanced augmented train+no noise val')

CNNEvaluate(model_cnn_aug_noise, 
            X_test,
            y_test, 
            title='CNN whitenoise balanced augmented train+no noise test')

#%% [markdown]
# The augmented CNN achieves f1 of .80 on the augmented+noise data, a significant improvement on the .44 with augmented+noise+pca+SVM. Frankly I am extremely surprised the latter did as well as .44, I suspect if we augment more aggressively it will shoot down as we desmonstrated on the external test set. The augmented CNN with noise only achieves .12 on the external test set..
#%% [markdown]
# The key point here is that the noise actually make the CNN generalise to test WORSE not better. You would think that the CNN would ignore the noise, but it must have somehow lost some of its predictive power because of it
# 
# Let's do the augmented CNN with NO NOISE

#%%
history, model_cnn_aug = runCNNModel(
    X_train_bal_aug,
    X_val_bal,
    y_train_bal_aug,
    y_val_bal, epochs=40, batchsize=200, dropout=0.3)

plotHistory(history)


#%%
CNNEvaluate(model_cnn_aug, 
            X_val_bal,
            y_val_bal, 
            title='CNN whitenoise balanced augmented train+no noise val')

CNNEvaluate(model_cnn_aug, 
            X_test,
            y_test, 
            title='CNN whitenoise balanced augmented train+no noise test')

#%% [markdown]
# We do get slightly better transferability to the test set (0.29)
#%% [markdown]
# Just to close the loop on this one, let's try transfer learning and see if it's any better

#%%
from keras.applications import Xception

def runCNNTransferModel( xtr, xte, ytr, yte, epochs=5, batchsize=100, dropout=0.4 ):
    
    conv_base = Xception(weights='imagenet',
                  include_top=False,
                  input_shape=(128, 128, 3))
    
    conv_base.trainable = False
    
    xtr_1 = addColourDimension(xtr)
    xte_1 = addColourDimension(xte)
    ytr_1 = to_categorical(ytr, num_classes=12)
    yte_1 = to_categorical(yte, num_classes=12)
    
    model = Sequential()
    # clever trick to "learn" a colour mapping from black and white into the the colour channels of
    # the pretrained Xception model
    model.add(Conv2D(10, kernel_size = (1,1), input_shape=(128, 128, 1), padding = 'same', activation = 'relu'))
    model.add(Conv2D(3, kernel_size = (1,1), padding = 'same', activation = 'relu'))
    model.add(conv_base)
    model.add(Flatten())
    model.add(Dropout(dropout))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(12, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    callbacks = [ EarlyStopping(
        monitor='val_acc', 
        min_delta=0, 
        patience=10, 
        verbose=0, 
        mode='auto', 
        baseline=None, 
        # this will restore the best weights back to the model
        restore_best_weights=True) ]
    
    #train the model
    history = model.fit(xtr_1, ytr_1, 
                        epochs=epochs, batch_size=batchsize, 
                        validation_data=(xte_1, yte_1),
                        callbacks=callbacks, verbose=1)
    
    plotHistory(history)
    
    return history, model


#%% [markdown]
# First of all we can try training on the external test set itself to see what the best possible performance could be

#%%
history, model2 = runCNNTransferModel(
    X_test,
    X_test,
    y_test,
    y_test, epochs=40, batchsize=200, dropout=0.65)

plotHistory(history)

CNNEvaluate(model2, 
            X_test,
            y_test, 
            title='CNN Xception baseline training on test')

#%% [markdown]
# As you can see, with real-world images -- this is extremely difficult! We are getting 0.39 f-1 baseline so it should in theory be almost impossible to improve on this baseline

#%%
history, model2 = runCNNTransferModel(
    X_train_bal_aug,
    X_val_bal,
    y_train_bal_aug,
    y_val_bal, epochs=40, batchsize=200, dropout=0.65)

plotHistory(history)

CNNEvaluate(model2, 
            X_test,
            y_test, 
            title='CNN Xception balanced augmented train+ external test')

#%% [markdown]
# This is actually quite impressive, we are getting .5 on the external test set! Not brilliant but it's *something*
# 
# On some classes it seems to do quite well, especially tents which is a hard one on the test set.
# 
# One thing which might be going on here is that there are some classes which the convolutional base has higher level representations for, learned from imagenet (tents perhaps?)
# 
# Still -- this is an impressive result - better than the baseline and a LOT better than all the other algorithms on the external test set. 
#%% [markdown]
# Let's try the same for the noise dataset
#%% [markdown]
# The next frontier! LIME! We can use Lime to explain to us how our classifiers are working. Currently I am struggling to get it to work well so we will have to leave that for another day. 

#%%
get_ipython().run_cell_magic(u'time', u'', u"\nfrom lime import lime_image\nimport time\nfrom keras.utils import to_categorical\nfrom skimage.segmentation import mark_boundaries\n\nexplainer = lime_image.LimeImageExplainer()\n\ndef predict_fn_id3(images):\n    xf = flattenImage(images)\n    xf = np.mean(xf,axis=2)\n    xf = xf.reshape(xf.shape[0],xf.shape[1])\n    return clf.predict_proba(xf)\n\ndef predict_fn_cnn(images):\n    images = np.mean(images,axis=3)\n    images = images.reshape(images.shape[0],128,128,1)\n    return model.predict(images)\n\nexplanation = explainer.explain_instance(X_test[480], \n                                         predict_fn_cnn, \n                                         top_labels=12,\n                                         hide_color=False, \n                                         num_samples=5000,\n                                         num_features=100000,\n                                         random_seed=42)\n\ntemp, mask = explanation.get_image_and_mask(imageTypes['gloves'], \n                                            positive_only=False, \n                                            num_features=100000, \n                                            hide_rest=True)\n\nplt.imshow(mark_boundaries(temp / 2 + 0.5, mask))")

#%% [markdown]
# Lime gives me different results every time, does not show prediction affinities against whitespace, and I assume the contigous pixel segmentation algorithm doesnt work on the whitespace. Let's move to good old SHAP ;)
# 
# See below for an illustration of how this visualisation looks like for mnist
#%% [markdown]
# <img src="mnist_shap.PNG" style="width: 80%" />

#%%
get_ipython().run_cell_magic(u'time', u'', u'import numpy as np\nimport shap\n\ndef ShapInspection( Xtr, Xte, model, background_count=10, test_count=20 ):\n\n    # select a set of background examples to take an expectation over\n    background = Xtr[np.random.choice(Xtr.shape[0], background_count, replace=False)]\n    background = addColourDimension(background)\n\n    # explain predictions of the model on four images\n    e = shap.DeepExplainer(model, background)\n\n    random_val = np.random.choice(Xte.shape[0], test_count, replace=False)\n\n    # e = shap.DeepExplainer((model.layers[0].input, model.layers[-1].output), background)\n    shap_values = e.shap_values(addColourDimension(Xte[random_val]))\n\n    # plot the feature attributions\n    shap.image_plot(shap_values, \n                    -addColourDimension(Xte[random_val]), \n                   # looking at source, seems this is labels for the examples, not classes, duh\n                   # labels=np.array(list(imageTypes.keys())),\n                    show=True)')


#%%
print( enumerate( imageTypes.keys() ) | as_list )

#%% [markdown]
# Let's evaluate SHAP on the transfer learning model

#%%
get_ipython().run_cell_magic(u'time', u'', u'ShapInspection(X_train, X_test, model2) ')

#%% [markdown]
# Here we can really see some local activation, 
# 
# * it seems to be learning the subject of focus
# * on the jackets, it seems to be looking for the shame in the middle of the picture even if the test image doesn't fill it out!
# * On the pulley example, it's clearly learning the characteristic holes in the pulley
#%% [markdown]
# And on the vanilla CNN model

#%%

get_ipython().run_cell_magic(u'time', u'', u'ShapInspection(X_train, X_test, model_cnn_aug) ')

#%% [markdown]
# * On one of these you can even see it trying to activate the shape of a helmet even though it's a pulley!

