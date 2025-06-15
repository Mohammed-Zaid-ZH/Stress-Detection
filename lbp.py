from skimage.feature import local_binary_pattern as lbp
import torch 
from torchvision import models,transforms
from torchvision.models import VGG16_Weights
from PIL import Image
import matplotlib.pyplot as plt
import dlib
import numpy as np 
import cv2 

global vgg16,detector,predictor
vgg16=models.vgg16(weights=VGG16_Weights.DEFAULT)
vgg16.eval()
transform=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
def process(img,r=1,n=8):
    i=cv2.imread(img,cv2.IMREAD_GRAYSCALE)
    if i is None or not i.size:
        print('Failure to detect',img)
        return None
    l=lbp(i,n,r,method="uniform")
    hist, _ = np.histogram(l.ravel(), bins=np.arange(0, n + 3), range=(0, n + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)  # Normalize histogram
    '''plt.hist(l.ravel(), bins=n + 2, range=(0, n + 2), density=True, color=plt.cm.viridis(0.5)  ,edgecolor='black')
    plt.xlabel("LBP Pattern Index")
    plt.ylabel("Normalized Frequency")
    plt.title("LBP Histogram (Computed with Matplotlib)")
    plt.grid(axis='y', linestyle='--')
    plt.show()'''
    return hist

def vgg(img):
    img1=cv2.imread(img)
    if img1 is None or not img1.size:
        print("failure",img)
        return None
    it=transform(Image.fromarray(cv2.cvtColor(img1,cv2.COLOR_BGR2RGB))).unsqueeze(0)
    
    with torch.no_grad():
        features=vgg16.features(it)
    return features.flatten().numpy()
def facs(img):
    try:
        i=cv2.imread(img,cv2.IMREAD_GRAYSCALE)
        if i is None or not i.size:
            print("Fail",img)
            return None
        faces=detector(i)
        if len(faces)==0:
            return None
        facial=np.array([(p.x,p.y)for p in predictor(i,faces[0]).parts()])
        return facial.flatten()
    except Exception as e:
        print(e)

def pad_or_replace(features, target_length):
    if features is None:
        
        return np.zeros(target_length)
    elif len(features) < target_length:
       
        return np.pad(features, (0, target_length - len(features)), mode='constant')
    else:
       
        return features[:target_length]

def combine(img, vgg_length=4096, lbp_length=10, facs_length=136):
    
    vgg_features = vgg(img)
    process_features = process(img)
    facs_features = facs(img)

    vgg_features = pad_or_replace(vgg_features, vgg_length)
    process_features = pad_or_replace(process_features, lbp_length)
    facs_features = pad_or_replace(facs_features, facs_length)

    combined_features = np.hstack([vgg_features, process_features, facs_features])
    return combined_features
