import joblib
import lbp
import cv2
import dlib
detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
xgb=joblib.load('xgb_model.joblib')
ss=joblib.load('scaler.joblib')

def out(img):
    print(img)
    x=lbp.combine(img)

    face=detector(cv2.imread(img,cv2.IMREAD_GRAYSCALE))
    if face!=None:
        if xgb.predict([x])==[1]:
            return 1
        elif xgb.predict([x]):
            return 2
    return 0