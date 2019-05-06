import os
from flask import Flask, render_template, request
import numpy as np
import base64
import glob
from PIL import Image
import sklearn
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.externals import joblib
import pickle

app = Flask(__name__)

UPLOAD_FOLDER = os.path.basename('uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def start_page():
    print("Start")
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files.get('image')

    # Save file
    #filename = 'static/' + file.filename
    #file.save(filename)

    # Read image
    #img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    #img=np.array(Image.open(file.read()))
    #img = np.array(file.read())

    #filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    #file.save(file.filename)
    file.save(file.filename)
    
    img=np.array(Image.open(file.filename))
    img = img.reshape(img.shape[0]*img.shape[1])
    
    os.remove(file.filename)
    # Detect 

    with open('trained_parameters/eigenfaces_yale.pickle', 'rb') as f:
        eigen_faces = pickle.load(f)
    with open('trained_parameters/normalize', 'rb') as f:
        normalizer = pickle.load(f)
    #with open('model_knn', 'rb') as f:
        #clf2 = pickle.load(f)
    with open('trained_parameters/trainX_norm.pickle', 'rb') as f:
        trainX_norm = pickle.load(f)
    with open('trained_parameters/trainY.pickle', 'rb') as f:
        trainY = pickle.load(f)

    #filename= 'finalized_modelknn.sav'
    #clf2_new = joblib.load(filename)

    projected_testX_img = np.dot(img, eigen_faces[:30].T)
    projected_testX_img_2=projected_testX_img.reshape(1, -1)
    testX_norm_img = normalizer.transform(projected_testX_img_2)
    clf2= KNeighborsRegressor(n_neighbors=1)
    clf2.fit(trainX_norm,trainY)
    c=clf2.predict(testX_norm_img)
    if len(c) > 0:
        faceDetected = True
    else:
        faceDetected = False


    return render_template('index.html', faceDetected=faceDetected, num_faces= c[0]+1, init=True)


if __name__ == "__main__":
    # Only for debugging while developing
    #app.run(host='0.0.0.0', debug=True, port=80)
    app.run(debug=True)