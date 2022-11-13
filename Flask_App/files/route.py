from flask import Flask,render_template,redirect,url_for
from files import app
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField, validators
from werkzeug.utils import secure_filename
import os
import cv2
import pickle
from skimage import feature 


class file_upload(FlaskForm):
    filew = FileField(label='Choose wave image',validators=[validators.DataRequired()])
    files = FileField(label='Choose spiral image',validators=[validators.DataRequired()])
    submit = SubmitField(label='Predict')

@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/info')
def info_page():
    return render_template('info.html')

@app.route('/result',defaults={'results':"Something went wrong",'resultw':"Something went wrong"})
@app.route('/result/<string:results>/<string:resultw>')
def result_page(results,resultw):
    return render_template('result.html',res=results,rew=resultw)

@app.route('/predict',methods=['GET', 'POST'])
def predict_page():
    form = file_upload()
    if form.validate_on_submit():
        filewave = form.filew.data
        filespiral = form.files.data
        filewave.save(os.path.join(os.path.abspath(os.path.dirname(__file__)),
                    app.config['UPLOAD_FOLDER_WAVE'],
                    secure_filename(filewave.filename)))
        filespiral.save(os.path.join(os.path.abspath(os.path.dirname(__file__)),
                    app.config['UPLOAD_FOLDER_SPIRAL'],
                    secure_filename(filespiral.filename)))

        modelw = pickle.loads(open('parkinsonWave.pkl', 'rb').read())
        models = pickle.loads(open('parkinsonSpiral.pkl', 'rb').read())  

        def pre_wave(loc):
            file = os.listdir(loc)[0]
            img = os.path.join(loc, file)
            image = cv2.imread(img)
            output=image.copy()
            output=cv2.resize(output, (128, 128)) 
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image=cv2.resize(image, (200, 200)) 
            image=cv2.threshold (image, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU) [1]
        
            features = feature.hog(image, orientations=9,
            pixels_per_cell=(10, 10), cells_per_block=(2, 2),
            transform_sqrt=True, block_norm="L1") 
            preds=modelw.predict([features])
            print(preds)
            ls=["healthy", "parkinson"]
            result = ls[preds[0]]
            return result

        def pre_spiral(loc):
            file = os.listdir(loc)[0]
            img = os.path.join(loc, file)
            image = cv2.imread(img)
            output=image.copy()
            output=cv2.resize(output, (128, 128)) 
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image=cv2.resize(image, (200, 200)) 
            image=cv2.threshold (image, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU) [1]
        
            features = feature.hog(image, orientations=9,
            pixels_per_cell=(10, 10), cells_per_block=(2, 2),
            transform_sqrt=True, block_norm="L1") 
            preds=models.predict([features])
            print(preds)
            ls=["healthy", "parkinson"]
            result = ls[preds[0]]
            return result

        locw = f'G:\\Placement\\parkinson_wave_spiral\\Flask_App\\files\\static\\images\\wave'
        locs = f'G:\\Placement\\parkinson_wave_spiral\\Flask_App\\files\\static\\images\\spiral'
        resultw = pre_wave(locw)
        results = pre_spiral(locs)
        

        return redirect(url_for('result_page',resultw=resultw,results=results))
    return render_template('index6.html',form=form)