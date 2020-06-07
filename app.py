import os
from flask import Flask, flash, request, redirect, url_for,render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img,img_to_array
#from flask_ngrok import run_with_ngrok
import numpy as np
import cv2

UPLOAD_FOLDER = 'static/upload'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg','mp4'}
model=load_model("mod2.hdf5")
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#run_with_ngrok(app)


def allowed_file(filename):
	return '.' in filename and \
		   filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_video(vid,name):
	
	path='static/upload/temp'
	vidcap = cv2.VideoCapture(vid)
	def getFrame(sec):
	    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
	    hasFrames,image = vidcap.read()
	    if hasFrames:
	        cv2.imwrite(path+"/"+name+str(count)+".jpg", image)     # save frame as JPG file
	    return hasFrames
	sec = 0
	frameRate = 0.5 #//it will capture image in each 0.25 second
	count=1
	success = getFrame(sec)
	while success:
	    count = count + 1
	    sec = sec + frameRate
	    sec = round(sec, 2)
	    success = getFrame(sec)
	final_pred=np.zeros(3)
	din=len(os.listdir(path))
	for item in os.listdir(path):
	    img  = load_img(os.path.join(path,item), target_size=(224,224))
	    img = img_to_array(img)/255.0
	    img = np.expand_dims(img, axis=0)
	    probs = model.predict(img)
	    os.remove(os.path.join(path,item))
	    final_pred+=probs[0]
	return(final_pred/din)

def predict_img(img):
	img  = load_img(img, target_size=(224,224))
	img = img_to_array(img)/255.0
	img = np.expand_dims(img, axis=0)
	probs = model.predict(img)
	#print(type(probs))
	return(probs[0])


@app.route('/result_vid <pred0> <pred1> <pred2> <img>', methods=['GET', 'POST'])
def result_vid(pred0,pred1,pred2,img):
	if request.method == 'POST':
		return redirect(url_for('upload_file'))
	img=os.path.join(app.config['UPLOAD_FOLDER'],img)
	pred0=round(float(pred0),3)*100
	pred1=round(float(pred1),3)*100
	pred2=round(float(pred2),3)*100
	return render_template('result_vid.html',p0=pred0,p1=pred1,p2=pred2,image=img)

@app.route('/result_img <pred0> <pred1> <pred2> <img>', methods=['GET', 'POST'])
def result_img(pred0,pred1,pred2,img):

	if request.method == 'POST':
		return redirect(url_for('upload_file'))
	img=os.path.join(app.config['UPLOAD_FOLDER'],img)
	pred0=round(float(pred0),3)*100
	pred1=round(float(pred1),3)*100
	pred2=round(float(pred2),3)*100
	return render_template('result_img.html',p0=pred0,p1=pred1,p2=pred2,image=img)



@app.route('/', methods=['GET', 'POST'])
def upload_file():
	if request.method == 'POST':
		file = request.files['file']
		#print(file)
		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)
			#print(filename)
			if (filename[-1:-4:-1][::-1]) =='mp4':
				file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
				l=predict_video(os.path.join(app.config['UPLOAD_FOLDER'], filename),filename[:-4])
				print(l)
				return redirect(url_for('result_vid',pred0=l[0],pred1=l[1],pred2=l[2],img=filename))
			else:
				file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
				l=predict_img(os.path.join(app.config['UPLOAD_FOLDER'], filename))
				return redirect(url_for('result_img',pred0=l[0],pred1=l[1],pred2=l[2],img=filename))
	return(render_template("home.html"))

if __name__ == '__main__':
	app.run(debug=True)
