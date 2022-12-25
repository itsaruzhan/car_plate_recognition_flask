"""
Simple app to upload an image via a web form 
and view the inference results on the image in the browser.
"""
import argparse
import io
import os
from PIL import Image
import datetime
import pytesseract
import cv2
import numpy as np
import sqlite3 as sql
import re
import torch
from flask import Flask, render_template, request, redirect, url_for
from pathlib import Path
import math
import pandas as pd
import cv2


pytesseract.pytesseract.tesseract_cmd = r'C:\Users\anarb\AppData\Local\Tesseract-OCR\tesseract.exe'
app = Flask(__name__)
DATETIME_FORMAT = "%Y-%m-%d_%H-%M-%S-%f"

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def compute_skew(src_img):
    
    if len(src_img.shape) == 3:
        h, w, _ = src_img.shape
    elif len(src_img.shape) == 2:
        h, w = src_img.shape
    else:
        print('upsupported image type')
        
    img = cv2.medianBlur(src_img, 3)
    
    edges = cv2.Canny(img,  threshold1 = 30,  threshold2 = 100, apertureSize = 3, L2gradient = True)
    lines = cv2.HoughLinesP(edges, 1, math.pi/180, 30, minLineLength=w / 4.0, maxLineGap=h/4.0)
    angle = 0.0
    nlines = lines.size
    
    #print(nlines)
    cnt = 0
    for x1, y1, x2, y2 in lines[0]:
        ang = np.arctan2(y2 - y1, x2 - x1)
        #print(ang)
        if math.fabs(ang) <= 30: # excluding extreme rotations
            angle += ang
            cnt += 1
    
    if cnt == 0:
        return 0.0
    return (angle / cnt)*180/math.pi

def deskew(src_img):
    return rotate_image(src_img, compute_skew(src_img))
  

@app.route('/get_number/plate_num')
def get_number():
    plate_num = request.args.get('plate_num')
    print(plate_num)
    image_path = request.args.get('image_path')
    image_path = "/" + image_path
    if plate_num is not None:
        
        with sql.connect("database.db") as con:
                cur = con.cursor()
                q_name = f"Select  c_name from customers  Where license_number Like '{plate_num}' ;" 
                q_surname = f"Select  c_surname from customers Where license_number Like '{plate_num}' ;" 
                q_job = f"Select  jop_position from customers Where license_number Like '{plate_num}' ;" 
                print(q_name)
                cur.execute(q_name)
                c_name = cur.fetchall()
                if len(c_name)>0  :
                    customer_name = c_name[0][0]
                    cur.execute(q_surname)
                    customer_surname = cur.fetchall()[0][0]
                    cur.execute(q_job)
                    customer_job = cur.fetchall()[0][0]
                    
                else: 
                    return render_template("not_found.html",image_path = image_path)   
    else:
        return render_template("not_found.html", image_path = image_path)

    return render_template('results.html', name=customer_name,surname = customer_surname, image_path = image_path,plate_num=plate_num,customer_job=customer_job)

@app.route("/", methods=["GET", "POST"])
def predict():
  
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if not file:
            return
            
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        results = model([img])

        results.render()  # updates results.imgs with boxes and labels
        now_time = datetime.datetime.now().strftime(DATETIME_FORMAT)
        img_savename = f"static/detected/{now_time}.png"
        Image.fromarray(results.ims[0]).save(img_savename)    
                
        for i, row in results.pandas().xyxy[0].iterrows():
            x1 = row['xmin']
            x2 = row['xmax']
            y1 = row['ymin']
            y2 = row['ymax']
            img2 = np.array(img)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            img_alpr = img2[int(y1):int(y2),int(x1):int(x2)]  
            alrp_filename = f'static/cropped/{now_time}.png'
            corrected_img = deskew(img_alpr)
            cv2.imwrite(alrp_filename, corrected_img)
            gray = cv2.imread(alrp_filename, cv2.IMREAD_GRAYSCALE)
            gray = cv2.resize( gray, None, fx = 3, fy = 3, interpolation = cv2.INTER_CUBIC)
            blur = cv2.GaussianBlur(gray, (5,5), 0)
            gray = cv2.medianBlur(gray, 3)
            # perform otsu thresh (using binary inverse since opencv contours work better with white text)
            ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

            rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

            # apply dilation 
            dilation = cv2.dilate(thresh, rect_kern, iterations = 1)
            #cv2.imshow("dilation", dilation)
            #cv2.waitKey(0)
            # find contours
            try:
                contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            except:
                ret_img, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

            # create copy of image
            im2 = gray.copy()

            plate_num = ""
            # loop through contours and find letters in license plate
            for cnt in sorted_contours:
                x,y,w,h = cv2.boundingRect(cnt)
                height, width = im2.shape
                
                # if height of box is not a quarter of total height then skip
                if height / float(h) > 6: continue
                ratio = h / float(w)
                # if height to width ratio is less than 1.5 skip
                if ratio < 1.2: continue
                area = h * w
                # if width is not more than 25 pixels skip
                if width / float(w) > 18: continue
                # if area is less than 100 pixels skip
                if area < 100: continue
                # draw the rectangle
                rect = cv2.rectangle(im2, (x,y), (x+w, y+h), (0,255,0),2)
                roi = thresh[y-5:y+h+5, x-5:x+w+5]
                roi = cv2.bitwise_not(roi)
                roi = cv2.medianBlur(roi, 5)
                #cv2.imshow("ROI", roi)
                #cv2.waitKey(0)
                text = pytesseract.image_to_string(roi, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8 --oem 3')
                text = text.strip()
                plate_num += text      
            plate_num = str(plate_num).strip()
            pattern = r'[^A-Za-z0-9]+'
            plate_num = re.sub(pattern, '', plate_num).upper()
        
            print(plate_num)  

            if len(plate_num) == 8 :
                return redirect(url_for("get_number", plate_num=plate_num, image_path = img_savename)) 
            elif(len(plate_num)==9):
                plate_num = plate_num[1::]
                return redirect(url_for("get_number", plate_num=plate_num, image_path = img_savename)) 
        img_savename = "/" +img_savename    
        print(img_savename)
        return render_template("not_found.html", image_path=img_savename)      
    return render_template("index.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing yolov5 models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()
    CKPT_PATH = 'best1.pt'
    model = torch.hub.load('yolov5/','custom',path=CKPT_PATH,source='local',force_reload=True)
    #model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)  # force_reload = recache latest code
    model.eval()
    app.run(host="0.0.0.0", port=args.port)  # debug=True causes Restarting with stat

