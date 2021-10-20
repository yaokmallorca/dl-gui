# # # # # # # # # # # # # # # 
# Author: Mustafa Mert Tunalı
# ---------------------------
# ---------------------------
# Deep Learning Training GUI - Backend Side
# ---------------------------
# ---------------------------
# # # # # # # # # # # # # # # 

import numpy as np
from multiprocessing import Process
import threading
from flask import Flask, request, jsonify, render_template
# from dltgui.dlgui import dl_gui
from dltgui.dlgui_torch import dl_gui
import tensorflow as tf

import pandas as pd
import glob
import json
import random
import time
from datetime import datetime
import base64
import io

# Set Flask
app = Flask(__name__, static_folder="static/")

global_params                     = {}
global_params['cur_epoch']        = 0
global_params['training_epoch']   = 10
global_params['dataset']          = ""
global_params['task_type']        = ""
global_params['model']            = ""
global_params['dataset']          = ""
global_params['infer_index']      = 0
global_params['infer_input']      = []
global_params['infer_result']     = []

@app.route('/')
def home():
    return render_template('index.html', title ="Version 1.0.2")


@app.route('/training')
def training():
    return render_template('training.html')

@app.route('/terminal-object-detection',methods = ['POST'])
def terminal_object_detection():
    if request.method == 'POST':
      result = request.form
      dataset = result['dataset']
      type_of_label = result['type_of_label']
      # split_dataset = result['split_dataset']
      project_name = result['project_name']
      pre_trained_model = result['Pre-trained Model']
      number_of_classes = result['noc']
      batch_size = result['batch_size']
      epoch = result['epoch']
      return render_template("terminal-object-detection.html",result = result)


@app.route('/terminal',methods = ['POST', 'GET'])
def terminal():
    if request.method == 'POST':
        '''Read the values from HTML file and set the values for training.''' 
        result = request.form
        dataset = result['dataset']
        project_name = result['project_name']
        pre_trained_model = result['Pre-trained Model']
        number_of_classes = result['noc']
        batch_size = result['batch_size']
        epoch = result['epoch']
        cpu_gpu = result['CPU/GPU']

        task_type = result['task_type']
        if task_type == "cl":
            task_type = "classification"
            flip = result['flip'] 
            rotation = result['rotation'] 
            zoom = result['zoom']
        elif task_type == "seg":
            task_type = "segmentation"
            flip = result['flip']
            cropresize = result['cropresize']
            loss_type = result['loss_type']
        
        input_size_str = result['input_size']
        input_size = tuple(list(map(int, input_size_str.split(','))))      
        fine_tune_epochs = 10 # result['fine_tune_epochs']
        learning_rate = result['learning_rate']
        image_type = result['image_type']
        optim_type = result['optim']

        global_params['cur_epoch'] = 0
        global_params['training_epoch'] = int(result['epoch'])
        global_params['dataset'] = dataset
        gui = dl_gui(dataset=dataset, 
                     task_type = task_type,
                     project_name = str(project_name),
                     input_size = input_size, 
                     pre_trained_model = pre_trained_model,
                     cpu_gpu = cpu_gpu,
                     number_of_classes = int(number_of_classes), 
                     batch_size = int(batch_size), 
                     epoch = int(epoch),
                     learning_rate = float(learning_rate),
                     fine_tune_epochs = int(fine_tune_epochs),
                     img_suffix = str(image_type),
                     stage='train')

        if task_type == 'classification':
            aug_ops = [flip, rotation, zoom]
            if flip == "True" or rotation == "True" or zoom == "True":
                gui.load_dataset(n_class = number_of_classes, 
                                imgaugmentation = True, aug_ops = aug_ops, 
                                task = task_type)
            else:
                gui.load_dataset(n_class = number_of_classes, 
                                imgaugmentation = False, aug_ops = [], 
                                task = task_type)

            train_thread_gui = threading.Thread(target= gui.train, 
                    args=(task_type, optim_type, global_params, )).start() 
            # training_threads.append(train_thread_gui)
            return render_template("terminal.html",result = result,)
        elif task_type == 'segmentation':
            aug_ops = [flip, cropresize]
            if flip == "True" or cropresize == "True":
                gui.load_dataset(n_class = number_of_classes, 
                                imgaugmentation = True, aug_ops = aug_ops, 
                                task = task_type)
            else:
                gui.load_dataset(n_class = number_of_classes, 
                                imgaugmentation = False, aug_ops = [], 
                                task = task_type)

            print("loss_type: ", loss_type)
            train_thread_gui = threading.Thread(target= gui.train, 
                    args=(task_type, optim_type, global_params, loss_type,)).start() 
            # training_threads.append(train_thread_gui)
            return render_template("terminal.html",result = result,)
        else:
            NotImplemented



@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/result', methods = ['POST'])
def result():
    if request.method == 'POST':
        result              = request.form
        task                = result['task_type']
        dataset             = result['dataset']
        model_dir           = result['model']
        trained_model       = result['Pre-trained Model']
        class_ind           = result["class_ind"]
        noc                 = result['noc']
        input_size_str      = result['input_size']
        input_size          = tuple(list(map(int, input_size_str.split(',')))) 

        global_params['task_type']    = task
        global_params['model']        = trained_model
        global_params['dataset']      = dataset
        global_params['infer_input']  = []
        global_params['infer_result'] = []
        global_params['infer_index']  = 0

        gui = dl_gui(
            task_type = task,
            input_size = input_size,
            project_name = "Test", 
            dataset=dataset,
            number_of_classes = int(noc),
            pre_trained_model = trained_model,
            stage='test'
            )
        input_imgs, pred_result, show_heatmap = gui.predict(dataset, model_dir, class_ind, task)

        global_params['infer_input']  = input_imgs
        global_params['infer_result'] = pred_result
        ind = global_params['infer_index']


        img_data = io.BytesIO()
        global_params['infer_input'][ind].save(img_data, "PNG")
        input_img_data = base64.b64encode(img_data.getvalue())
        if show_heatmap:
          pred_data = io.BytesIO()
          global_params['infer_result'][ind].save(pred_data, "PNG")
          pred_img_data = base64.b64encode(pred_data.getvalue())
          return render_template('result.html', result= result, pred_img = pred_img_data.decode('utf-8'), 
                input_img = input_img_data.decode('utf-8'), show_heatmap=show_heatmap,) #  mimetype="text/event-stream"
        else:
          pred_data = global_params['infer_result'][ind]
          return render_template('result.html', result= result, pred_img = pred_data, 
                input_img = input_img_data.decode('utf-8'), show_heatmap=show_heatmap,) #  mimetype="text/event-stream"


@app.route('/show_result', methods = ['POST'])
def show_result():
    if request.method == 'POST':
        
          result = {}
          result['task_type'] = global_params['task_type']
          result['model']     = global_params['model']
          result['dataset']   = global_params['dataset']

          if global_params['infer_index'] < len(global_params['infer_input'])-1:
            global_params['infer_index'] += 1
          ind = global_params['infer_index']

          img_data = io.BytesIO()
          global_params['infer_input'][ind].save(img_data, "PNG")
          input_img_data = base64.b64encode(img_data.getvalue())

          if result['task_type'] == 'segmentation':
            show_heatmap = True
            pred_data = io.BytesIO()
            global_params['infer_result'][ind].save(pred_data, "PNG")
            pred_img_data = base64.b64encode(pred_data.getvalue())
            return render_template('result.html', result= result, pred_img = pred_img_data.decode('utf-8'), 
                  input_img = input_img_data.decode('utf-8'), show_heatmap=show_heatmap,) #  mimetype="text/event-stream"
          else:
            show_heatmap = False
            pred_data = global_params['infer_result'][ind]
            return render_template('result.html', result= result, pred_img = pred_data, 
                  input_img = input_img_data.decode('utf-8'), show_heatmap=show_heatmap,) #  mimetype="text/event-stream"

@app.route('/_stuff', methods = ['GET'])
def stuff():
    return jsonify(cur_epoch=global_params['cur_epoch'])

@app.route('/test', methods = ['POST'])
def test():
     if request.method == 'POST':
      '''Read the values from HTML file and set the values for training.''' 
      result = request.form
      dataset = result['dataset']
      # split_dataset = result['split_dataset']
      project_name = result['project_name']
      pre_trained_model = result['Pre-trained Model']
      cpu_gpu = result['CPU/GPU']
      number_of_classes = result['noc']
      batch_size = result['batch_size']
      epoch = result['epoch']
      flip = result['flip'] 
      rotation = result['rotation'] 
      zoom = result['zoom'] 
      return "Testing page - look at the conda terminal for values.."

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=False,) # threaded=True
   
   