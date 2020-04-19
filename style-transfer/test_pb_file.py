from __future__ import print_function
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0";

import sys
sys.path[0] = "/tensorflow-2.1.0/python3.6"
import tensorflow as tf	 
import cv2	
import time
import numpy as np
import argparse
import matplotlib.pyplot as plt
from scipy import misc

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', default='model/style_graph_frozen.pb', help='the path to the model file')
    parser.add_argument('-n', '--test_image', default='images/test/test2.jpg', help='the name of the model')
    parser.set_defaults(is_debug=False)
    return parser.parse_args()
def imshow(image, title=None):
  if len(image.shape) > 3:
    image = tf.squeeze(image, axis=0)

  plt.imshow(image)
  if title:
    plt.title(title)

def main(args):
  img = cv2.imread(args.test_image)
  with tf.Graph().as_default():
    output_graph_path = args.model_name
    with tf.gfile.FastGFile(output_graph_path, 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read()) 
      _ = tf.import_graph_def(graph_def, name='') 
    with tf.Session() as sess:
      tf.initialize_all_variables().run()
      input_name = graph_def.node[0].name
      print('Input node name: ', input_name)
      output_name = graph_def.node[-1].name
      print('Output node name: ', output_name)
      input_x = sess.graph.get_tensor_by_name(str(input_name)+":0")
      print(input_x)
      output = sess.graph.get_tensor_by_name(str(output_name)+":0")
      print(output)

      # converter = tf.lite.TFLiteConverter.from_frozen_graph(output_graph_path, input_arrays=[input_name], input_shapes={'input':[512, 512, 3]},output_arrays=[output_name])
      converter = tf.lite.TFLiteConverter.from_frozen_graph(output_graph_path, input_arrays=[input_name],output_arrays=[output_name])
      converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
      tflite_model=converter.convert()
      open("drive/My Drive/XXX/XXX_model_pb.tflite", "wb").write(tflite_model)
      generated = output
      '''generated = tf.cast(output, tf.uint8)
      generated = tf.squeeze(generated, [0])'''
      
      start_time = time.time()
      
      img_ = cv2.resize(img,(512,512))
      X = np.zeros((1,512,512,3),dtype=np.float32)
      X[0] = img_
      print('Input shape: ', X.shape)
      image_transfer = sess.run(generated, feed_dict={input_x: X})
      image_transfer = (image_transfer - image_transfer.min())/(image_transfer.max()-image_transfer.min())
      image_transfer = (image_transfer * 255).astype('uint8')
      misc.imsave('drive/My Drive/XXX/XXX_style_transfer.jpg', image_transfer)
      print('Output shape: ', image_transfer.shape)
      
      end_time = time.time()
      time_spend = end_time - start_time
      print('Time cost: ', time_spend)
if __name__ == '__main__':
    args = parse_args()
    print(args)
    main(args)
    pb_path = args.model_name

				 

