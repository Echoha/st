import time
import tensorflow as tf
from src.functions import *
from tensorflow.python.framework import graph_util
import time
from tensorflow.python.tools.freeze_graph import freeze_graph
from tensorflow.python import pywrap_tensorflow
class op(object):

    
    def __init__(self, args, sess):
        self.sess = sess

        ## Train
        self.gpu_number = args.gpu_number
        self.project_name = args.project

        ## Train images
        self.content_dataset = args.content_dataset ## test2015
        self.content_data_size = args.content_data_size
        self.style_image = args.style_image

        # result dir
        self.result_dir = args.result_dir
        if tf.gfile.Exists(self.result_dir):
          tf.gfile.DeleteRecursively(self.result_dir)
        else:
          tf.gfile.MakeDirs(self.result_dir)
        ### graph names
        self.graph_model_dir = os.path.join(self.result_dir,'model')
        if tf.gfile.Exists(self.graph_model_dir):
          tf.gfile.DeleteRecursively(self.graph_model_dir)
        else:
          tf.gfile.MakeDirs(self.graph_model_dir)
        self.graph_model_name = self.result_dir + '_graph'
        ### save result image
        self.result_image_dir = os.path.join(self.result_dir, 'image')
        if tf.gfile.Exists(self.result_image_dir):
          tf.gfile.DeleteRecursively(self.result_image_dir)
        else:
          tf.gfile.MakeDirs(self.result_image_dir)

        

        ## Train Iteration
        self.niter = args.niter
        self.niter_snapshot = args.nsnapshot
        self.max_to_keep = args.max_to_keep

        ## Train Parameter
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.alpha = args.alpha
        self.momentum = args.momentum
        self.momentum2 = args.momentum2

        self.content_weights = args.content_loss_weights
        self.style_weights = args.style_loss_weights
        self.tv_weight = args.tv_loss_weight

        ## Result Dir & File
        self.project_dir = '{0}/'.format(self.project_name)
        make_project_dir(self.project_dir)
        self.ckpt_dir = os.path.join(self.project_dir, 'models')

        ## Test
        
        # self.test_dataset = args.test_dataset
        self.test_image = args.test_image
        self.style_control = args.style_control_weights

        ## build model
        self.build_model()


    def train(self,Train_flag):
        data = data_loader(self.content_dataset)
        print('Shuffle ....')
        random_order = np.random.permutation(len(data))
        print(len(data))
        data = [data[i] for i in random_order[:10000*self.batch_size]]
        print('Shuffle Done')

        start_time = time.time()
        count = 0

        try:
            self.load()
            print('Weight Load !!')
        except:
            self.sess.run(tf.global_variables_initializer())

        for epoch in range(self.niter):
            batch_idxs = len(data) // self.batch_size

            for idx in range(0, batch_idxs):
                count += 1

                batch_files = data[idx * self.batch_size: (idx + 1) * self.batch_size]
                batch_label = [(get_image(batch_file, self.content_data_size)) for batch_file in batch_files]

                feeds = {self.content_input: batch_label}

                _, loss_all, loss_c, loss_s, loss_tv = self.sess.run(self.optimize, feed_dict=feeds)
                train_time = time.time() - start_time
                if count % self.niter_snapshot == int((self.niter_snapshot - 1) / 2) or :
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.4f, loss_c: %.4f, loss_s: %.4f, loss_tv: %.4f"
                        % (epoch, idx, batch_idxs, train_time, loss_all, loss_c, loss_s, loss_tv))
                
                
                ## Test during Training
                if count % self.niter_snapshot == (self.niter_snapshot-1):
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.4f, loss_c: %.4f, loss_s: %.4f, loss_tv: %.4f"
                        % (epoch, idx, batch_idxs, train_time, loss_all, loss_c, loss_s, loss_tv))
                    self.count = count
                    self.save()
                    
                    # save model graph
                    self.evaluate_img(img_in=self.test_image, img_path=os.path.join(self.result_image_dir, self.result_dir + '_'+ str(count)+'.jpg'))
                    # Freeze graph.
                    # saver = tf.train.Saver()
                    if os.path.isdir(self.ckpt_dir):
                        ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
                        if ckpt and ckpt.model_checkpoint_path:
                            self.load()
                        else:
                            raise Exception("No checkpoint found...")
                    else:
                        ckpt = self.saver.save(self.sess, os.path.join(self.ckpt_dir, self.model_name), global_step=self.count)
                    freeze_graph(
                        input_graph=os.path.join(self.graph_model_dir,self.graph_model_name + '.pb.txt'),
                        input_saver='',
                        input_binary=False,
                        input_checkpoint=ckpt.model_checkpoint_path,
                        output_node_names='output_1',
                        restore_op_name='save/restore_all',
                        filename_tensor_name='save/Const:0',
                        output_graph=os.path.join(self.graph_model_dir, self.graph_model_name + '_frozen.pb'),
                        clear_devices=False,
                        initializer_nodes='')
                    print('save frozen graph down')
          


    
      
    def save(self):
        style_name = os.path.basename(self.style_image)[:-4]
        self.model_name = "{0}.model".format(style_name)

        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
        self.saver.save(self.sess, os.path.join(self.ckpt_dir, self.model_name), global_step=self.count)
        
    def load(self):
        ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        self.saver.restore(self.sess, os.path.join(self.ckpt_dir, ckpt_name))
    
    def evaluate_img(self, img_in, img_path):
      img_shape = (512, 512, 3)
      batch_shape = (1,512, 512, 3)
          
      soft_config = tf.ConfigProto(allow_soft_placement=True)
      soft_config.gpu_options.allow_growth = True
      with tf.Graph().as_default(), tf.Session(config=soft_config) as sess:
          # Declare placeholders we'll feed into the graph
          X_inputs = tf.placeholder(
              tf.float32, [1, 512, 512, 3], name='X_inputs')

          # Define output node
          preds = self.mst_net(X_inputs, alpha=self.alpha, style_control=self.style_control, reuse=True)
          tf.identity(preds[0], name='output')
          

          # For restore training checkpoints (important)
          # saver = tf.train.Saver()
          # ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
          # saver.restore(sess, ckpt)  # run
          saver = tf.train.Saver()
          if os.path.isdir(self.ckpt_dir):
              ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
              if ckpt and ckpt.model_checkpoint_path:
                  saver.restore(sess, ckpt.model_checkpoint_path)  # run
              else:
                  raise Exception("No checkpoint found...")
          else:
              ckpt = saver.restore(sess, self.ckpt_dir)
          

          X = np.zeros(batch_shape, dtype=np.float32)  # feed

          img = get_img(img_in, img_shape)
          X[0] = img

          _preds = sess.run(preds, feed_dict={X_inputs: X})
          save_img(img_path, _preds[0])

          # Write graph.
          # start_time = time.time()
          tf.train.write_graph(
              sess.graph.as_graph_def(),
              self.graph_model_dir,
              self.graph_model_name + '.pb',
              as_text=False)
          tf.train.write_graph(
              sess.graph.as_graph_def(),
              self.graph_model_dir,
              self.graph_model_name + '.pb.txt',
              as_text=True)
          print('Save pb and pb.txt done!')
    
