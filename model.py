from __future__ import print_function

import os
import time                                                      
import random                                                     
from PIL import Image                                             
                                   
import tensorflow.compat.v1 as tf
                                      
from utils import *                                               

def Server(input_im):                                                                              
  with tf.variable_scope('Server'):                                                                 
    input_rs = tf.image.resize_images(input_im, [96, 96],method=0)                             
                                                                
    p_conv1 = tf.layers.conv2d(input_rs, 64, 3, 2, padding='same', activation=tf.nn.relu)  
    p_conv2 = tf.layers.conv2d(p_conv1,  64, 3, 2, padding='same', activation=tf.nn.relu)   
    p_conv3 = tf.layers.conv2d(p_conv2,  64, 3, 2, padding='same', activation=tf.nn.relu) 
    p_conv4 = tf.layers.conv2d(p_conv3,  64, 3, 2, padding='same', activation=tf.nn.relu)  
    p_conv5 = tf.layers.conv2d(p_conv4,  64, 3, 2, padding='same', activation=tf.nn.relu)    
    p_conv6 = tf.layers.conv2d(p_conv5,  64, 3, 2, padding='same', activation=tf.nn.relu)    

    p_deconv1 = tf.image.resize_images(p_conv6, [3, 3],method=1)                         
    p_deconv1 = tf.layers.conv2d(p_deconv1, 64, 3, 1, padding='same', activation=tf.nn.relu)    
    p_deconv1 = p_deconv1 + p_conv5                                                           
    p_deconv2 = tf.image.resize_images(p_deconv1, [6, 6],method=1)                            
    p_deconv2 = tf.layers.conv2d(p_deconv2, 64, 3, 1, padding='same', activation=tf.nn.relu)                
    p_deconv2 = p_deconv2 + p_conv4                                                                         
    p_deconv3 = tf.image.resize_images(p_deconv2, [12, 12],method=1)                           
    p_deconv3 = tf.layers.conv2d(p_deconv3, 64, 3, 1, padding='same', activation=tf.nn.relu)    
    p_deconv3 = p_deconv3 + p_conv3
    p_deconv4 = tf.image.resize_images(p_deconv3, [24, 24],method=1)                           
    p_deconv4 = tf.layers.conv2d(p_deconv4, 64, 3, 1, padding='same', activation=tf.nn.relu)    
    p_deconv4 = p_deconv4 + p_conv2
    p_deconv5 = tf.image.resize_images(p_deconv4, [48, 48],method=1)                          
    p_deconv5 = tf.layers.conv2d(p_deconv5, 64, 3, 1, padding='same', activation=tf.nn.relu)
    p_deconv5 = p_deconv5 + p_conv1
    p_deconv6 = tf.image.resize_images(p_deconv5, [96, 96],method=1)                          
    p_deconv6 = tf.layers.conv2d(p_deconv6, 64, 3, 1, padding='same', activation=tf.nn.relu)
                                                                                                             
    p_output = tf.image.resize_images(p_deconv6, [tf.shape(input_im)[1], tf.shape(input_im)[2]],method=0)
                                                                                                             
    a_input = tf.concat([p_output, input_im], axis=3)                                           
                                                                                                
    print(p_output.shape)                                                                  
    p_midput = tf.reduce_sum(p_output,axis=3)    
    print(p_midput.shape)                                                                                                                         
    
    a_conv1_1 = tf.layers.conv2d(a_input, 128, 3, 1, padding='same')
    a_conv1 = tf.nn.relu(a_conv1_1)
    a_conv2_1 = tf.layers.conv2d(a_conv1, 128, 3, 1, padding='same')
    a_conv2 = tf.nn.relu(a_conv2_1)
    a_conv3_1 = tf.layers.conv2d(a_conv2, 128, 3, 1, padding='same')
    a_conv3 = tf.nn.relu(a_conv3_1 + a_conv1)                              
    a_conv4_1 = tf.layers.conv2d(a_conv3, 128, 3, 1, padding='same')
    a_conv4 = tf.nn.relu(a_conv4_1 + a_conv2)                             
    a_conv5 = tf.layers.conv2d(a_conv4, 3, 3, 1, padding='same', activation=tf.nn.relu)
    print(a_conv5.shape)                 
    return a_conv5, p_midput              
                     
class illumination_Corr(object):       
    def __init__(self, sess):             
        self.sess = sess                
        self.base_lr = 0.001          

        self.input_uneven = tf.placeholder(tf.float32, [None, None, None, 3], name='input_uneven')    
        self.input_normal = tf.placeholder(tf.float32, [None, None, None, 3], name='input_normal') 
        self.output ,self.midput= Server(self.input_uneven)     
        self.loss =  (1 - bright_SSIM(self.input_normal,self.output))*0.6 + 0.4*tf.reduce_mean(tf.abs((self.output - self.input_normal)))        
         
                                                                                           
        self.global_step = tf.Variable(0, trainable = False)                               
        self.lr = tf.train.exponential_decay(self.base_lr, self.global_step, 100, 0.9)  
        

        optimizer = tf.train.AdamOptimizer(self.lr, name='AdamOptimizer')

        self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)

        self.sess.run(tf.global_variables_initializer())                                  
                                                                                         
        self.saver = tf.train.Saver()                                                   
        print("[*] Initialize model successfully...")                                     



    def evaluate(self, epoch_num, eval_uneven_data, sample_dir):
        print("[*] Evaluating for epoch %d..." % (epoch_num))                              

        for idx in range(len(eval_uneven_data)):                                               
            input_uneven_eval = np.expand_dims(eval_uneven_data[idx], axis=0)                    
            result = self.sess.run(self.output, feed_dict={self.input_uneven: input_uneven_eval})
            save_images(os.path.join(sample_dir, 'eval_%d_%d.png' % (idx + 1, epoch_num)), input_uneven_eval, result)  


    def train(self, train_uneven_data, train_normal_data, eval_uneven_data, batch_size, patch_size, epoch, sample_dir, ckpt_dir, eval_every_epoch):
                                                                                           
        assert len(train_uneven_data) == len(train_normal_data)
        numBatch = len(train_uneven_data) // int(batch_size)                                         

        load_model_status, global_step = self.load(self.saver, ckpt_dir)
        if load_model_status:
            iter_num = global_step
            start_epoch = global_step // numBatch
            start_step = global_step % numBatch
            print("[*] Model restore success!")
        else:
            iter_num = 0
            start_epoch = 0
            start_step = 0
            print("[*] Not find pretrained model!")

        print("[*] Start training with start epoch %d start iter %d : " % (start_epoch, iter_num))

        start_time = time.time()
        image_id = 0

        for epoch in range(start_epoch, epoch):
            for batch_id in range(start_step, numBatch):
                batch_input_uneven = np.zeros((batch_size, patch_size, patch_size, 3), dtype="float32")
                batch_input_normal = np.zeros((batch_size, patch_size, patch_size, 3), dtype="float32")
                for patch_id in range(batch_size):
                    h, w, _ = train_uneven_data[image_id].shape
                    x = random.randint(0, h - patch_size)
                    y = random.randint(0, w - patch_size)
            
                    rand_mode = random.randint(0, 7)
                    batch_input_uneven[patch_id, :, :, :] = data_augmentation(train_uneven_data[image_id][x : x+patch_size, y : y+patch_size, :], rand_mode)
                    batch_input_normal[patch_id, :, :, :] = data_augmentation(train_normal_data[image_id][x : x+patch_size, y : y+patch_size, :], rand_mode)
                    
                    image_id = (image_id + 1) % len(train_uneven_data)
                    if image_id == 0:
                        tmp = list(zip(train_uneven_data, train_normal_data))
                        random.shuffle(list(tmp))
                        train_uneven_data, train_normal_data  = zip(*tmp)

                _, loss = self.sess.run([self.train_op, self.loss], feed_dict={self.input_uneven: batch_input_uneven, \
                                                                           self.input_normal: batch_input_normal})

                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.6f" \
                      % (epoch + 1, batch_id + 1, numBatch, time.time() - start_time, loss))
                iter_num += 1

            if (epoch + 1) % int(eval_every_epoch) == 0:
                self.evaluate(epoch + 1, eval_uneven_data, sample_dir=sample_dir)
                self.save(self.saver, iter_num, ckpt_dir, "ServerNet")

        print("[*] Finish training")

    def save(self, saver, iter_num, ckpt_dir, model_name):                                            
        if not os.path.exists(ckpt_dir):                                                      
            os.makedirs(ckpt_dir)                                                             
        print("[*] Saving model %s" % model_name)                                             
        saver.save(self.sess, \
                   os.path.join(ckpt_dir, model_name), \
                   global_step=iter_num)

    def load(self, saver, ckpt_dir):                                                          
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)                                        
        if ckpt and ckpt.model_checkpoint_path:                                              
            full_path = tf.train.latest_checkpoint(ckpt_dir)                                       
            try:
                global_step = int(full_path.split('/')[-1].split('-')[-1])
            except ValueError:
                global_step = None
            saver.restore(self.sess, full_path)
            return True, global_step
        else:
            print("[*] Failed to load model from %s" % ckpt_dir)
            return False, 0

    def test(self, test_uneven_data, test_normal_data, test_uneven_data_names, save_dir):             
        tf.global_variables_initializer().run()

        print("[*] Reading checkpoint...")
        load_model_status, _ = self.load(self.saver, './model/')
        if load_model_status:
            print("[*] Load weights successfully...")
        
        print("[*] Testing...")
        total_run_time = 0.0
        for idx in range(len(test_uneven_data)):
            print(test_uneven_data_names[idx])
            [_, name] = os.path.split(test_uneven_data_names[idx])
            suffix = name[name.find('.') + 1:]
            name = name[:name.find('.')]

            input_uneven_test = np.expand_dims(test_uneven_data[idx], axis=0)
            start_time = time.time()
           
            resultl = self.sess.run(self.output, feed_dict = {self.input_uneven: input_uneven_test})
            results = self.sess.run(self.midput, feed_dict = {self.input_uneven: input_uneven_test})
            total_run_time += time.time() - start_time
            
            save_images(os.path.join(save_dir, name + "_res."   + suffix), resultl)
            save_images(os.path.join(save_dir, name + "_mid."   + suffix), results)

        ave_run_time = total_run_time / float(len(test_uneven_data))
        print("[*] Average run time: %.4f" % ave_run_time)