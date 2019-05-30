import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
import keras.backend as K


def load_images(images_list, img_size):
    loaded_images = list()
    for img_path in images_list:
        img = image.load_img(img_path, target_size=(img_size, img_size))
        img = np.expand_dims(img, 0)
        loaded_images.append(img)
    loaded_images = np.vstack(loaded_images)
    preprocessed_images = preprocess_input(loaded_images)
    return preprocessed_images


class PerceptualModel:
    def __init__(self, img_size, layer=9, batch_size=1, sess=None):
        self.sess = tf.get_default_session() if sess is None else sess
        K.set_session(self.sess)
        self.img_size = img_size
        self.layer = layer
        self.batch_size = batch_size
        self.perceptual_loss_layers = {'conv1_1': 1, 'conv1_2': 2, 'conv3_2': 8, 'conv4_2': 12}
        #self.perceptual_loss_layers = {'conv4_2': 9}
        self.perceptual_model = None
        self.ref_img_features = None
        self.loss = None
        self.discriminator_conv_layer_variable_names = [
                'D/1024x1024/Conv0/LeakyReLU/IdentityN',
                'D/1024x1024/Conv1_down/LeakyReLU/IdentityN',
                'D/512x512/Conv0/LeakyReLU/IdentityN',
                'D/512x512/Conv1_down/LeakyReLU/IdentityN',
                'D/256x256/Conv0/LeakyReLU/IdentityN',
                'D/256x256/Conv1_down/LeakyReLU/IdentityN',
                'D/128x128/Conv0/LeakyReLU/IdentityN',
                'D/128x128/Conv1_down/LeakyReLU/IdentityN',
                'D/64x64/Conv0/LeakyReLU/IdentityN',
                'D/64x64/Conv1_down/LeakyReLU/IdentityN',
                'D/32x32/Conv0/LeakyReLU/IdentityN',
                'D/32x32/Conv1_down/LeakyReLU/IdentityN',
                'D/16x16/Conv0/LeakyReLU/IdentityN',
                'D/16x16/Conv1_down/LeakyReLU/IdentityN',
                'D/8x8/Conv0/LeakyReLU/IdentityN',
                'D/8x8/Conv1_down/LeakyReLU/IdentityN',
                'D/4x4/Conv/LeakyReLU/IdentityN'
        ] 
    

    def build_perceptual_model_vgg(self, generated_image_tensor):
        vgg16 = VGG16(include_top=False, input_shape=(self.img_size, self.img_size, 3))
        #Image2StyleGAN uses 1,2,8,12 layers from VGG
        self.perceptual_models = [Model(vgg16.input, vgg16.layers[v].output) for v in self.perceptual_loss_layers.values()]
        generated_image = preprocess_input(tf.image.resize_images(generated_image_tensor,
                                                                  (self.img_size, self.img_size), method=1))
        generated_img_features = [model(generated_image) for model in self.perceptual_models]
        self.input_image = tf.get_variable('input_image', 
                            shape=generated_image.shape, 
                            dtype='float32', 
                            initializer=tf.initializers.zeros())
        self.ref_img_features = [
            tf.get_variable('ref_img_features_%d' % index, 
                            shape=generated_image_feature.shape, 
                            dtype='float32', 
                            initializer=tf.initializers.zeros()
                           ) for index, generated_image_feature in enumerate(generated_img_features)]
        
        # Perceptual loss from Image2StyleGAN
        #self.loss = tf.reduce_sum([
        #            tf.math.sqrt(
        #                tf.losses.mean_squared_error(
        #                self.ref_img_features[i], 
        #                generated_img_features[i])
        #            )
        #    for i in range(len(generated_img_features))])
        ## L2 loss between generated image and input image
        #self.loss += tf.sqrt(tf.losses.mean_squared_error(generated_image, self.input_image))
        # Use SSIM as the loss function
        self.loss =  - tf.image.ssim(generated_image, self.input_image, 256)
        # Use MS-SSIM as the loss function
        # self.loss = tf.image.ssim_multiscale(generated_image, self.input_image, 256)



    def build_perceptual_model_discriminator(self, generated_image_tensor, num_layers_to_use=4):
        self.loss = 0
        self.ref_img_features = []
        for index, conv_variable_name in enumerate(self.discriminator_conv_layer_variable_names[0:num_layers_to_use]):
            # TODO: Check the correct method for getting variable.
            conv_variable = tf.get_default_graph().get_variable_by_name(conv_variable_name)
            variable_shape = conv_variable.shape, 
            self.ref_img_features.append(
                    tf.get_variable('ref_img_features_%d' % index, 
                            shape=variable_shape, 
                            dtype='float32', 
                            initializer=tf.initializers.zeros()))
            self.loss += tf.sqrt(
                    tf.losses.mean_squared_error(
                        conv_variable,
                        self.ref_image_features[index]
                    ) / (variable_shape[0] * variable_shape[1])
            )


    def set_reference_images_vgg(self, images_list):
        assert(len(images_list) != 0 and len(images_list) <= self.batch_size)
        loaded_image = load_images(images_list, self.img_size)
        image_features = [model.predict_on_batch(loaded_image) for model in self.perceptual_models]
        for index, image_feature in enumerate(image_features):
            self.sess.run(tf.assign(self.ref_img_features[index], image_feature))
        self.sess.run(tf.assign(self.input_image, loaded_image))
    
    def set_discriminator_input_image():


    def set_reference_images_discriminator(self, images_list):
        assert(len(images_list) != 0 and len(images_list) <= self.batch_size)
        assert(self.batch_size == 1)
        input_expr = [tf.get_default_graph().get_tensor_by_name(name) for name in ['D/images_in:0']]
        tf.get_default_session().run(expressions, dict(zip(in_expr, mb_in)))

        loaded_image = load_images(images_list, self.img_size)
        image_features = [model.predict_on_batch(loaded_image) for model in self.perceptual_models]
        for index, image_feature in enumerate(image_features):
            self.sess.run(tf.assign(self.ref_img_features[index], image_feature))
        self.sess.run(tf.assign(self.input_image, loaded_image))

    def optimize(self, vars_to_optimize, iterations=500, learning_rate=1.):
        vars_to_optimize = vars_to_optimize if isinstance(vars_to_optimize, list) else [vars_to_optimize]
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        min_op = optimizer.minimize(self.loss, var_list=[vars_to_optimize])
        self.sess.run(tf.variables_initializer(optimizer.variables()))
        for _ in range(iterations):
            _, loss = self.sess.run([min_op, self.loss])
            print(loss)
            yield loss

