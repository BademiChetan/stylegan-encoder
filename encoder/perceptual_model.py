import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
import keras.backend as K


def load_images(images_list, img_size):
    loaded_images = list()
    for img_path in images_list:
        img = np.array(image.load_img(img_path, target_size=(img_size, img_size)))
        # img = img.transpose(2,0,1)
        img = np.expand_dims(img, 0)
        loaded_images.append(img)
    return np.vstack(loaded_images)


class PerceptualModel:
    def __init__(self, img_size, layer=9, batch_size=1, sess=None, use_discriminator=False, num_layers_to_use=4):
        self.sess = tf.get_default_session() if sess is None else sess
        K.set_session(self.sess)
        self.img_size = img_size
        self.layer = layer
        self.batch_size = batch_size
        # L2 loss only
        # self.perceptual_loss_layers = {}
        # Loss function proposed by Image2StyleGAN
        self.perceptual_loss_layers = {'conv1_1': 1, 'conv1_2': 2, 'conv3_2': 8, 'conv4_2': 12}
        # Loss function by stylegan-encoder
        # self.perceptual_loss_layers = {'conv4_2': 9}
        self.perceptual_model = None
        self.ref_img_features = None
        self.loss = None
        self.use_discriminator = use_discriminator
        self.num_layers_to_use= num_layers_to_use
        self.discriminator_conv_layer_variable_names = [
                '/1024x1024/Conv0/LeakyReLU/IdentityN',
                '/1024x1024/Conv1_down/LeakyReLU/IdentityN',
                '/512x512/Conv0/LeakyReLU/IdentityN',
                '/512x512/Conv1_down/LeakyReLU/IdentityN',
                '/256x256/Conv0/LeakyReLU/IdentityN',
                '/256x256/Conv1_down/LeakyReLU/IdentityN',
                '/128x128/Conv0/LeakyReLU/IdentityN',
                '/128x128/Conv1_down/LeakyReLU/IdentityN',
                '/64x64/Conv0/LeakyReLU/IdentityN',
                '/64x64/Conv1_down/LeakyReLU/IdentityN',
                '/32x32/Conv0/LeakyReLU/IdentityN',
                '/32x32/Conv1_down/LeakyReLU/IdentityN',
                '/16x16/Conv0/LeakyReLU/IdentityN',
                '/16x16/Conv1_down/LeakyReLU/IdentityN',
                '/8x8/Conv0/LeakyReLU/IdentityN',
                '/8x8/Conv1_down/LeakyReLU/IdentityN',
                '/4x4/Conv/LeakyReLU/IdentityN'
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
        self.loss = tf.reduce_sum([
                    tf.math.sqrt(
                        tf.losses.mean_squared_error(
                        self.ref_img_features[i], 
                        generated_img_features[i])
                    )
            for i in range(len(generated_img_features))])
        # L2 loss between generated image and input image
        # self.loss += tf.sqrt(tf.losses.mean_squared_error(generated_image, self.input_image))
        # Use SSIM as the loss function
        # self.loss =  - tf.image.ssim(generated_image, self.input_image, 256)
        # Use MS-SSIM as the loss function
        # self.loss = tf.image.ssim_multiscale(generated_image, self.input_image, 256)



    def build_perceptual_model_using_discriminator(self):
        self.loss = 0
        self.ref_img_features = []
        for index, conv_variable_name in enumerate(self.discriminator_conv_layer_variable_names[0:self.num_layers_to_use]):
            # TODO: Check the correct method for getting variable.
            conv_variable = tf.get_default_graph().get_tensor_by_name('D' + conv_variable_name + ":0")
            variable_shape = conv_variable.shape
            print("variable shape = ")
            print(variable_shape)
            self.ref_img_features.append(
                    tf.get_variable('ref_img_features_%d' % index,
                            shape=(self.batch_size, variable_shape[1], variable_shape[2], variable_shape[3]), 
                            dtype='float32',
                            initializer=tf.initializers.zeros()))
            self.loss += tf.sqrt(tf.losses.mean_squared_error(conv_variable, self.ref_img_features[index]))


    def set_reference_images_vgg(self, images_list):
        assert(len(images_list) != 0 and len(images_list) <= self.batch_size)
        loaded_images = load_images(images_list, self.img_size)
        preprocessed_images = preprocess_input(loaded_images)
        image_features = [model.predict_on_batch(preprocessed_images) for model in self.perceptual_models]
        for index, image_feature in enumerate(image_features):
            self.sess.run(tf.assign(self.ref_img_features[index], image_feature))
        self.sess.run(tf.assign(self.input_image, preprocessed_images))
        return loaded_images
    
    def set_reference_images_discriminator(self, images_list):
        assert(len(images_list) != 0 and len(images_list) <= self.batch_size)
        assert(self.batch_size == 1)
        loaded_images = load_images(images_list, 1024)
        input_expr = tf.get_default_graph().get_tensor_by_name('D/images_in:0')
        output_operations = [ 'D' + x + ":0" for x in self.discriminator_conv_layer_variable_names[0:self.num_layers_to_use]]
        print("output_operations = ")
        print(output_operations)
        print(input_expr)
        print(zip(input_expr, loaded_images))
        #print(dict(zip(input_expr, loaded_images)))
        image_features = self.sess.run(output_operations, feed_dict = {input_expr: loaded_images})
        for x in image_features:
            print(x.shape)
        for index, image_feature in enumerate(image_features):
            self.sess.run(tf.assign(self.ref_img_features[index], image_feature))

    def optimize(self, vars_to_optimize, iterations=500, learning_rate=1., generated_image=None):
        vars_to_optimize = vars_to_optimize if isinstance(vars_to_optimize, list) else [vars_to_optimize]
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        min_op = optimizer.minimize(self.loss, var_list=[vars_to_optimize])
        self.sess.run(tf.variables_initializer(optimizer.variables()))
        for _ in range(iterations):
            _, loss = self.sess.run([min_op, self.loss])
            print(loss)
            yield loss

