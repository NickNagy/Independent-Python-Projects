import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
#import upResNet
import upResNet_v2
from image_util import ImageDataProvider as generator
from skimage.transform import resize
from scipy.misc import imresize

training_path = 'D:\\upResNet\\256x256\\training'
validation_path = training_path #'D:\\flickr-image-dataset\\flickr30k_images\\256x256\\smallset\\Validation'
testing_path = training_path #'D:\\flickr-image-dataset\\flickr30k_images\\256x256\\smallset\\Testing'

def compare_accuracies(net, generator, model_path, channels, superior_save_path, inferior_save_path):
    pred_superior_counter = 0
    total_files = generator._get_number_of_files()
    for i in range(0, total_files):
        test_x, test_y, test_w = generator(1)
        prediction = net.predict(model_path + "/model.ckpt", test_x)
        x_resize = resize(test_x[0], (prediction.shape[1],prediction.shape[2]))
        x_resize_acc = np.mean(np.equal(x_resize.astype(np.int32), test_y[0].astype(np.int32))).astype(np.float32)
        pred_acc = np.mean(np.equal(prediction[0].astype(np.int32), test_y[0].astype(np.int32))).astype(np.float32)
        fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
        if channels == 3:
            x_disp = x_resize
            pred_disp = prediction[0]
        else:
            plt.gray()
            x_disp = x_resize[:, ..., 0]
            pred_disp = prediction[0, ..., 0]
        ax[0].imshow(x_disp.astype('uint8'), aspect="auto")
        ax[0].set_title("X Resize:\nAccuracy: {:.4f}".format(x_resize_acc))
        ax[1].imshow(pred_disp.astype('uint8'), aspect="auto")
        ax[1].set_title("Prediction:\nAccuracy: {:.4f}".format(pred_acc))
        if pred_acc > x_resize_acc:
            plt.savefig(superior_save_path + str(i) + ".jpg")
            pred_superior_counter += 1
        else:
            plt.savefig(inferior_save_path + str(i) + ".jpg")
        plt.close()
    pred_superior_ratio = pred_superior_counter / total_files
    print(str(int(pred_superior_ratio * 100)) + "% of predictions were more accurate than the resized x images.")

def test_single_img_file(img_path, net, model_path, channels, start_size=128):
    img = Image.open(img_path)
    if channels == 1:
        img = img.convert('L')
    img_x = imresize(img, (start_size,start_size,channels))
    prediction = net.predict(model_path + "/model.ckpt", img_x.reshape(1, start_size, start_size, channels))
    if channels == 1:
        prediction = prediction[0,...,0]
    else:
        prediction = prediction[0]
    plt.imshow(prediction)
    plt.show()
    pred_img = Image.fromarray(prediction.astype('uint8'))
    pred_img.save(img_path[0:-4] + "_prediction.jpg")

restore = True
padding = True
test_after = False
channels = 1
in_shape = 128
resolution = 2
batch_size = 4
validation_batch_size = batch_size
epochs = 50
learning_rate = 0.001
layers_per_transpose = 1
features_root = 64

weight_type = "_sobel"
weight_suffix = weight_type + ".npy"

summary_str = str(resolution) + "res_" + str(layers_per_transpose) + "layers_" + str(channels) + "chan_" + \
              str(features_root) + "feat" + weight_type

output_path = "D:\\upResNet\\temp\\" + summary_str
restore_path = output_path

if channels == 3:  # RGB
    data_suffix = "_x.npy"
    mask_suffix = "_y.npy"
else:  # grayscale
    data_suffix = "_x_gray.npy"
    mask_suffix = "_y_gray.npy"

training_generator = generator(search_path=training_path + "/*npy", data_suffix=data_suffix, mask_suffix=mask_suffix,
                               weight_suffix=weight_suffix, shuffle_data=False, n_class=channels)
validation_generator = generator(search_path=validation_path + "/*npy", data_suffix=data_suffix,
                                 mask_suffix=mask_suffix,
                                 weight_suffix=weight_suffix, shuffle_data=False, n_class=channels)
testing_generator = generator(search_path=testing_path + "/*npy", data_suffix=data_suffix, mask_suffix=mask_suffix,
                              weight_suffix='_w.npy', shuffle_data=False, n_class=channels)

training_iters = int(training_generator._get_number_of_files() / batch_size) + \
                 (training_generator._get_number_of_files() % batch_size > 0)
total_validation_data = validation_generator._get_number_of_files()

net = upResNet(padding=padding, in_shape=in_shape, channels=channels, resolution=resolution,
               layers_per_transpose=layers_per_transpose, features_root=features_root)

opt_kwargs = dict(learning_rate=learning_rate)

trainer = Trainer(net=net, batch_size=batch_size, validation_batch_size=validation_batch_size, opt_kwargs=opt_kwargs)
trainer.train(training_data_provider=training_generator, validation_data_provider=validation_generator,
              testing_data_provider=testing_generator, restore_path=restore_path, output_path=output_path,
              total_validation_data=total_validation_data, training_iters=training_iters, epochs=epochs,
              restore=restore, test_after=test_after, prediction_path=output_path + '\\prediction')

#test_single_img_file(img_path="C:\\Users\\Nick Nagy\\Desktop\\Samples\\t.png", net=net, model_path=output_path,
#                     channels=channels)

#compare_path = validation_path + "\\Accuracy Comparisons\\" + summary_str + "\\"
#compare_accuracies(net=net, generator=validation_generator, model_path=output_path, channels=channels,
#                   superior_save_path=compare_path + "Superior\\", inferior_save_path=compare_path + "Inferior\\")
