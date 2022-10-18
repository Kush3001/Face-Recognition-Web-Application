from keras.layers import Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from glob import glob
import matplotlib.pyplot as plt

# Initializing the Image Size:
IMAGE_SIZE = [224, 224]
# Initializing the training and test path:
train_path = 'Face-Recognition-Web-Application\DataSets\Train'
test_path = 'Face-Recognition-Web-Application\DataSets\Test'

vgg = VGG16(input_shape=IMAGE_SIZE+[3], weights='imagenet', include_top=False)
for layer in vgg.layers:
    layer.trainable = False

folders = glob('Face-Recognition-Web-Application/DataSets/*')

x = Flatten()(vgg.output)
prediction = Dense(len(folders), activation='softmax')(x)
model = Model(inputs=vgg.input, outputs=prediction)
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('Face-Recognition-Web-Application\DataSets\Train',
                                                 target_size=(224, 224),
                                                 batch_size=32,
                                                 class_mode='categorical')

test_set = test_datagen.flow_from_directory('Face-Recognition-Web-Application\DataSets\Test',
                                            target_size=(224, 224),
                                            batch_size=32,
                                            class_mode='categorical')

print(len(training_set))
print(len(test_set))
# r = model.fit(training_set, validation_data=test_set, epochs=5,
#               steps_per_epoch=len(training_set), validation_steps=len(test_set))

# # Loss:
# plt.plot(r.history['loss'], label='train loss')
# plt.plot(r.history['val_loss'], label='val loss')
# plt.legend()
# plt.show()
# plt.savefig('LassVal_loss')

# # Accuracies:
# plt.plot(r.history['acc'], label='train acc')
# plt.plot(r.history['val_acc'], label='val acc')
# plt.legend()
# plt.show()
# plt.savefig('AccVal_acc')

# model.save('FaceModel1.h5')
