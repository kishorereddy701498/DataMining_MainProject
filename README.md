# DataMining_MainProject

The motication behond this project can be found https://github.com/kishorereddy701498/DataMining_MainProject/blob/main/Weather%20Classification%20Proposal.docx
This Weather classification algorithm and the data set used for this project can be found here https://drive.google.com/file/d/14h1zDIBkBDuLPfkO8EjsK8yQQD5n5oxJ/view?usp=sharing
This contains different set od images that contains different weather conditions
By using the abvoe dataset we are going to develop a Weather Classifier to find the weather based on the images.




Here in this project we have developed 2 convolution models with different hyper parameters
Model 1 Summary :

Model: "sequential_4"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 rescaling_6 (Rescaling)     (None, 180, 180, 3)       0         
                                                                 
 conv2d_13 (Conv2D)          (None, 180, 180, 16)      448       
                                                                 
 max_pooling2d_13 (MaxPoolin  (None, 90, 90, 16)       0         
 g2D)                                                            
                                                                 
 conv2d_14 (Conv2D)          (None, 90, 90, 32)        4640      
                                                                 
 max_pooling2d_14 (MaxPoolin  (None, 45, 45, 32)       0         
 g2D)                                                            
                                                                 
 conv2d_15 (Conv2D)          (None, 45, 45, 64)        18496     
                                                                 
 max_pooling2d_15 (MaxPoolin  (None, 22, 22, 64)       0         
 g2D)                                                            
                                                                 
 flatten_4 (Flatten)         (None, 30976)             0         
                                                                 
 dense_8 (Dense)             (None, 128)               3965056   
                                                                 
 dense_9 (Dense)             (None, 4)                 516       
                                                                 
=================================================================
Total params: 3,989,156
Trainable params: 3,989,156
Non-trainable params: 0
______________________


Model 2 Summary :

num_classes = len(class_names)

model = Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.1),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()
epochs=10
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)


