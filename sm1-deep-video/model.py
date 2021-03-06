from tensorflow.python.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16


def create_model():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.20)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.20)(x)
    x = Dense(256, activation='relu')(x)
    preds = Dense(2, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=preds)

    for i, layer in enumerate(model.layers):
        print(i, layer.name)

    # Fix the weight fo the first 20 layers
    for layer in model.layers[:20]:
        layer.trainable = False

    for layer in model.layers[20:]:
        layer.trainable = True

    return model
