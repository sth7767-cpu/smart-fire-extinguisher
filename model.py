import keras
from keras import layers
from keras import models
from keras import optimizers
from keras import applications

# -----------------------------
# 설정
# -----------------------------
DATASET_DIR = "dataset"   # good / Bad_low / Bad_high 로 나눠진 폴더
BATCH_SIZE = 16
IMG_SIZE = (224, 224)
EPOCHS = 20
LR = 0.0001

# -----------------------------
# 데이터 로딩
# -----------------------------
train_data = keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_data = keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# 클래스 매핑 확인
print("클래스:", train_data.class_names)
# ['Bad_high', 'Bad_low', 'good'] 이런 식으로 자동 지정됨

# -----------------------------
# 데이터 전처리 + 증강
# -----------------------------
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),
    layers.RandomBrightness(0.15)
])

normalization_layer = layers.Rescaling(1./255)

train_data = train_data.map(lambda x, y: (normalization_layer(data_augmentation(x)), y))
val_data = val_data.map(lambda x, y: (normalization_layer(x), y))


# -----------------------------
# 모델 구성 (MobileNetV2)
# -----------------------------
base_model = applications.MobileNetV2(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3)
)

base_model.trainable = False   # 1단계: feature extractor

inputs = layers.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(3, activation="softmax")(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer=optimizers.Adam(LR),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -----------------------------
# 1단계 학습
# -----------------------------
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# -----------------------------
# Fine-tuning (2단계)
# -----------------------------
base_model.trainable = True

model.compile(
    optimizer=optimizers.Adam(1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history_finetune = model.fit(
    train_data,
    validation_data=val_data,
    epochs=5
)

# -----------------------------
# 모델 저장
# -----------------------------
model.save("pressure_model.keras", save_format="keras")
print("🎯 모델 저장 완료 → pressure_model.keras")