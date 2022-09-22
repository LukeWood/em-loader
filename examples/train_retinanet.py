"""
Title: Train a RetinaNet to Detect ElectroMagnetic Signals
Author: [lukewood](https://lukewood.xyz), Kevin Anderson, Peter Gerstoft
Date created: 2022/08/16
Last modified: 2022/08/16
Description:
"""

"""
## Overview
"""

import sys

import keras_cv
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from absl import flags
from keras_cv import bounding_box
from tensorflow import keras
from tensorflow.keras import callbacks as callbacks_lib
from tensorflow.keras import optimizers

import em_loader
import wandb

flags.DEFINE_integer("batch_size", 8, "Training and eval batch size.")
flags.DEFINE_integer("epochs", 1, "Number of training epochs.")
flags.DEFINE_string("wandb_entity", "scisrs", "wandb entity to use.")
flags.DEFINE_string("experiment_name", None, "wandb run name to use.")
flags.DEFINE_string("checkpoint_path", None, "checkpoint path to use.")
flags.DEFINE_string("artifacts_dir", None, "artifact directory to use.")
FLAGS = flags.FLAGS

FLAGS(sys.argv)

if FLAGS.wandb_entity and FLAGS.experiment_name:
    wandb.init(
        project="scisrs",
        entity=FLAGS.wandb_entity,
        name=FLAGS.experiment_name,
    )

"""
## Data loading
"""


"""
Great!  Our data is now loaded into the format
`{"images": images, "bounding_boxes": bounding_boxes}`.  This format is supported in all
KerasCV preprocessing components.

Lets load some data and verify that our data looks as we expect it to.
"""

dataset, dataset_info = em_loader.load(
    split="train", bounding_box_format="xywh", batch_size=9,
    version=2,
)


def visualize_dataset(dataset, bounding_box_format):
    color = tf.constant(((255.0, 0, 0),))
    plt.figure(figsize=(7, 7))
    iterator = iter(dataset)
    for i in range(9):
        example = next(iterator)
        images, boxes = example["images"], example["bounding_boxes"]
        boxes = keras_cv.bounding_box.convert_format(
            boxes, source=bounding_box_format, target="rel_yxyx", images=images
        )
        boxes = boxes.to_tensor(default_value=-1)
        plotted_images = tf.image.draw_bounding_boxes(images, boxes[..., :4], color)
        plt.subplot(9 // 3, 9 // 3, i + 1)
        plt.imshow(plotted_images[0].numpy().astype("uint8"))
        plt.axis("off")
    plt.show()


visualize_dataset(dataset, bounding_box_format="xywh")

"""
Looks like everything is structured as expected.  Now we can move on to constructing our
data augmentation pipeline.
"""

"""
## Data augmentation
"""

# train_ds is batched as a (images, bounding_boxes) tuple
# bounding_boxes are ragged
train_ds, train_dataset_info = em_loader.load(
    bounding_box_format="xywh", split="train", batch_size=FLAGS.batch_size,
    version=2
)
val_ds, val_dataset_info = em_loader.load(
    bounding_box_format="xywh", split="val", batch_size=FLAGS.batch_size,
    version=2
)


def resize_data(inputs, size):
    image = inputs["images"]
    bboxes = inputs["bounding_boxes"]

    # Convert bounding boxes to relative format
    bboxes = bounding_box.convert_format(
        bboxes, source="xywh", target="rel_yxyx", images=image
    )

    # Resize image
    image = tf.image.resize(image, size)

    # Convert bounding boxes back to original format
    bboxes = bounding_box.convert_format(
        bboxes, source="rel_yxyx", target="xywh", images=image
    )

    inputs["images"] = image
    inputs["bounding_boxes"] = bboxes

    return inputs


size = [512, 512]
train_ds = train_ds.map(
    lambda x: resize_data(x, size), num_parallel_calls=tf.data.AUTOTUNE
)
val_ds = val_ds.map(lambda x: resize_data(x, size), num_parallel_calls=tf.data.AUTOTUNE)

visualize_dataset(train_ds, bounding_box_format="xywh")


def unpackage_dict(inputs):
    return inputs["images"], inputs["bounding_boxes"]


train_ds = train_ds.map(unpackage_dict, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.map(unpackage_dict, num_parallel_calls=tf.data.AUTOTUNE)

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

"""
Our data pipeline is now complete.  We can now move on to model creation and training.
"""

"""
## Model creation

We'll use the KerasCV API to construct a RetinaNet model.  In this tutorial we use
a pretrained ResNet50 backbone using weights.  In order to perform fine-tuning, we
freeze the backbone before training.  When `include_rescaling=True` is set, inputs to
the model are expected to be in the range `[0, 255]`.
"""

model = keras_cv.models.RetinaNet(
    classes=1,
    bounding_box_format="xywh",
    backbone="resnet50",
    backbone_weights=None,
    include_rescaling=True,
)

optimizer = tf.optimizers.SGD(global_clipnorm=10.0)
metrics = [
    keras_cv.metrics.COCOMeanAveragePrecision(
        class_ids=range(1),
        bounding_box_format="xywh",
        name="MaP",
    ),
    keras_cv.metrics.COCORecall(
        class_ids=range(1),
        bounding_box_format="xywh",
        max_detections=100,
        name="Recall",
    ),
]

model.compile(
    classification_loss=keras_cv.losses.FocalLoss(from_logits=True, reduction="none"),
    box_loss=keras_cv.losses.SmoothL1Loss(l1_cutoff=1.0, reduction="none"),
    optimizer=optimizer,
    metrics=metrics,
)

"""
All that is left to do is construct some callbacks:
"""

callbacks = [
    callbacks_lib.TensorBoard(log_dir="logs"),
    callbacks_lib.EarlyStopping(patience=20),
    callbacks_lib.ReduceLROnPlateau(patience=5),
]

if FLAGS.checkpoint_path is not None:
    callbacks += [
        keras.callbacks.ModelCheckpoint(FLAGS.checkpoint_path, save_weights_only=True)
    ]
if FLAGS.wandb_entity:
    callbacks += [
        wandb.keras.WandbCallback(save_model=False),
    ]

"""
And run `model.fit()`!
"""

model.fit(
    train_ds,
    validation_data=val_ds.take(20),
    epochs=FLAGS.epochs,
    callbacks=callbacks,
)

model.load_weights(FLAGS.checkpoint_path)
metrics = model.evaluate(val_ds, return_dict=True)
print("FINAL METRICS:", metrics)


def visualize_detections(model):
    train_ds, val_dataset_info = keras_cv.datasets.pascal_voc.load(
        bounding_box_format="xywh", split="train", batch_size=9
    )
    train_ds = train_ds.map(dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)
    images, labels = next(iter(train_ds.take(1)))
    predictions = model.predict(images)
    color = tf.constant(((255.0, 0, 0),))
    plt.figure(figsize=(10, 10))
    predictions = keras_cv.bounding_box.convert_format(
        predictions, source="xywh", target="rel_yxyx", images=images
    )
    predictions = predictions.to_tensor(default_value=-1)
    plotted_images = tf.image.draw_bounding_boxes(images, predictions[..., :4], color)
    for i in range(9):
        plt.subplot(9 // 3, 9 // 3, i + 1)
        plt.imshow(plotted_images[i].numpy().astype("uint8"))
        plt.axis("off")
    plt.savefig(f"{FLAGS.artifacts_dir}/demo.png")

visualize_detections(model)
print("FINAL METRICS:", metrics)

with open(f"{FLAGS.artifacts_dir}/MaP.txt", "w") as f:
    f.write(metrics["MaP"])

with open(f"{FLAGS.artifacts_dir}/Recall.txt", "w") as f:
    f.write(metrics["Recall"])
