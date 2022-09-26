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
from luketils import artifacts

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

assert FLAGS.checkpoint_path is not None

if FLAGS.wandb_entity and FLAGS.experiment_name:
    wandb.init(
        project="scisrs",
        entity=FLAGS.wandb_entity,
        name=FLAGS.experiment_name,
    )

artifacts_dir = FLAGS.artifacts_dir
if artifacts_dir:
    artifacts.set_base(artifacts_dir)


class_ids = [
    "Source",
]
class_mapping = dict(zip(range(len(class_ids)), class_ids))

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


example = next(iter(dataset))
images, boxes = example["images"], example["bounding_boxes"]
visualization.plot_bounding_box_gallery(
    images,
    value_range=(0, 255),
    bounding_box_format='xywh'',
    y_true=boxes,
    scale=4,
    rows=3,
    cols=3,
    show=True,
    thickness=4,
    font_scale=1,
    class_mapping=class_mapping,
    show=artifacts_dir is None,
    path=artifacts.path('ground-truth.png')
)

"""
Looks like everything is structured as expected.  Now we can move on to constructing our
data pipeline
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

def unpackage_dict(inputs):
    return inputs["images"], inputs["bounding_boxes"]

train_ds = train_ds.map(unpackage_dict, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.map(unpackage_dict, num_parallel_calls=tf.data.AUTOTUNE)

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

"""
Our data pipeline is now complete.  We can now move on to model creation and training.

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
    evaluate_train_time_metrics=True,
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
    keras.callbacks.ModelCheckpoint(FLAGS.checkpoint_path, save_weights_only=True)
]

if FLAGS.wandb_entity:
    callbacks += [
        wandb.keras.WandbCallback(save_model=False),
    ]

"""
And run `model.fit()`!
"""

history = model.fit(
    train_ds,
    steps_per_epochs=1
    # validation_data=val_ds.take(20),
    epochs=FLAGS.epochs,
    callbacks=callbacks,
)

model.load_weights(FLAGS.checkpoint_path)
metrics = model.evaluate(val_ds, return_dict=True)
print("FINAL METRICS:", metrics)

if artifacts_dir is not None:
    for metric in metrics:
        with open('artifacts_dir/metrics_{metric}.txt', 'w') as f:
            f.write(metrics[metric])

def visualize_detections(model, split='train'):
    train_ds, val_dataset_info = em_loader.load(
        bounding_box_format="xywh", split=split, batch_size=9
    )
    train_ds = train_ds.map(dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)
    images, y_true = next(iter(train_ds.take(1)))
    y_pred = model.predict(images)
    visualization.plot_bounding_box_gallery(
        images,
        value_range=(0, 255),
        bounding_box_format=bounding_box_format,
        y_true=y_true,
        y_pred=y_pred,
        scale=4,
        rows=3,
        cols=3,
        show=True,
        thickness=4,
        font_scale=1,
        class_mapping=class_mapping,
        show=artifacts_dir is None,
        path=artifacts.path(f'{split}.png')
    )

visualize_detections(model, split='train')
visualize_detections(model, split='val')
