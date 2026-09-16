# Literature Survey

## Scope

This survey connects the project design to primary research papers and official technical documentation. It separates established evidence from project-specific choices. The cited work supports the dataset, model family, optimization, validation, input pipeline, augmentation, callbacks, and interpretability decisions; it does not claim that the present implementation reproduces the best published GTSRB score.

## Verified sources and their relevance

| Source | Verified contribution | Why it matters here | Where it is used |
|---|---|---|---|
| Stallkamp et al., IJCNN 2011, *The German Traffic Sign Recognition Benchmark* | Introduces GTSRB as a multi-category, real-world traffic-sign classification benchmark with variation in illumination, weather, distance, occlusion, and rotation. | Establishes the problem definition and explains why a 43-class cropped-image benchmark is useful. | Dataset selection, problem statement, EDA discussion, official test-set evaluation. |
| Sandler et al., CVPR 2018, *MobileNetV2: Inverted Residuals and Linear Bottlenecks* | Describes an efficient mobile architecture based on inverted residuals, linear bottlenecks, and depthwise convolutions. | Motivates MobileNetV2 as a lightweight comparison model suitable for deployment-oriented experiments. | `build_mobilenet_v2`, transfer-learning comparison, future mobile deployment discussion. |
| Kingma and Ba, 2015, *Adam: A Method for Stochastic Optimization* | Presents Adam as an adaptive first-order optimizer using estimates of gradient moments, with low memory requirements and practical performance. | Supports the optimizer choice for a first reproducible baseline. | `train.py` and the Colab notebook compile both models with Adam; it is not claimed to be optimal. |
| Selvaraju et al., ICCV 2017, *Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization* | Uses gradients flowing into a final convolutional layer to produce a class-discriminative coarse localization map. | Provides a principled way to inspect whether predictions rely on the sign rather than irrelevant background. | `gradcam.py`, `generate_gradcam.py`, local demo, Colab Grad-CAM cell, presenter explanation. |
| scikit-learn, `GroupShuffleSplit` documentation | Defines randomized train/test splitting by groups and supports a reproducible random state. | Supports the leakage-control decision when related frames or sequences should stay together. | `split_train_validation` and the standalone Colab grouped validation split. |
| TensorFlow, `tf.data.Dataset` and data-performance documentation | Describes streaming dataset transformations and recommends parallel mapping and prefetching to overlap input preparation with model execution. | Supports a memory-conscious full-dataset pipeline for Colab GPU training. | `make_dataset` / `make_ds`, parallel decode, batching, and `prefetch(AUTOTUNE)`. |
| Keras preprocessing and callback documentation | Documents training-only image augmentation and callbacks such as EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint. | Supports moderate augmentation, stopping rules, adaptive learning-rate control, and reproducible best-model saving. | `models.py`, `train.py`, and the Colab training cell. |

## Synthesis and project decisions

The GTSRB paper establishes that sign images vary in ways that are relevant to recognition, so the project retains RGB information, standardizes image size, and augments only the training data. The custom CNN is included as a transparent from-scratch baseline. MobileNetV2 is included because its efficient architecture offers a meaningful comparison, but the current transfer-learning configuration freezes its backbone; therefore its Colab result should be interpreted as a baseline rather than as evidence that MobileNetV2 is intrinsically worse.

Adam, learning-rate reduction, early stopping, and checkpointing form a practical training protocol. The `tf.data` design keeps image files streamed instead of loading the complete dataset into a single in-memory tensor. Group-based splitting is a project safeguard against overly optimistic validation when related images share a sequence-like identifier. The official test set remains untouched until final evaluation.

Grad-CAM is used for qualitative inspection only. A heatmap can reveal whether a model is looking at a plausible region, but it is not a proof of causal reasoning, correctness, or safety. Similarly, the optional YOLO scripts are an engineering scaffold: the GTSRB classification archive does not provide the full-scene bounding-box annotations required to train and evaluate a detector.

## References

[1] J. Stallkamp, M. Schlipsing, J. Salmen, and C. Igel, “The German Traffic Sign Recognition Benchmark: A multi-class classification competition,” IJCNN, 2011. [Primary paper](https://www.ini.rub.de/upload/file/1470692848_f03494010c16c36bab9e/StallkampEtAl_GTSRB_IJCNN2011.pdf)

[2] M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L.-C. Chen, “MobileNetV2: Inverted Residuals and Linear Bottlenecks,” CVPR, 2018. [Paper](https://arxiv.org/abs/1801.04381)

[3] D. P. Kingma and J. Ba, “Adam: A Method for Stochastic Optimization,” ICLR, 2015. [Paper](https://arxiv.org/abs/1412.6980)

[4] R. R. Selvaraju et al., “Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization,” ICCV, 2017. [Open-access paper](https://openaccess.thecvf.com/content_ICCV_2017/papers/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.pdf)

[5] scikit-learn developers, “GroupShuffleSplit.” [Official documentation](https://scikit-learn.org/1.5/modules/generated/sklearn.model_selection.GroupShuffleSplit.html)

[6] TensorFlow developers, “tf.data.Dataset” and “Better performance with the tf.data API.” [Official API](https://www.tensorflow.org/api_docs/python/tf/data/Dataset), [performance guide](https://www.tensorflow.org/guide/data_performance)

[7] Keras developers, “Preprocessing layers” and “Callbacks API.” [Preprocessing documentation](https://keras.io/api/layers/preprocessing_layers/), [callbacks documentation](https://keras.io/api/callbacks/)
