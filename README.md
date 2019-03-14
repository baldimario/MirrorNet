
# MirrorNet
Framework able to build neural network without training from different decorrelant transformations.
It can work with the PCA for dimensionality reduction.
Invertibility of the extractor it's mathematically guaranteed so the data transformation doesn't present any information loss.
The filters can be static, random or data dependent as the problem requires.

---
Import MirrorNet class
    from MirrorNet import MirrorNet 

---
Load a model

    modelexists = MirrorNet.load(file_name)
Return False if file_name not exists

---

Save the model in file_name

    MirrorNet.save(file_name)

---

Build the feature extractor

    MirrorNet.build_extractor(data, layers=n_layers, percentage=pcad, mode=mode)

data it's the data provided as a numpy array
percentage it's the amount of energy discarded from pca analysis (0 = full energy)
mode it's the weights type it can be:
  - '**pca**' data dependent filters build on autocorrelation [decorrelant]
  - '**hadamard**' static filters build from walsh hadamard transform [decorrelant]
  - '**gauss**' random filters on a normal distribution [decorrelant]

---

Compute the cross correlation matrix for the last layer for the optimal classificator

    MirrorNetbuild_classifier(data, onehot)

---

Show the MirrorNet summary

    MirrorNet.summary()

---

Compute the inference and provide predictions as a result

    predictions = MirrorNet.classify(data)
