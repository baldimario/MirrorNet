from MirrorNet import MirrorNet
import numpy as np
import matplotlib.pyplot as plt

def main():
    n_layers = 6
    energy_percentage_discarded = 0
    mode = 'pca'
    
    data = np.random.normal(loc=0, scale=1, size=[1000, 2])
    labels = np.zeros(data.shape[0])
    labels[data[:, 1] < 0] = 1
    onehot = np.zeros((labels.shape[0], 2))
    onehot[np.arange(labels.shape[0]), labels.astype('uint8')] = 1
    
    
    if not MirrorNet.load('model.net'):
        MirrorNet.build_extractor(data, layers=n_layers, percentage=energy_percentage_discarded, mode=mode).build_classifier(data, onehot)
        MirrorNet.save('model.net')
    
    predictions = MirrorNet.classify(data)
    
    MirrorNet.summary()
    
    predictions = np.argmax(predictions, axis=-1)
    wrongs = np.sum((predictions != labels).astype('uint8'))
    percentage = 100 * wrongs / labels.shape[0]

    print('Error (with {}% Energy Loss): {}/{} -> {}%'.format(100 * energy_percentage_discarded, int(wrongs), labels.shape[0], percentage))

    print('Done.')
    
if __name__ == '__main__':
    main()
