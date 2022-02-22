import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

def make_visualize_loss_accuracy(results, save_path, loss_name='loss', accuracy_name='categorical_accuracy'):
    loss_y = results['history'][loss_name]
    accuracy_y = results['history'][accuracy_name]
    x = results['epoch']
    fig = plt.figure(figsize=(12, 6))
    ax_loss = fig.add_subplot(121)
    ax_loss.set_xlabel('epoch')
    ax_loss.set_ylabel('loss')
    ax_loss.set_title('weighted categorical cross entropy')
    ax_loss.plot(x, loss_y)
    ax_loss.set_position([0.15,0.1,0.3,0.8])

    ax_accuracy = fig.add_subplot(122)
    ax_accuracy.set_xlabel('epoch')
    ax_accuracy.set_ylabel('accuracy')
    ax_accuracy.set_title('categorical accuracy')
    ax_accuracy.plot(x, accuracy_y, color='orange')
    ax_accuracy.set_position([0.55,0.1,0.3,0.8])

    fig.savefig(os.path.join(save_path,'loss_accuracy.png'))


with open('results_real2/results.pkl', 'rb') as f:
    rs = pickle.load(f)
    print(rs)

make_visualize_loss_accuracy(rs,"results_real2")
