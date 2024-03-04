# AIMI_Lab1
Detect pneumonia from chest X-Ray images

## Explanation of code
val_acc_list is actually for testing data. We can see every thing about validation to testing.

### batch size
If the size is bigger, you will need less steps to finish 1 epoch. But at the same time, it also costs more RAM.

### plot_accuracy and plot_f1_score
Use matplotlib to plot and store the plot in png file

### plot_confusion_matrix
Use seaborn heatmap to generate the graph

### tran_set
Use serveral functions in transforms to augment data

### model
Use pre-trained resnet50 in torchvision module
