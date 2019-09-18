# Binary-Classifier
It is a binary classifier which works on keras module in python

This script uses data from https://drive.google.com/file/d/1dbcWabr3Xrr4JvuG0VxTiweGzHn-YYvW/view 
The Zip file consists of images of cars and planes. They have been separated into 2 folders of training and test. Further, they have been labeled as 'cars' and 'planes'.

Using this script I was able to achieve a validation accuracy of above 85% consistently after the 15th epoch.

From looking at the graph we can easily conclude that training the model for more number of epochs will result in better accuracy.
As the number of epochs will increase, the accuracy will continue to increase. But the validation accuracy will increase till a particular epoch and then start decreasing because the model will start overfitting after that.

For now, from the plot we can say that increasing the number of epochs will help.

In the next update I intend to find the exact epoch after which this happens
