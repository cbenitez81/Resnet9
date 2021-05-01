# Resnet9

## Dependencies
- pytorch = 1.8
- cuda 10.2
- torchvision
- matplotlib

## Training:

Uncompress the folder in any location and run the following command

`python test_geo.py --train PATH_TO_TRAINIG_DATA`

The training data folder must have 2 folders called set_test and set_train

## Testing

For testing you can run the following command

`python test_geo.py --test PATH_TO_TEST_IMAGE --k 10`

where k is the number of top k images similar to the testing one



### References

https://github.com/matthias-wright/cifar10-resnet
https://github.com/lambdal/cifar10-fast
https://medium.com/swlh/natural-image-classification-using-resnet9-model-6f9dc924cd6d
https://github.com/digantamisra98/Mish

