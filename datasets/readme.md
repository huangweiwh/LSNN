# Dataset

## Cifar10/100

- Just download the dataset from official website.

## Mini-ImageNet

The miniImageNet contains a total of 60,000 color images in 100 categories, with 600 samples in each category. However, its train/val/test subset doesn't has all 100 categories of images, so it needs to redivide this dataset like the form below:

- data_miniimagenet

  - train

    - categories 1

      - 1.jpg

      - 2.jpg

      - ......

    - categories 2

    - ......

    - categories 100

  - val

    - categories 1

      - 1.jpg

      - 2.jpg

      - ......

    - categories 2

    - ......

    - categories 100

  - test

    - categories 1

      - 1.jpg

      - 2.jpg

      - ......

    - categories 2

    - ......

    - categories 100

    -

## ImageNet

When you download this dataset, turn it into the form below:

- imagenet

  - train

    - categories 1

      - 1.jpg

      - 2.jpg

      - ......

    - categories 2

    - ......

    - categories 1000

  - val

    - categories 1

      - 1.jpg

      - 2.jpg

      - ......

    - categories 2

    - ......

    - categories 1000
