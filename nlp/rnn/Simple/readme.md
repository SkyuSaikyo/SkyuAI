Implemented classification of IMDB movie reviews directly using a linear layer after word embedding  
File information:
- basic.py: Contains Dataset and model loading for the IMDB dataset
- train.py: Training script loading the dataset may take some time, please be patient. Each epoch will generate a model.pth file in the current directory, and each new model trained will overwrite the previous one. This training script supports resuming training. If training is interrupted and this script is run again, it will continue training from the last epoch until completion
- test.py: Evaluates the model files generated by train.py
- evaluate.py: Evaluates the network. After running, it will generate a corresponding CSV file. This script takes a long time to run. here, a few typical values have been selected for reference


embedding_dim=8
| **epoch\w** | **16** | **32** | **64** | **128** | **256** |
|-------------|--------|--------|--------|---------|---------|
| **4**       | 56.16  | 58.64  | 59.50  | 67.30   | 66.53   |
| **8**       | 61.12  | 64.71  | 68.34  | 74.79   | 75.96   |
| **12**      | 63.88  | 67.95  | 71.75  | 76.81   | 78.83   |
| **16**      | 65.45  | 69.01  | 72.64  | 77.62   | 79.74   |
| **20**      | 65.87  | 69.42  | 72.86  | 77.73   | 79.79   |
| **24**      | 65.77  | 69.25  | 72.78  | 77.77   | 79.91   |
| **28**      | 65.57  | 69.14  | 72.48  | 77.61   | 79.80   |
| **32**      | 65.39  | 68.93  | 72.25  | 77.27   | 79.72   |
| **36**      | 65.15  | 68.46  | 71.73  | 76.93   | 79.62   |
| **40**      | 64.81  | 68.09  | 71.46  | 76.74   | 79.67   |
| **44**      | 64.57  | 67.61  | 71.21  | 76.76   | 79.60   |
| **48**      | 64.40  | 67.37  | 70.88  | 76.64   | 79.68   |
| **52**      | 64.14  | 66.96  | 70.64  | 76.63   | 79.76   |
| **56**      | 63.82  | 66.66  | 70.49  | 76.52   | 79.79   |
| **60**      | 63.55  | 66.30  | 70.41  | 76.53   | 79.86   |
| **64**      | 63.45  | 66.15  | 70.40  | 76.53   | 79.80   |

embedding_dim=16
| **epoch\w** | **16** | **32** | **64** | **128** | **256** |
|-------------|--------|--------|--------|---------|---------|
| **4**       | 58.30  | 59.80  | 63.75  | 68.99   | 70.02   |
| **8**       | 62.63  | 65.56  | 70.03  | 74.83   | 76.54   |
| **12**      | 64.64  | 67.94  | 71.60  | 75.83   | 77.98   |
| **16**      | 65.37  | 68.42  | 71.74  | 76.11   | 78.41   |
| **20**      | 65.43  | 68.09  | 71.71  | 75.97   | 78.77   |
| **24**      | 65.31  | 67.84  | 71.31  | 75.82   | 78.77   |
| **28**      | 64.81  | 67.43  | 70.98  | 75.68   | 79.00   |
| **32**      | 64.44  | 66.92  | 70.64  | 75.76   | 79.06   |
| **36**      | 64.09  | 66.76  | 70.48  | 75.78   | 79.10   |
| **40**      | 63.60  | 66.52  | 70.37  | 75.82   | 79.25   |
| **44**      | 63.36  | 66.20  | 70.24  | 75.83   | 79.23   |
| **48**      | 63.24  | 65.94  | 70.24  | 75.96   | 79.35   |
| **52**      | 62.99  | 65.70  | 70.20  | 75.92   | 79.39   |
| **56**      | 62.94  | 65.62  | 70.25  | 76.00   | 79.38   |
| **60**      | 62.80  | 65.51  | 70.24  | 75.93   | 79.52   |
| **64**      | 62.72  | 65.52  | 70.36  | 76.06   | 79.57   |

embedding_dim=32
| **epoch\w** | **16** | **32** | **64** | **128** | **256** |
|-------------|--------|--------|--------|---------|---------|
| **4**       | 59.67  | 61.32  | 66.64  | 69.22   | 69.70   |
| **8**       | 63.21  | 65.95  | 70.58  | 72.76   | 74.90   |
| **12**      | 64.49  | 66.58  | 71.19  | 73.34   | 76.07   |
| **16**      | 64.51  | 66.52  | 70.93  | 73.60   | 76.69   |
| **20**      | 64.47  | 66.14  | 70.67  | 73.90   | 76.95   |
| **24**      | 64.17  | 65.88  | 70.53  | 74.16   | 77.19   |
| **28**      | 63.86  | 65.77  | 70.36  | 74.29   | 77.31   |
| **32**      | 63.59  | 65.71  | 70.30  | 74.37   | 77.54   |
| **36**      | 63.37  | 65.58  | 70.36  | 74.48   | 77.65   |
| **40**      | 63.18  | 65.53  | 70.40  | 74.55   | 77.81   |
| **44**      | 63.06  | 65.45  | 70.47  | 74.62   | 77.89   |
| **48**      | 62.97  | 65.38  | 70.50  | 74.72   | 77.97   |
| **52**      | 62.94  | 65.41  | 70.43  | 74.83   | 78.10   |
| **56**      | 62.78  | 65.30  | 70.48  | 74.86   | 78.22   |
| **60**      | 62.68  | 65.36  | 70.52  | 74.93   | 78.29   |
| **64**      | 62.68  | 65.39  | 70.50  | 75.01   | 78.43   |

embedding_dim=64
| **epoch\w** | **16** | **32** | **64** | **128** | **256** |
|-------------|--------|--------|--------|---------|---------|
| **4**       | 61.70  | 63.04  | 66.18  | 69.43   | 70.19   |
| **8**       | 63.93  | 65.77  | 68.70  | 71.32   | 72.91   |
| **12**      | 63.87  | 65.59  | 68.77  | 71.95   | 73.99   |
| **16**      | 63.53  | 65.31  | 68.81  | 72.49   | 74.78   |
| **20**      | 63.41  | 65.03  | 68.95  | 72.72   | 75.18   |
| **24**      | 63.24  | 65.02  | 69.06  | 73.10   | 75.46   |
| **28**      | 63.01  | 64.94  | 69.11  | 73.24   | 75.78   |
| **32**      | 62.92  | 64.94  | 69.24  | 73.43   | 76.08   |
| **36**      | 62.71  | 64.95  | 69.32  | 73.53   | 76.28   |
| **40**      | 62.46  | 64.99  | 69.45  | 73.70   | 76.49   |
| **44**      | 62.47  | 65.05  | 69.46  | 73.81   | 76.63   |
| **48**      | 62.22  | 65.08  | 69.56  | 73.94   | 76.92   |
| **52**      | 62.29  | 65.03  | 69.67  | 74.03   | 77.10   |
| **56**      | 62.32  | 65.09  | 69.74  | 74.12   | 77.14   |
| **60**      | 62.32  | 65.12  | 69.78  | 74.22   | 77.30   |
| **64**      | 62.25  | 65.11  | 69.83  | 74.32   | 77.50   |

embedding_dim=128
| **epoch\w** | **16** | **32** | **64** | **128** | **256** |
|-------------|--------|--------|--------|---------|---------|
| **4**       | 62.26  | 63.38  | 65.65  | 67.48   | 68.59   |
| **8**       | 63.48  | 64.37  | 66.96  | 69.59   | 71.76   |
| **12**      | 63.31  | 64.38  | 67.27  | 70.36   | 72.60   |
| **16**      | 62.89  | 64.48  | 67.61  | 70.90   | 73.10   |
| **20**      | 62.60  | 64.55  | 67.76  | 71.28   | 73.61   |
| **24**      | 62.49  | 64.67  | 67.91  | 71.62   | 73.99   |
| **28**      | 62.47  | 64.70  | 68.02  | 71.90   | 74.28   |
| **32**      | 62.35  | 64.82  | 68.16  | 72.08   | 74.55   |
| **36**      | 62.41  | 64.85  | 68.28  | 72.32   | 74.77   |
| **40**      | 62.24  | 64.88  | 68.33  | 72.44   | 75.03   |
| **44**      | 62.22  | 64.90  | 68.42  | 72.64   | 75.34   |
| **48**      | 62.34  | 64.93  | 68.54  | 72.82   | 75.58   |
| **52**      | 62.18  | 65.00  | 68.61  | 72.96   | 75.74   |
| **56**      | 62.36  | 65.03  | 68.67  | 73.12   | 75.93   |
| **60**      | 62.21  | 65.04  | 68.76  | 73.24   | 76.05   |
| **64**      | 62.33  | 65.08  | 68.81  | 73.34   | 76.32   |