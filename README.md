# PPI_Detection_Xmodality_Pytorch
**TODO1**: Curate the data (sequences / contact maps etc.)
**TODO2**: Get the statistics of the data; split and process the data in terms of the number of unseen proteins in the training set
**TODO3**: Finish coding utils.py according to TODO1, TODO2 and procudures of analyzing the results
**TODO4**: Improve the model (transformer?)



**Training:**
```
python main_concatenation.py --train 1 --l0 0.001 --l1 0.001 --l2 0.001 --l3 1000 --data_processed_dir $DIR/processed_data/
```

**Inference:**
```
python main_concatenation.py --l0 0.001 --l1 0.001 --l2 0.001 --l3 1000 --data_processed_dir $DIR/processed_data/
```
