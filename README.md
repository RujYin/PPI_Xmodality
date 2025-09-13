# PPI_Detection_Xmodality_Pytorch

**Extracted_data:**
```
https://drive.google.com/drive/folders/1Dl8p7dKjkDPfw4fFH4sw4IVA8noyzx0y
Includes data from iRefWeb and Negatome
```


**Training:**
```
python main_concatenation.py --train 1 --l0 0.001 --l1 0.001 --l2 0.001 --l3 1000 --data_processed_dir $DIR/processed_data/
```

**Inference:**
```
python main_concatenation.py --l0 0.001 --l1 0.001 --l2 0.001 --l3 1000 --data_processed_dir $DIR/processed_data/
```
