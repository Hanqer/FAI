# FAI
clothes styles classfication
Project for FashionAI competition.

Use googlenet inceptionV3 to finetune the model. 
To solve the multi-task classfication, train 8 model to 8 different tasks.

The 8 classifiers is described as follows:

+ AttrKey : skirt_length_labels
+ AttrValues :
  - Invisible
  - Short Length
  - Knee Length
  - Midi Length
  - Ankle Length
  - Floor Length

+ AttrKey : coat_length_labels
+ AttrValues :
  - Invisible
  - High Waist Length
  - Regular Length
  - Long Length
  - Micro Length
  - Knee Length
  - Midi Length
  - Ankle&Floor Length

+ AttrKey : collar_design_labels
+ AttrValues :
  - Invisible
  - Shirt Collar
  - Peter Pan
  - Puritan Collar
  - Rib Collar

+ AttrKey : lapel_design_labels
+ AttrValues :
  - Invisible
  - Notched
  - Collarless
  - Shawl Collar
  - Plus Size Shawl

+ AttrKey : neck_design_labels
+ AttrValues :
  - Invisible
  - Turtle Neck
  - Ruffle Semi-High Collar
  - Low Turtle Neck
  - Draped Collar

+ AttrKey : neckline_design_labels
+ AttrValues :
  - Invisible
  - Strapless Neck
  - Deep V Neckline
  - Straight Neck
  - V Neckline
  - Square Neckline
  - Off Shoulder
  - Round Neckline
  - Sweat Heart Neck
  - One	Shoulder Neckline

+ AttrKey : pant_length_labels
+ AttrValues :
  - Invisible
  - Short Pant
  - Mid Length
  - 3/4 Length
  - Cropped Pant
  - Full Length

+ AttrKey : sleeve_length_labels
+ AttrValues :
  - Invisible
  - Sleeveless
  - Cup Sleeves
  - Short Sleeves
  - Elbow Sleeves
  - 3/4 Sleeves
  - Wrist Length
  - Long Sleeves
  - Extra Long Sleeves



#To ues the code, you should build download dataset for fashionAI, which train data labels are in '/Annotations/' , train pics are in '/Images/', test data are in '/rank/' .

#Additionally, you should download 'inception_v3.ckpt' which in '../' when train the model to initialize the weights to finetune.

#Run '../Annotations/label.py' to preprocess the labels, but you should modify the tags and paths in the code to generate 8 labels files such as 'skirt_length_labels.csv'.

#You could run  python3 train_skirt_length.py as a sample to train 8 models.

#The model will be saved in '../model/'  and logs in '../logs/' and events in '../logs/train/'.

#To evaluate the model, you should run '../rank/Tests/pre.py' to preprocess the test labels, but you should modify the tags and paths in the code to generate 8 test labels files such as 'skirt_length.csv'.

#Then, run python3 evaluation.py to generate result in '../rank/Tests/', but you should modify the tags and paths and N-CLASSES for output layer in the code.Then you could merge the results into one file.

