# Strategy

* Approach : Classification problem with 12 classes (one per completed figure)
* Test Dataset (input) : Fixed **images** with **completed figure**s and with **hands** of participants
* Model : Convolutional Neural Network
  * Convolutional layers
  * Pooling Layers 
  * Final Layer : softmax.
* Framework : Keras with TF2 as backend

# Status

| Task                                         | Status                              | Comments                                                     |
| -------------------------------------------- | ----------------------------------- | ------------------------------------------------------------ |
| Data acquisition                             | over 2k images (all classes)        | Need more data - May need to clean data (no/fewer hands?)    |
| Model Training                               | Done                                | Optimisation may be needed                                   |
| Model Test                                   | Tested on  images not seen by model | Accuracy smaller than than 50 %  (tested on finished figures without hands) |
| Model - Optimization                         | to do                               | Parameters: layers, learning rate, etc                       |
| Model implementation on real time processing | to do                               |                                                              |

# Results / Improvement Strategy

* Accuracy better than 8.4 % (random prediction) is still very low. 

​	

* Data-based improvements : 
  * Clean Data to have less hands present in the frames
  * Get more data :  extract more frames from videos
* Model-based improvements (lower impact on accuracy may be expected) :
  * Parameter tuning
  * Use another framework ? 