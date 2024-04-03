CSE498R-Recognizing-Gender-From-Voice-Audio
CHAPTER 1: INTRODUCTION   1.1 Basic Introduction

We are all acknowledged with the relevance of speech and its applications in our surroundings. Our phones are welcoming your voice more than ever. The use of voice-to-assistance concept is all around us. One can now ask their personal phone about today's weather right after they wake up by calling the name of that virtual-assistance inside their phone; some call it magic, others call it otherwise. In this project report, we try to present a process to model an intuitive application for Gender classification; where it receives an audio file and processes that into a single output; a 'male' or a 'female' voice. Gender classification is a decent problem for someone who is at the door-step of learning in this tremendous field. Regardless of all the work that must be done prior to building this model; we as Machine Learning/ Deep Learning practitioners, can work together toward to further the impact of what Voice-Recognition can do to ease the life of others.
Women speak at a higher pitch (frequency). An adult woman’s average range is from 165 to 255 Hz. Higher vocals create the impression that they are soft, because of their lower resonance. Women usually speak at one octave higher than men. Retrospectively, man’s voice pitch fall under low frequency. It is between 85 to 155 Hz. The rush of testosterone generated during adolescence leads men's vocal cords to extend and thicken, resulting in a deeper voice. Because of their higher resonance, lower-pitched voices appear to be louder.
Gender recognition can be used to detect emotions such as male or female sadness, angry, happiness. Personal assistants such as Siri and Google Assistant provides female generic or male generic answers to questions and this has a lot to do with voice gender recognition. Adding tags to sound also helps classify them and clarify and decrease search area. Overall, the application of this project is numerous.
 

1.2 Social and Technological Importance of Voice Recognition

Molly Babel, Grant McGuire and Joseph King carried out a study on vocal attractiveness. Although opinions by both sexes were often correlated, males generally rated fellow males as less attractive than females did. However, both females and males had similar ratings of female voices. The study arrived at the conclusion that the judgement of vocal attractiveness is complex. This conclusion might stem from the fact that different audiences have different voice preferences. Another study by Harrison Interactive stated that 48% of the respondents thought male voices were more forceful while 49% were of the opinion that gender made no difference. 46% of the respondents were of the opinion that female voices were more soothing. 19% of the respondents said that female voices were more persuasive while 18% said male voices were more persuasive. However, there was a large concession that they were most likely to buy gadgets like computer and cars advertised using male voice over. Several other studies have also revealed correlated opinions about choosing a male or female voice over. However, one thing is clear; different voices work for different products and different people. To answer this recurring question, we also decided to look to the consumers themselves. What did they prefer? Based on our review of opinions around the topic, we discovered that most consumers weren’t always aware of which voice is suitable for which product except when they’re asked. The common opinion among customers was that it didn’t matter if it was a male or female voice, rather, they were more concerned about a voice that says “trust me”. Trust was more important to these consumers than the sex behind the voice, a reason why the Scottish accent is winning call center work. However, it is important to note that female voices are typically seen to be more trustworthy, with 66% of survey respondents claiming to prefer a female voice when watching an internet video for this very reason. This could also explain why we typically hear the voice of a woman when we’re on hold, or reach a company’s IVR system, as they help us feel more relaxed. Alternatively, it’s usually a male voice heard when communicating authoritative messages. Although most respondents believed that gender did not matter when voicing things like ads, when it comes to voiceovers like emergency announcements, it will more often than not be a man’s voice heard. [13]

1.3 Importance of Benchmarking

Benchmarking is the process of comparing the results of various algorithms or models on a set of data. Various models that are trained and tested on different sets of data adds bias in the results, hence their results might not translate onto other similar datasets. On the other hand, models that are trained on the same training set and got similar results, might still give different results on the test set depending on the overfitting or generalization of the models, or even the preprocessing of the images. Benchmarking different state of the models gives us the opportunity to see how the models perform on the same set of data, preprocessed using the same pipeline. Keeping the other variable constant, we can see how the architecture and hyper parameters of each model affects the results. This also allows us to attempt to explain the working of the models, reducing the “black-box” characteristic and using that knowledge to farther tune our state of the art models.

1.4 Aim of this research

In this research, my aim is to compare multiple state of the art models in voice gender recognition on the OPENSLR12 database. This research aims to find an effective way to train models to differentiate between male speaker and female speakers and strives to develop a platform where, given an audio input, the model would give out an output of the gender of the speaker. This work will positively impact artificial intelligence and AI assistants and make it easier to do gender classifications in the future.   CHAPTER 2: LITERATURE REVIEW   While working on this research project, we studied various research papers related to our project topic and picked a few papers from there which were conducted on gender detection using different deep learning neural networks. We chose these particular papers because their working approach is closely related to our research work. We also deduced some ideas from there for doing our work.

2.1 EXISTING LITERATURE EXPLANATION

A study has been done by Mücahit Büyükyılmaz and Ali Osman where they use Multilayer Perceptron (MLP) deep learning model and it has been described to recognize voice gender. Their data set have 3,168 recorded samples of male and female voices. The samples are produced by using acoustic analysis. This model achieves 96.74% accuracy on the test data set. Though their accuracy is good, their training methodology is very singular.

Another study named “Automatic Identification of Gender from Speech” by Sarah Ita Levitan, Taniya Mishra, Srinivas Bangalore deals with similar problems where they deal with women, men and children’s voices. They have above average success rate while distinguishing the genders.  

CHAPTER 3: METHODOLGY

  In this chapter, the methodology followed in this research is explained in details. Starting with a description of the dataset and how it has been collected, then introducing the frameworks that were attempted and their features and limitations, and finally the models that were implemented using these frameworks

3.1 Workflow

To solve this project, we followed certain steps:

 Firstly, we took the dataset of .flac audio files and assigned them as inputs to extract_features to retrieve features from this audio files. The features that were used are MFCCs, Intensity (chroma), mel-scaled spectrogram, Spectral Contrast, tonnetz. The python parselmouth library parses an audio file and uses a.praat file to extract various information such as the number of syllables in a speaking period, speaking time, articulation, count of pauses, mean-frequency, min-frequency, and max-frequency.

 Then the extracted features are added to the dataset or dataframe.

 After that, we use SPEAKER.TXT file to extract the audio file's gender (target-label) and add it to another dataframe.

 After passing through all audio files, we join both dataframes to get the whole dataset and get the output dataset as a .csv.

 Finally we saved the features vector so to use it later.

 In the next steps, we loaded the previously saved features vector and shuffled training and testset.

 After that, we split to X and Y for both the training-set and Test-set and created a dev/crossvalidation set from the test-set because both of these sets must be from the same distribution and different from the training-set.

 We also one-hot-encoded the labels to be able to use it in the softmax activation function in the output neuron rather than having a restricted '0' or '1'; to have a conditional probability instead. Then we used standardization on the feature vector to have a zero-mean with a standard deviation to get an uniform result.

3.2 Dataset

The data and the model are intimately connected. The data determines how good a model is. Kaggle was utilized to get the dataset for detecting gender based on audio events. We used OPENSLR12 for gender classification. This dataset is concerned with books as different speakers from different genders read different books on diverse periods. This dataset is solely audio files with '.flac' formats. Dataset has following folder structure:

train-clean-100: is a folder which contains multiple folders names with the ID of the speaker, inside each of folder there are multiple folders of the ID of the book; inside that the .flac audio files exist alongside a transcript .txt file.

BOOKS.TXT: is a .txt file that contains the names of the books. There are 1568 books that were read by both genders.

CHAPTERS.TXT: is a .txt file that contains the read chapters from the speakers.

SPEAKERS.TXT: is a .txt file that contains the names of the speakers alongside their Gender; which will help us in determining the gender its corresponding audio file.

The length of this training-set is 300 hours with 6.3 GB in size. The size of this Test-set (which comes from a different distribution than the training-set) is 346 MB.

3.3 Feature Extraction

When the audio-file is supplied to the extract feature function, a different set of features is extracted. Those features are

• MFCCs • Chroma (intensity) • mel-scaled spectrogram • Spectral Contrast • Tonnetz • Audio Duration • Rate of speech • Number of syllables • Speaking time (s) • Articulation (Speed)
• Count of pauses with fillers • Mean frequency • Minimum frequency • Maximum frequency

The above features are helpful in nature to visualize their properties and get a glimpse of the behavior of this data. EDA is needed the most in the Speech Recognition field, because the features are so 'dependent', removing one feature or adding an 'unnecessary' feature might outcome unwanted results. Being extremely selective with the features is mandatory to ensure validity between different models.

3.4 Architecture Used

CNN 
The CNN model is a form of neural network that allows us to extract higher representations for image input. Unlike traditional image recognition, which requires the user to define the image features, CNN takes the image's raw pixel data, trains the model, and then extracts the features for improved categorization. For the modelling we used Convolutional Neural Network that included

 Baseline Convolutional Neural Network:

                                    Fig: Baseline Convolutional Neural Network      
Convolution Network start with separate temporal and spatial convolutional layers and are followed by a pooling layer. Before the final fully linked linear layer with softmax activation, an extra average pooling layer was included. The kernels are represented by the integers between brackets.

 Improved Convolutional Neural Network  Hyper-parameter tuned deep Convolutional Neural Network

Hyper-parameter tuning 
Hyper-parameters are variables that determine the network topology (for example, the number of hidden units) and how the network is trained (Eg: Learning Rate). Prior to training, hyper-parameters are set.

For hyper-parameter tuning we use Convolutional Neural Network and that included

 Creating a baseline model to compare its loss and accuracy to the improved model.

 Creating an improved model upon the baseline and demonstrate its accuracy.

 Using Keras Tuner with the some different parameters and ranges to random-search and return the best model.

 

CHAPTER 4: Deployment

4.1 RESTFUL API Deployment using the resulting model

After the training of the model has finished and it performs satisfactorily, this project will be served as a Restful API with a docker-file to build to be tested on. Flask is used to allow testing for your own voice; keeping in mind that the available audio-file-extensions are: 'flac and .wav; that is due to the lossless nature of those extensions; especially the .wav; it is LINEAR16 or LINEAR PCM; which is an example of uncompressed audio in that the digital data is stored exactly as the standards imply. You can run the Flask API on your local machine and use Postman or Advanced Restful Client on Google Chrome App or any other Restful client that you prefer to test your voice on those models. Two APIs have been built, one with the best generated model and the other one is based on a voting-behavior system, that takes into account the majority of decisions.

The procedure of the prediction for the best-generated-model (CNN) goes as follows: • Upload a .flac or .wav audio-file to the server, it gets saved in ./Data/temp. • Feature Extraction for CNN. • Delete audio-file and spectrogram to retain space. • Predict. The procedure of the prediction for the voting-behavior from the best model of ANN, CNN and ML, then return the prediction with the most votes (Note: ANN and ML only can predict the unrecognizability of the audio-file, that is due to the .praat file from parselmouth, which is not available for CNN as it is only concerned with images.

The process of predicting the gender goes as follows: • Upload a .flac or .wav audio-file to the server, it gets saved in ./Data/temp. • Feature extraction whether for CNN or ANN. • Delete audio-file and spectrogram to retain space. • Predict. • Most votes is the determined final prediction

The first version is preferred which is the best-generated-model. This eliminates confusion between models, there are mainly four reasons for this: CNN is the only model that uses images as features while ANN and ML use the same numeric features, so they will be biased toward each other when it comes to learning and predicting. There is a feature of ANN and ML model is that it can detect unnatural sounds and that is not available in CNN for the reason that we do not have a third label that identifies 'Unrecognized voice' but in ANN and ML this feature is doable through feature extraction by using parselmouth library with the .praat file (yet it needs improvements in this regard). CNN can generalize more because there is no a 'clear' feature extraction, what we actually do is to extract a spectrogram image and can obtain features through the CNN Architecture. Also, CNN offers the availability of Image Augmentation which can increase the generalization of the model; as I have noticed this on multiple voices that I have personally recorded with noise and bad microphone quality yet it still recognizes the gender. ANN has gotten used to the train and validation data regarding reading from a textbook. ANN sometimes misinterpret the reader if the reader is: singing, speaking fast or young/very old in age.

  CHAPTER 5: DISCUSSION

In this research, majority of the limitations faced was due to unavailability of resources. The work was done on Google Colab, which is a free platform provided by Google to aid in the work of small, independent researchers or Machine learning enthusiasts/hobbyists. However, dealing with a huge dataset as OPENSLR12 on Colab is very difficult, due to the shortage of RAM and shared GPUs. A lot of time has been spent on making the pipeline executable on the platform, which otherwise, would have been given in implementing models and improving their results, or evaluation. Due to the limitations on patch size, batch size, epoch time vs epoch duration, the results obtained are not very dependable or reproducible either. Convolutional neural networks are set to be a suitable (if not the best) choice to do your Voice Gender Detector, the variety of different parameters that can be alterd are quite large in number, and relatively expensive in terms of computational complexity. CNN is generally slower becuase the diversity of processes that belong to it is absolutely second to none.

The baseline model has achieved a good combination of both CV accuracy and Test-set accuracy of 94.89%.
The improved model has merely performed less than the baseline model, that is due to the addition of Dropout after the pooling process of each Conv layer.
Hyperparameter tuning was done on Keras-tuner as it has achieved a blowing accuracy of 95.6% on Test and 97.8% on CV; This model from keras-tuner will be used as the main model to distinguish between a Male and a Female.   CHAPTER 6: CONCLUSION   In this research, benchmarking ANN and CNN models on audio samples to recognize gender of the speaker has been attempted. Due to resource limitations for the ongoing pandemic, a big part of the initial research proposal has not been possible. However, this opportunity has allowed me to learn extensively about how to deal with such limitations, how important efficiency and cost effectiveness of a system is, and how to use existing resources to get the maximum outcome possible. Benchmarking models used on audio samples to differentiate the genders is extremely important in increasing the transparency and legitimacy of models and doing such research is important to progress the technological field. Human limitations might get alleviated greatly if AI assistants can automate mundane chores, so advancement in this field means a significant improvement in the technological sector, allowing humans to focus on the more important work. Hence, I shall carry on working in this domain to make any contribution possible, however small it might be.   References
Gender identification from speech signal by examining the speech production characteristics (http://ieeexplore.ieee.org/document/7980584/)

Speech Recognition with Deep Learning (https://medium.com/@ageitgey/machinelearningis-fun)

Gender Clustering and Classification Algorithms in Speech Processing: A Comprehensive Performance Analysis
(http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.258.9728&rep=rep1&type=pdf)

Dataset: Voice gender dataset (https:// www.openslr.org/resources/12/train-clean-100.tar.gz)

A.P. Vogel, P. Maruff, P. J. Snyder, J.C. Mundt, Standardization of pitch-range settings in voice acoustic analysis, Behavior Research Methods,

L. Breiman, J.H. Friedman, R.A. Olshen, C.J. Stone, Classification and Regression Trees, CRC Press, 1984.

L. Breiman, “Random forests”, Machine Learning, Springer US, 45:5–32, 2001.

OmarHory GitHub Repo (https://github.com/OmarHory/)

D. Nagajyothi and P. Siddaiah, “Speech Recognition Using Convolutional Neural Networks”, International Journal of Engineering & technology 7(4.6) (2018).

Akhil Thomas, “Speaker Recognition using MFCC and CORDIC Algorithm”, International Journal of Innovative Research in Science, Engineering and Technology Vol. 7, Issue 5, May 2018.

Du Guiming et al, “Speech Recognition Based on Convolutional Neural Network”

Automatic Identification of Gender from Speech by Sarah Ita Levitan, Taniya Mishra, Srinivas Bangalore (http://www.cs.columbia.edu/~sarahita/papers/speech_prosody16.pdf)

Difference between male and female voice (https://matinee.co.uk/blog/difference-male-female-voice/)
