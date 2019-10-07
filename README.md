# MLDS2017
Projects in the course of "Machine learning and having it deep and structured" (Spring 2017)

The machine learning part are using Tensorflow.
* Number recognition.
  * Use CNN to do classical MNIST number recognition.
* Language model.
  * Use LSTM to build a language model to solve English cloze multiple choices problem.
* Attention model.
  * Use sequence-to-sequence LSTM and attention mechanism to construct a caption for a video. 
  * The input is already the 4096-dim features for each video frame.
* GAN.
  * Plot a comic face that achieve the requirements of specific hair color and eyes color.
  * Use conditional-DCGAN, Least squared GAN, WGAN and WGAN with gradient penalty.
* Reinforcement learning.
  * Use Seq2Seq model with RL to build a chatbot.
* Final project: Adam analysis
  * Discuss Adam optimizer on 5 different tasks, including MNIST, Cifar-10, regression problem, and GAN. 
  * Discuss how Adam behaves near the saddle point in the hyper loss space.
  * Discuss whether the default parameters of Adam in popular ML library, such as Keras, are the best ones.
  * Online report link (Chinese): https://ntumlds.wordpress.com/2017/03/26/b03202047_california/
