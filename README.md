# Text2Image implementation with StackGAN
# Motivation for the project and an explanation of the problem statement

<h2>The power of visualization</h2>

* 인간은 시각화된 자료를 쉽게 인지할 수 있다고 함.
* 오감 중에서 시각이 70~80%의 정보를 습득하는 수용체
* 시각화가 가능해지면 "Read"가 아닌, "See"의 방법으로 더 빨리 정보를 받아들일 수 있게 됨.

![대체 텍스트](https://i.ibb.co/CMwcnZb/num-dummy.png)

<h2>Text2Image? => StackGAN</h2>

![대체 텍스트](https://miro.medium.com/max/3356/1*g-0onhpbu6dU0aZbpfEUeA.jpeg)

# A description of the data

![대체 텍스트](https://vision.cornell.edu/se3/wp-content/uploads/2017/04/Screenshot-from-2017-04-29-16-36-17-705x456.png)

* Caltech-UCSD Birds 200
* 총 200개의 다른 종으로 이루어진 11,788장의 이미지
* 각 이미지는 새를 segmentation하고, 속성이 어떠한지 카테고리별로 설명되어있음.
* ex) 부위별 색상은 어떤 색인지, 사이즈는 small/medium/large인지, 부위별 패턴은 어떤지 영단어로 표시되어 있음.

# Hyperparameter and architecture choices that were explored

<h2>Idea</h2>

* 기존의 Text to image : 주어진 문장을 기반으로 하나의 GAN을 통해 이미지를 생성함
* Text to image는 어려운 문제이므로 **두가지 sub problem으로 나누어 보다 고해상도의 이미지를 생성하자!**

<h2>StackGAN Model Architecture</h2>

![대체 텍스트](https://miro.medium.com/max/4756/1*NwDrP1Zi6xj1bGN62wrf3g.jpeg)

<h2>StackGAN Model - Conditioning Augmentation</h2>

* 한정적인 데이터가 고차원의 text embedding 공간에서 불연속성을 야기하는 문제점을 해결하기 위해 도입한 구조.

* $N(\mu(\rho_t)),\sigma(\rho_t))$ 에서 샘플링한 conditioning variable을 text feature로 사용하여 generator의 input에 추가시킴.

* Gaussian distribution에서 샘플링한 text embedding을 사용하므로 randomness가 더해져 동일한 문장에 대해서도 다양한 이미지를 생성할 수 있는 효과가 있음.

* 해당 방법은 generator의 loss에 $D_K$$_L(N(\mu(\rho_t)),\sigma(\rho_t))||N(0,1))$를 추가함으로서 가능함.

![대체 텍스트](https://i.ibb.co/vH2D052/1.pnghttps://)

<h2>StackGAN Model - Stage1</h2>

* GAN을 이용해 text에 대한 초기 shape과 color를 나타내는 저화질 이미지를 1차적으로 생성하는 단계

* Generator : conditioning variable과 noise로부터 저화질의 초기 이미지를 생성함.

* Discriminator : image와 text feature를 기반으로 real / fake 판단.

![대체 텍스트](https://i.ibb.co/R3t6hLK/2.png)

<h2>StackGAN Model - Stage2</h2>

* Stage1에서 생성한 이미지를 수정하고 추가적으로 디테일한 부분을 생성하는 단계.

* Generator : conditioning variable과 stage1의 결과로부터 보다 자세한 고화질 이미지 생성.

* Discriminator : image와 text feature를 기반으로 real / fake 판단.

![대체 텍스트](https://i.ibb.co/C5pqPcP/4.png)

# 구현 방법
1. Batch normalization X + Adam optimizer 사용
2. Batch normalization X + RMSprop optimzer 사용
3. Batch normalization X + RMSprop optimizer + Wassertein Loss 변경
4. Batch normalization O + Adam optimizer 사용 
5. Batch normalization O + Adam optimizer + Wassertein Loss 변경 + Dropout 추가
6. Learning rate 기존의 10배로 학습 진행 

# Code implementation

PT_mateiral_with_code.ipynb 파일 참고

# Presentation of results

* MODEL : Batch normalization X + Adam optimizer 사용

  <img src=https://i.ibb.co/Q7VqfcG/black.png, width=150><img src=https://i.ibb.co/4PfNyKG/black2.png, width=150><img src=https://i.ibb.co/4PfNyKG/black2.png, width=150><img src=https://i.ibb.co/4PfNyKG/black2.png, width=150>
  
* MODEL : Learning rate 기존의 10배로 학습 진행

  <img src=https://i.ibb.co/hydFCT2/legend.png, width=450>
  
* MODEL : Batch normalization X + RMSprop optimzer 사용

  <img src=https://i.ibb.co/c3W0KLT/NBN.png, width=850>
  
* MODEL :Batch normalization X + RMSprop optimizer + Wassertein Loss 변경
 
 <img src=https://i.ibb.co/qYw6nc3/wgan-rmsprop.png>
 
* MODEL : Batch normalization O + Adam optimizer 사용 -> Mode Collapse 발생

  <img src=https://i.ibb.co/8cGhWbN/original-epoch.png, width=800>
  
* MODEL : Batch normalization O + Adam optimizer + Wassertein Loss 변경 + Dropout 추가

  * STAGE1
  ![대체 텍스트](https://i.ibb.co/ZTrjXtb/ungyeong.png)

  * STAGE2
  <img src=https://i.ibb.co/gMH1YSr/image.jpg, width=600>
  
# Analysis of results
* 모델에 따른 학습시간
  * learning rate 기존의 10배로 학습 진행
 
    -> STAGE 1 : 94epochs, 약 12시간 소요 / 88epoch 이후로 loss 값에 NAN 값만 출력 / 높은 Learning rate가 문제인 것으로 보임.
  *  Batch normalization X + RMSprop optimzer 사용
  
    -> STAGE 1 : 150epochs, 약 7시간 소요 / 150epoch 이후 런타임 끊김 현상. 
  * Batch normalization X + RMSprop optimizer + Wassertein Loss 변경
 
    -> STAGE 1 : 180epochs, 약 4시간 소요 / 180epoch에서 세션 끊김 현상 지속적 발생

  *  Batch normalization O + Adam optimizer 사용  
    -> STAGE 1 : case1 = 138epochs, 약11시간 소요 / case2 = 600epochs, 약 8시간 소요        
     
  * Batch normalization O + Adam optimizer + Wassertein Loss 변경 + Dropout 추가
 
    -> STAGE 1 : 600epochs, 약 4-5시간 소요
 
    -> STAGE 2 : 8epochs, 24시간 소요
    
  * 학습 시간이 너무 오래 걸려 learning rate를 0.002도 작은 learning rate라고 생각해서 0.0002에서 0.002로 높여봤음. 
  * 이는 학습에 별로 도움이 되지 못하고 오히려 loss값에 nan값을 띄게 됨. loss가 증가하다가 무한대로 수렴했기 때문일 것.

   ![대체 텍스트](https://i.ibb.co/P6hJ0jK/nan.png)
  
  * 출처 : https://stackoverflow.com/questions/52211665/why-do-i-get-nan-loss-value-in-training-discriminator-and-generator-of-gan

  * GAN 변형 모델인 StackGAN의 성능을 높이기 위해서도 Wasserstein loss가 효과가 있었음. 다른 StyleGAN이나 CycleGAN에도 Wasserstein loss로 학습하면 더 좋은 학습이 가능해 질 수도 있다고 생각.
  
* inception score 측정

  ![대체 텍스트](https://i.ibb.co/1Lb0dMj/inception-1.png)
  
  * Stage1밖에 학습을 진행하지 못했기 때문에 기존의 StackGAN의 Inception score에 달하는 score 얻지 못함.
  * data를 뜯어서 봐보니 GAN이 생성한 이미지가 주변에 흰색 배경의 패딩이 있었음.

  ![대체 텍스트](https://i.ibb.co/4Rxm13S/inception-3.png)
  
  * 이를 해결하면 더 좋은 inception score를 얻을 수 있다고 생각함.

  ![대체 텍스트](https://i.ibb.co/yBVR6Y9/inception-2.png)
  
  * 겉의 흰 배경을 제거했더니 대체적으로 더 좋은 score를 받는 것 확인.
  
# Insights and discussions relevant to the project

* GAN은 학습이 굉장히 어려움
  * 학습이 잘 되기 위해서는 서로 비슷한 수준의 생성자와 구분자가 함께 조금씩 발전해야 힘 한쪽이 너무 급격하게 강력해지면 이 관계가 깨져서 학습이 이루어지지 않음


*   DCGAN, WGAN,EBGAN, BEGAN,CycleGAN, DiscoGAN 등 성능향상 및 모델 안정화 위해 다양한 모델 출현
  *   출처 : https://dreamgonfly.github.io/2018/03/17/gan-explained.html


* GAN은 상당히 오랜 시간 학습 필요, 장시간 학습 중 런타임 연결 끊김 계속 발생
  * 개발자 도구(F12)에서 console에 명령어 입력으로 해결
```
 function ClickConnect() { 
   var buttons = document.querySelectorAll("colab-dialog.yes-no-dialog paper-button#cancel"); 
   buttons.forEach(function(btn) { btn.click(); }); 
   console.log("1분마다 자동 재연결"); 
   document.querySelector("#top-toolbar > colab-connect-button").click();
    } 
   setInterval(ClickConnect,1000*60);
```
  * 출처: https://bryan7.tistory.com/1077 [민서네집]

* GAN관련 모델을 학습시키고 싶으시다면,,, GTX 1080Ti graphic card 추천..!

   ![대체 텍스트](https://i.ibb.co/BVGjN0V/gtx1080.pnghttps://)
   * 출처 : https://stackoverflow.com/questions/58595157/colab-gpu-vs-gtx-1080

* github 코드들 활용하려 했지만, 다른 python, tensorflow 버전에서 작성되어있어서 가상환경에서 다운그레이드를 해서 다시 시도.
 *  그래도 원하는대로 잘 안됐음.. 버전 상 호환 안되는 부분이 많기 때문인 듯. 클론 해보고 싶은 코드가 있다면 먼저 라이브러리 버전 확인을 해보시기를..!

# References

* 웹사이트
  * StackGAN 전반적인 구조 및 구현 메인 코드
   - https://medium.com/@mrgarg.rajat/implementing-stackgan-using-keras-a0a1b381125e

  * StackGAN Inception score 산출
   - https://github.com/hanzhanggit/StackGAN-inception-model
   - https://machinelearningmastery.com/how-to-implement-the-inception-score-from-scratch-for-evaluating-generated-images/

  * StackGAN 구조 이해
   - https://www.youtube.com/watch?v=G2_8Jc0IwYk


* 논문

  * “Generative Adversarial Network”(2014,Ian J. Goodfellow외) 

  * “Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks” (2016, Alec Radford외)

  * “StackGAN: Text to Photo-realistic Image Synthesis with Stacked Generative Adversarial Networks”(2017,Han Zhang 외)

  * "Stacked Generative Adversarial Networks"(Xun Huang et al)
