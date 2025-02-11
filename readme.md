# NLP
## HW1 - Multi-label Classification in NLP
* **描述 :** 給予twitter tweet，每個tweet都有一到多個concern，目標是為這些tweet進行concern分類
* **限制 :** 不可使用transformer-based model
* **嘗試的方法 :** 

   https://neptune.ai/blog/text-classification-tips-and-tricks-kaggle-competitions

  **1.資料前處理**(要善用工具)
  
  由於社群媒體的內容容易會有錯字、簡寫、縮寫以及表情符號等，因此應該特別針對這些問題作處理

  **相關資源:**
  
  word correction 、unpack contraction:
  https://github.com/cbaziotis/ekphrasis/tree/master
  
  刪除emoji:
  https://www.educative.io/answers/how-to-remove-emoji-from-the-text-in-python
  
  資料擴增:
  https://github.com/dsfsi/textaugment#rtt-based-augmentation
  https://github.com/hetpandya/textgenie
        
  **2.模型**
  
  採二階段模型:
  
  (1)ELMO : 負責產word embeddings
  
  (2)預測模型 : 主要參考下面連結架構，由兩層的bidirectional GRU組成。


  **相關資源:** https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion/52719

3. **模型超參數搜尋**
   
   使用optuna進行超參數調整
        
    **相關資源:** 
    https://blog.csdn.net/jasminefeng/article/details/119642221
        
## HW2 - Utterance classification
* **描述 :** 
You are given several conversations between the doctor and the patient.
Each utterance in the conversation belongs to one (or more) class.
Goal: Use any model (including transformers, LLM, …) to accurately predict the utterances.
## HW3 - Machine Reading Comprehension
* **描述 :** 給予中文問題，透過模型進行回答
* **模型 :** Mistral-7b
## Final Project
* **描述 :** You are given a question and the current situation. Please judge the quality of the response.
* **方法 :**
     使用Mistral LLＭ模型進行實驗，採用兩種方法：
     (1)Prompt engineering
     (2)Finetuning
     
![image](https://github.com/alexsui/NLP/assets/53047989/0c242efc-3f5a-42ee-bb7c-3e58a5753000)
