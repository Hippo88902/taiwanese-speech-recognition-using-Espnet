# taiwanese-speech-recognition-using-Espnet
# 學號: 311511057 姓名: 張詔揚
## 使用ESPnet來做台語語音辨認
模型說明詳見(Espnet改進與調整.pdf)
## 資料規格:

1. 單人女聲聲音（高雄腔）
2. 輸入：台語語音音檔（格式：wav檔, 22 kHz, mono, 32 bits） 
3. 輸出：台羅拼音（依教育部標準）
   
## 訓練資料目錄:
```
• downloads/
•    data_aishell/
•       resource_aishell/
•          --speaker.info # 關於語者資訊
•          --lexicon.txt  # 辭典
•       wav/
•          train/
•             --global/ # 訓練音檔
•          test/
•             --global/ # Kaggles競賽用音檔
•          dev/
•             --global/ # 測試音檔(不能是空的)
•       transcript/
•          --aishell_transcript.txt #放與音檔對應的文字檔
```

## Table of Contents

- [使用ESPnet做台語語音辨認](#使用espnet做台語語音辨認)
- [環境設置](#環境設置)
- [Data-Preprocessing-for-ESPnet](#data-preprocessing-for-espnet)
- [Training-ESPnet](#Training-ESPnet)
- [Conclusion](#conclusion)

## 使用ESPnet做台語語音辨認

ESPnet是使用類神經網路模型，因此在訓練前，不需要跟kaldi一樣，使用MFCC去求切割位置，而是利用深度學習的方式去訓練特徵參數。

事前準備:

1. 在server上安裝ESPnet (建議新建一個虛擬環境)
2. 確認各種驅動程式，以及套件的版本與ESPnet是相容的 (ex: pytorch、cuda...)
3. 由於有些shell script裡會使用到kaldi，需事先將soft link建立好，否則會出錯


## 環境設置
1. 建立虛擬環境，需先安裝好Anaconda
```sh
$ conda create --name espnet python=3.10
$ conda activate espnet
```
- 記得每新開一次terimnal都要conda activate espnet
  
2. 安裝需要的套件 & ESPnet
```sh
$ sudo apt-get install cmake
$ sudo apt-get install sox
$ sudo apt-get install libsndfile1-dev
$ sudo apt-get install ffmpeg
$ sudo apt-get install flac
$ git clone https://github.com/espnet/espnet
$ cd <espnet-root>/tools
$ CONDA_TOOLS_DIR=$(dirname ${CONDA_EXE})/..
$ ./setup_anaconda.sh ${CONDA_TOOLS_DIR} espnet 3.9
$ make
```

3. 環境設定
```sh
# 設定conda環境
$ cd espnet/tools
$ CONDA_TOOLS_DIR=$(dirname ${CONDA_EXE})/..
$ ./setup_anaconda.sh ${CONDA_TOOLS_DIR} espnet 3.10
$ make -j 40 # CPU核心數，可以用htop看
$ make kenlm.done
```

4. 從 TEMPLATE 創一個新的 recipes
```sh
$ cd espnet/egs2
$ ./TEMPLATE/asr1/setup.sh ./taiwanese/asr1
```

5. 從其他 recipes(Ex: aishell) cp 一個 run.sh 和 local/data.sh 和 conf

現在 `egs2/taiwanese/asr1` ****裡面有：

1. **conf　（訓練的配置）**
    - 如果out of memory，在conf裡找到train的yaml，降低batch_bins
2. **local　（很重要）**
    - **data.sh**：把資料換成 **ESPnet** 可以吃的格式，跟 **kaldi** 差不多
3. **steps　（不用動）**
4. **utils　（不用動）**
5. **pyscripts　（不用動）**
6. **scripts　（不用動）**
7. **cmd.sh　（不用動）**
8. **path.sh　（有問題再動）**
9. **asr.sh　（不用動）**
10. **db.sh　（放database的地方，預設是在downloads）**
11. **downloads**
12. **run.sh　（主要執行的腳本，送參數進去asr.sh）**

裝好可以跑yesno測試環境有沒有問題

## Data-Preprocessing-for-ESPnet

1. 與Kaldi不同，ESPnet除train/test外，另外還需一組dev資料來幫助訓練，使ESPnet在training時，能及時協助評估訓練效能。此外我們通常會將dataset分割為train:test:validation三個部分，三者比例分別為8:1:1，並將所有資料放在downloads目錄裡。
   
2. 在downloads/resource_aishell裡放入speaker.info和lexicon，兩者分別為語者性別及與之對應的辭典。
   
3. 因為資料集為單語者資料，所以可以將所有訓練資料都放在downloads/data_aishell/wav/train/global裡面，若資料為多語者資料，可在download/data_aishell/wav/train裡面依每個語者編號順序放置訓練資料的.wav檔。
   
4. 由於音檔格式為（wav檔, 22 kHz, mono, 32 bits），因此使用 sox 將音檔轉成（wav檔, 16 kHz, mono, 16 bits）

## Training-ESPnet

```sh
$ sudo nvidia-smi -c 3
# 讓gpu進入獨佔模式，可加快訓練的速度(不過要先跟其他人協調好再下這行指令)
$ nohup ./run.sh >& run.sh.log &
# 保證登出不會中斷執行程式，因為training時間較久，下這個指令能確保訓練過程不會因為突發情況中斷。
```
## 訓練參數
```sh
train_set=train
valid_set=dev
test_sets="dev test"

asr_config=conf/train_asr_branchformer.yaml
inference_config=conf/decode_asr_branchformer.yaml

lm_config=conf/train_lm_transformer.yaml
use_lm=false
use_wordlm=false

# speed perturbation related
# (train_set will be "${train_set}_sp" if speed_perturb_factors is specified)
speed_perturb_factors="0.9 1.0 1.1"

./asr.sh \
    --nj 32 \
    --inference_nj 32 \
    --ngpu 1 \
    --lang zh \
    --audio_format "flac.ark" \
    --feats_type raw \
    --token_type char \
    --use_lm ${use_lm}                                 \
    --use_word_lm ${use_wordlm}                        \
    --lm_config "${lm_config}"                         \
    --asr_config "${asr_config}"                       \
    --inference_config "${inference_config}"           \
    --train_set "${train_set}"                         \
    --valid_set "${valid_set}"                         \
    --test_sets "${test_sets}"                         \
    --speed_perturb_factors "${speed_perturb_factors}" \
    --asr_speech_fold_length 512 \
    --asr_text_fold_length 150 \
    --lm_fold_length 150 \
    --lm_train_text "data/${train_set}/text" "$@"
```

p.s.: 如果過程順利，就只要等training結束，若訓練中途出錯，則可根據.log檔去debug。

## 訓練執行過程:

1. 第一階段，特徵抽取(Feature extraction)，首先求fbank,並讓80%的fbank在每一frame都有pitch(訓練用)，接著做speed-perturbed 為了 data augmentation，以及求global CMVN(在某個範圍内統計若干聲學特徵的mean和variance)，如果提供了語句和語者的對應關係，即uut2spk，則進行speaker CMVN，否則做global CMVN。由於我們的資料是單語者資料，所以做global CMVN。

2. 第二階段，準備字典(Dictionary Preparation)，對訓練資料所有出現的word進行處理，為了使lexicon與wav音檔的答案對應。

3. 第三階段，準備語言模型(LM preparation) ，訓練我們的語言模型，並產生n-gram model(這裡我們用4-gram)。

4. 第四階段，Network training，最花時間也最耗費GPU計算資源的階段，訓練的過程會藉由dev的辨識結果進行修正，讓model能夠越訓練效果越好。

5. 最終階段，Decode完後，即可得到我們要的ASR model，並辨識test裡面的.wav檔案，並得到訓練的結果。

p.s.1: 前三個階段可看作是training的事前準備，都交由CPU去處理，因此所花費的時間並不多。

p.s.2: 除此之外，也能調整batch size大小，讓GPU的memory能夠盡量地吃滿(提高利用度)，此舉能夠加快training的速度。

## Conclusion:

![image](https://github.com/MachineLearningNTUT/taiwanese-asr-using-kaldi-toolkit-Hippo88902/blob/main/Kaldi%E7%B5%90%E6%9E%9C.jpg)

![image](https://github.com/MachineLearningNTUT/taiwanese-asr-using-kaldi-toolkit-Hippo88902/blob/main/ESPnet%E7%B5%90%E6%9E%9C.jpg)

總體來說，ESPnet的訓練效果會比Kaldi的訓練效果來的好，不過ESPnet所需要的訓練時間也比較長，也較耗費計算資源。然而在資料量較少的情況下，Kaldi基於傳統機率統計模型的訓練方式，或許會表現的比ESPnet還好，但是當data的量足夠多時，神經網路的訓練方式通常會outperform傳統的機率統計模型。因此在資料量充足時，我們會採用Neural network的方式進行訓練，在資料量不足以train起一個model時，則可以使用機率統計模型來得到一個還不差的結果，這或許成為了一種data數量上的trade-off。
