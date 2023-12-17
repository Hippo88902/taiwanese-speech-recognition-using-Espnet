# taiwanese-speech-recognition-using-Espnet
# 學號: 311511057 姓名: 張詔揚

## 使用ESPnet做台語語音辨認

ESPnet是使用類神經網路模型，因此在訓練前，不需要跟kaldi一樣，使用MFCC去求切割位置，而是利用深度學習的方式去訓練特徵參數。

模型說明詳見(Espnet改進與調整.pdf)

## 資料規格:

1. 單人女聲聲音（高雄腔）
2. 輸入：台語語音音檔（格式：wav檔, 22 kHz, mono, 32 bits） 
3. 輸出：台羅拼音（依教育部標準）
   
## Table of Contents

- [環境設置](#環境設置)
- [訓練資料目錄](#訓練資料目錄)
- [Data-Preprocessing-for-ESPnet](#data-preprocessing-for-espnet)
- [Training-ESPnet](#Training-ESPnet)
- [Conclusion](#conclusion)
- [附錄](#附錄)

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
- p.s.: 記得每新開一次terimnal都要conda activate espnet
  
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
   
   - **conf　（訓練的配置）**
       - 如果out of memory，在conf裡找到train的yaml，降低batch_bins
   - **local　（很重要）**
       - **data.sh**：把資料換成 **ESPnet** 可以吃的格式，跟 **kaldi** 差不多
   - **steps　（不用動）**
   - **utils　（不用動）**
   - **pyscripts　（不用動）**
   - **scripts　（不用動）**
   - **cmd.sh　（不用動）**
   - **path.sh　（有問題再動）**
   - **asr.sh　（不用動）**
   - **db.sh　（放database的地方，預設是在downloads）**
   - **downloads**
   - **run.sh　（主要執行的腳本，送參數進去asr.sh）**

   裝好可以跑yesno測試環境有沒有問題

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

現在有兩條路可以選(以aishell為例)：

1. 把資料放到downloads，在`local`寫一個 **taiwanese** 的`data.sh`　（較難） 
2. 沿用aishell的data.sh，把資料以 **aishell** 的格式放入`downloads`　（較易）
   
data.sh一樣需要：
```sh
spk2utt # Speaker information
text    # Transcription file
utt2spk # Speaker information
wav.scp # Audio file
```
- p.s: test音檔的id需要加到text後面，因為test是我們要的辨識結果所以先給 a e i o u
- 
![image](https://github.com/Hippo88902/taiwanese-speech-recognition-using-Espnet/blob/main/test.png)

aishell的檔案架構:
```sh
data_aishell/
   resource_aishell/
      --speaker.info
      --lexicon.txt(E2E不需要)
   wav/
      train/語者名稱資料夾/audiofile
      test/語者名稱資料夾/...
      dev/語者名稱資料夾/...
   transcript/
      --aishell_transcript_v0.8.txt #放與音檔對應的文字檔
```
- p.s.: 因為只有一個語者，所以train, test, dev裡面只有一個資料夾
  
## Data-Preprocessing-for-ESPnet

1. 與Kaldi不同，ESPnet除train/test外，另外還需一組dev資料來幫助訓練，使ESPnet在training時，能及時協助評估訓練效能。此外我們通常會將dataset分割為train:test:validation三個部分，三者比例分別為8:1:1，並將所有資料放在downloads目錄裡。(p.s.: 訓練的過程會藉由dev的辨識結果進行修正)
辨識後的結果放在：
`/espnet/egs2/taiwanese/asr1/exp/asr_train_.../decode_asr_.../test/text`

3. 在downloads/resource_aishell裡放入speaker.info和lexicon，兩者分別為語者性別及與之對應的辭典。
   
4. 因為資料集為單語者資料，所以可以將所有訓練資料都放在downloads/data_aishell/wav/train/global裡面，若資料為多語者資料，可在download/data_aishell/wav/train裡面依每個語者編號順序放置訓練資料的.wav檔。
   
5. 由於音檔格式為（wav檔, 22 kHz, mono, 32 bits），因此使用 sox 將音檔轉成（wav檔, 16 kHz, mono, 16 bits）

## Training-ESPnet

若有空閒的GPU資源可下:
```sh
$ sudo nvidia-smi -c 3
# 讓gpu進入獨佔模式，可加快訓練的速度(不過要先跟其他人協調好再下這行指令)
$ nohup ./run.sh >& run.sh.log &
# 保證登出不會中斷執行程式，因為training時間較久，下這個指令能確保訓練過程不會因為突發情況中斷。
```

如果放入處理好的資料可以跑完所有流程，便可以開始著手修改config，可以沿用aishell或是librispeech裡面的conf檔。

- **EX1：aishell**
    
    aishell的`conf/tuning`裡面有：
    
    - `train_asr_conformer.yaml`
        
        ```yaml
        # network architecture
        # encoder related
        encoder: conformer
        encoder_conf:
            output_size: 256    # dimension of attention
            attention_heads: 4
            linear_units: 2048  # the number of units of position-wise feed forward
            num_blocks: 12      # the number of encoder blocks
            dropout_rate: 0.1
            positional_dropout_rate: 0.1
            attention_dropout_rate: 0.0
            input_layer: conv2d # encoder architecture type
            normalize_before: true
            pos_enc_layer_type: rel_pos
            selfattention_layer_type: rel_selfattn
            activation_type: swish
            macaron_style: true
            use_cnn_module: true
            cnn_module_kernel: 15
        
        # decoder related
        decoder: transformer
        decoder_conf:
            attention_heads: 4
            linear_units: 2048
            num_blocks: 6
            dropout_rate: 0.1
            positional_dropout_rate: 0.1
            self_attention_dropout_rate: 0.0
            src_attention_dropout_rate: 0.0
        
        # hybrid CTC/attention
        model_conf:
            ctc_weight: 0.3
            lsm_weight: 0.1     # label smoothing option
            length_normalized_loss: false
        
        # minibatch related
        batch_type: numel
        batch_bins: 4000000
        
        # optimization related
        accum_grad: 4
        grad_clip: 5
        max_epoch: 50
        val_scheduler_criterion:
            - valid
            - acc
        best_model_criterion:
        -   - valid
            - acc
            - max
        keep_nbest_models: 10
        
        optim: adam
        optim_conf:
           lr: 0.0005
        scheduler: warmuplr
        scheduler_conf:
           warmup_steps: 30000
        
        specaug: specaug
        specaug_conf:
            apply_time_warp: true
            time_warp_window: 5
            time_warp_mode: bicubic
            apply_freq_mask: true
            freq_mask_width_range:
            - 0
            - 30
            num_freq_mask: 2
            apply_time_mask: true
            time_mask_width_range:
            - 0
            - 40
            num_time_mask: 2
        ```
            
    - p.s.: 把run.sh的`asr_config`改成這個路徑即可，這樣在stage11的訓練時，就會用這邊的conf來訓練
    
- **EX2：librispeech(s3prl)**
    
    librispeech的`conf/tuning`裡面有：
    
    - `train_asr_conformer7_wavlm_large.yaml`
        
        ```yaml
        # Trained with Ampere A6000(48GB) x 2 GPUs. It takes about 10 days.
        batch_type: numel
        batch_bins: 40000000
        accum_grad: 3
        max_epoch: 35
        patience: none
        init: none
        best_model_criterion:
        -   - valid
            - acc
            - max
        keep_nbest_models: 10
        unused_parameters: true
        freeze_param: [
        "frontend.upstream"
        ]
        
        frontend: s3prl
        frontend_conf:
            frontend_conf:
                upstream: wavlm_large  # Note: If the upstream is changed, please change the input_size in the preencoder.
            download_dir: ./hub
            multilayer_feature: True
        
        preencoder: linear
        preencoder_conf:
            input_size: 1024  # Note: If the upstream is changed, please change this value accordingly.
            output_size: 80
        
        encoder: conformer
        encoder_conf:
            output_size: 512
            attention_heads: 8
            linear_units: 2048
            num_blocks: 12
            dropout_rate: 0.1
            positional_dropout_rate: 0.1
            attention_dropout_rate: 0.1
            input_layer: conv2d2
            normalize_before: true
            macaron_style: true
            pos_enc_layer_type: "rel_pos"
            selfattention_layer_type: "rel_selfattn"
            activation_type: "swish"
            use_cnn_module:  true
            cnn_module_kernel: 31
        
        decoder: transformer
        decoder_conf:
            attention_heads: 8
            linear_units: 2048
            num_blocks: 6
            dropout_rate: 0.1
            positional_dropout_rate: 0.1
            self_attention_dropout_rate: 0.1
            src_attention_dropout_rate: 0.1
        
        model_conf:
            ctc_weight: 0.3
            lsm_weight: 0.1
            length_normalized_loss: false
            extract_feats_in_collect_stats: false   # Note: "False" means during collect stats (stage 10), generating dummy stats files rather than extract_feats by forward frontend.
        
        optim: adam
        optim_conf:
            lr: 0.0025
        scheduler: warmuplr
        scheduler_conf:
            warmup_steps: 40000
        
        specaug: specaug
        specaug_conf:
            apply_time_warp: true
            time_warp_window: 5
            time_warp_mode: bicubic
            apply_freq_mask: true
            freq_mask_width_range:
            - 0
            - 30
            num_freq_mask: 2
            apply_time_mask: true
            time_mask_width_range:
            - 0
            - 40
            num_time_mask: 2
        ```
        
    - **run.sh**：
        
        ```bash
        #!/usr/bin/env bash
        # Set bash to 'debug' mode, it will exit on :
        # -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
        set -e
        set -u
        set -o pipefail
        
        train_set=train
        valid_set=dev
        test_sets=test
        
        asr_config=conf/tuning_libri/train_asr_conformer7_wavlm_large.yaml
        inference_config=conf/decode_asr_transformer.yaml
        
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
            --lm_train_text "data/${train_set}/text" "$@" \
        ```
        
- p.s.: 如果**`cuda out of memory`**，可以降低conf裡面的`batch_bins`
        
- p.s.: 如果過程順利，就只要等training結束，若訓練中途出錯，則可根據.log檔去debug。

若有仔細觀察conf檔，則可以發現librispeech多了這一段程式碼，這段程式碼可以導入自己想要的pretrained model:
```sh
frontend: s3prl
frontend_conf:
    frontend_conf:
        upstream: wavlm_large  # Note: If the upstream is changed, please change the input_size in the preencoder.
    download_dir: ./hub
    multilayer_feature: True

preencoder: linear
preencoder_conf:
    input_size: 1024  # Note: If the upstream is changed, please change this value accordingly.
    output_size: 80
```

- p.s.: 注意input_size，可能導致無法正確執行
- p.s.: 可以到huggingface網站找預訓練模型：https://huggingface.co/s3prl/converted_ckpts/tree/main
- p.s.: 導入pre-trained model之前記得要cd到tools執行
```sh
$ ./install_s3prl.sh
$ ./install_fairseq.sh
```

導入pretrained model的範例：
```sh
frontend: s3prl
frontend_conf:
    frontend_conf:
        upstream: wavlm_url  
        path_or_url: https://huggingface.co/s3prl/converted_ckpts/resolve/main/wavlm_base_plus.pt
    download_dir: ./wavlm
    multilayer_feature: True
```

## 訓練執行過程:

1. 第一階段，特徵抽取(Feature extraction)，首先求fbank,並讓80%的fbank在每一frame都有pitch(訓練用)，接著做speed-perturbed 為了 data augmentation，以及求global CMVN(在某個範圍内統計若干聲學特徵的mean和variance)，如果提供了語句和語者的對應關係，即uut2spk，則進行speaker CMVN，否則做global CMVN。由於我們的資料是單語者資料，所以做global CMVN。

2. 第二階段，準備字典(Dictionary Preparation)，對訓練資料所有出現的word進行處理，為了使lexicon與wav音檔的答案對應。

3. 第三階段，準備語言模型(LM preparation) ，訓練我們的語言模型，並產生n-gram model(這裡我們用4-gram)。

4. 第四階段，Network training，最花時間也最耗費GPU計算資源的階段，訓練的過程會藉由dev的辨識結果進行修正，讓model能夠越訓練效果越好。

5. 最終階段，Decode完後，即可得到我們要的ASR model，並辨識test裡面的.wav檔案，並得到訓練的結果。

![image](https://github.com/Hippo88902/taiwanese-asr-using-kaldi-toolkit/blob/main/%E8%A8%93%E7%B7%B4%E7%B5%90%E6%9E%9C.png)

p.s.1: 前三個階段可看作是training的事前準備，都交由CPU去處理，因此所花費的時間並不多。

p.s.2: 除此之外，也能調整batch size大小，讓GPU的memory能夠盡量地吃滿(提高利用度)，此舉能夠加快training的速度。

## Conclusion:

![image](https://github.com/MachineLearningNTUT/taiwanese-asr-using-kaldi-toolkit-Hippo88902/blob/main/Kaldi%E7%B5%90%E6%9E%9C.jpg)

![image](https://github.com/MachineLearningNTUT/taiwanese-asr-using-kaldi-toolkit-Hippo88902/blob/main/ESPnet%E7%B5%90%E6%9E%9C.jpg)

總體來說，ESPnet的訓練效果會比Kaldi的訓練效果來的好，不過ESPnet所需要的訓練時間也比較長，也較耗費計算資源。然而在資料量較少的情況下，Kaldi基於傳統機率統計模型的訓練方式，或許會表現的比ESPnet還好，但是當data的量足夠多時，神經網路的訓練方式通常會outperform傳統的機率統計模型。因此在資料量充足時，我們會採用Neural network的方式進行訓練，在資料量不足以train起一個model時，則可以使用機率統計模型來得到一個還不差的結果，這或許成為了一種data數量上的trade-off。

## 附錄: 

****將音檔轉成轉成 16 kHz sampling, signed-integer, 16 bits****
    - **cd到存放音檔的資料夾**
    
    ```bash
    #!/bin/bash
    
    for x in ./*.wav;do
    b=${x##*/}
    sox $b -r 16000 -e signed-integer -b 16 tmp-$b
    rm -rf $b
    mv tmp-$b $b
    
    done
    ```
    
    - p.s.: 可以用`soxi <audio_file>`來檢查是否轉換成功

****將部分音檔切分為dev資料夾，到有train, test的資料夾那個目錄****
```sh
import os
import shutil
import random

# 設定資料夾路徑
train_dir = 'train'
eval_dir = 'dev'

# 創建eval資料夾（如果不存在的話）
if not os.path.exists(eval_dir):
    os.makedirs(eval_dir)

# 列出train資料夾中所有的文件
all_files = [f for f in os.listdir(train_dir) if os.path.isfile(os.path.join(train_dir, f))]

# 隨機選取10%的文件
num_eval = int(0.1 * len(all_files))
eval_files = random.sample(all_files, num_eval)

# 把這些文件移動到eval資料夾
for f in eval_files:
    shutil.move(os.path.join(train_dir, f), os.path.join(eval_dir, f))

print(f'Moved {num_eval} files from {train_dir} to {eval_dir}')
```

****text轉csv程式碼****
```sh
import csv

file = '/home/<username>/kaldi/egs/taiwanese/s5/exp/chain/tdnn_1d_sp/decode_test/scoring_kaldi/penalty_0.5/7.txt'

lst = []
with open(file, 'r') as lines:
    lines = lines.readlines()
    for line in lines:
        lst.append(line.split(' ')[0] + ',' + ' '.join(line.strip('\n').split(' ')[1:]))
sorted_lst = sorted(lst, key = lambda x: int(x.split(',')[0]))
print(sorted_lst)

with open('result_gpu.csv', 'w') as output_file:
    output_file.write('id' + ',' + 'text' + '\n')
    for line in sorted_lst:
        output_file.write(line + '\n')
```
