# 注意事項
学習(train)を行うには適切な環境が必要になります。
GPUのメモリなどが少なかったりすると動かない可能性があります。
推論はある程度の環境で動かせると思います。
また、このファイルはmdファイルです。ChromeのMarkdown Preview Plus(https://chrome.google.com/webstore/detail/markdown-preview-plus/febilkbfcbhebfnokafefeacimjdckgl)
などがあると適切に開けると思います。

# はじめに
これは画像のセグメンテーションを行うためのプログラムです。




# 各フォルダ、ファイルの説明
細かい部分は省略しました。赤文字のファイルは動かす必要はありません。すでにデータはフォルダに存在します。
もし動かす場合は中身のpathなどを適切なものに書き換える必要があります。また、赤文字のファイルは直接動かす前提で書いています。
+ configs　設定ファイルフォルダ
+ segmentator　プログラムのメインフォルダ
   + data　データフォルダ
       + JPEGImages3168　3168枚の画像フォルダ
       + JPEGImages3817　3817枚の画像フォルダ
       + SegmentationClass3168_yolo　JPEGImages3168に対応するセグメンテーションの教師画像フォルダ
       + SegmentationClass3817_yolo　JPEGImages3817に対応するセグメンテーションの教師画像フォルダ
       + Test_images　JPEGImages3817からとってきたテスト用の画像
   + models　モデルフォルダ
   + run　実行フォルダ
   + utils　その他の関数フォルダ
   + __ main __.py　メインファイル
   + data.py　データロードファイル
   + engine.py　学習エンジンファイル
+ result_real2
+ requirements.text　必要となるパッケージ情報
+ README.md　説明ファイル


# 実際の動かし方
学習に関しては適切な環境でないと動かないと思いますが、学習についても記します。
例を参考に説明します。また、全てsegmentatorフォルダが存在するディレクトリで実行するのが前提です。

### 動かす前に
必要なパッケージを以下でインストールします。
```shell
pip install -r requirements.txt
```
これを行っても、足りないパッケージなどがありましたら、適宜インストールしてください。

### 学習の方法
save_path,max_stepは任意です。configはconfigs/additionals/metrics.yaml
```shell
python3 -m segmentator train \
    --config \
        configs/unet.yaml \
        configs/additionals/data_options.yaml \
        configs/additionals/deploy_options.yaml \
        configs/additionals/metrics.yaml \
    --save_path results_real2 \
    --data_path \
        segmentator/data/JPEGImages3168 \
        segmentator/data/SegmentationClass3168_yolo \
    --max_steps \
        50 \
```

### 推論の方法
```shell
python3 -m segmentator predict \
    --config \
        configs/unet.yaml \
        configs/additionals/deploy_options.yaml \
    --save_path /kw_resources/Research_with_MAZDA/results_predict \
    --data_path \
        /kw_resources/Research_with_MAZDA/segmentator/data/Test_images \
    --ckpt_dir_path \
        /kw_resources/Research_with_MAZDA/results_real2/checkpoints \
```
