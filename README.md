# 注意事項
学習(train)を行うには適切な環境が必要になります。
GPUのメモリなどが少なかったりすると動かない可能性があります。
推論はある程度の環境で動かせると思います。
また、このファイルはmdファイルです。ChromeのMarkdown Preview Plus(https://chrome.google.com/webstore/detail/markdown-preview-plus/febilkbfcbhebfnokafefeacimjdckgl)
などがあると適切に開けると思います。

# はじめに
これは動画からキャプションを生成するためのプログラムです。
すでに、全てのデータ、特徴量となるデータ、出力結果などは保存されています。



# 各フォルダ、ファイルの説明
細かい部分は省略しました。赤文字のファイルは動かす必要はありません。すでにデータはフォルダに存在します。
もし動かす場合は中身のpathなどを適切なものに書き換える必要があります。また、赤文字のファイルは直接動かす前提で書いています。
+ configs　設定ファイルフォルダ
+ estimator　プログラムのメインフォルダ
   + data　データフォルダ
       + bdd100k_info　infoデータのフォルダ
       + processed　処理済みデータフォルダ
           + attn VAモデルからの特徴量のデータセットフォルダ
           + cam　前処理済み画像フレームデータセットフォルダ
           + cap　キャプションデータセットフォルダ
           + log　infoデータセットフォルダ
       + BDD-X-Annotations_v1.csv　元々のキャプションデータのcsv
   + models　モデルフォルダ
   + run　実行フォルダ
   + src　前処理などの関数フォルダ
   + utils　その他の関数フォルダ
   + __ main __.py　メインファイル
   + data_generator.py　データ生成ファイル
   + data_load.py　データロードファイル
   + engine.py　学習エンジンファイル
   + <span style="color: red; ">Step0_download_BDDVdata.py</span>　元データからデータをダウンロードするファイル
   + <span style="color: red; ">Step1_preprocessing.py</span>　ダウンロードしたデータを前処理するファイル
   + <span style="color: red; ">Step4_preprocessing_explanation.py</span>　キャプションのための前処理をするファイル
   + <span style="color: red; ">temp_caption.py</span>　別々にキャプションを分けるファイル
   + <span style="color: red; ">temp_time.py</span>　時間情報を連結するファイル
+ pre60　事前学習モデルの重みと結果のフォルダ
+ tempva　VAモデル（Vehicle Controller)の重みと結果のフォルダ
+ teacher_force_result キャプション生成モデルの重みと結果のフォルダ(teacher forcing 有り)
+ sampled_result　キャプション生成モデルの重みと結果のフォルダ(weak sampled)
+ use_logits_result　キャプション生成モデルの重みと結果のフォルダ(teacher forcing 無し)
+ test_sampled_generate_result　テストデータに対するsampled_resultを用いたキャプション生成結果フォルダ
+ test_teacher_force_generate_result　テストデータに対するteacher_force_resultを用いたキャプション生成結果フォルダ
+ test_use_logits_generate_result　テストデータに対するuse_logits_resultを用いたキャプション生成結果フォルダ
+ val_sampled_generate_result　検証データに対するsampled_resultを用いたキャプション生成結果フォルダ
+ val_teacher_force_generate_result　検証データに対するteacher_force_resultを用いたキャプション生成結果フォルダ
+ val_use_logits_generate_result　検証データに対するuse_logits_resultを用いたキャプション生成結果フォルダ
+ <span style="color: red; ">captions_show.py</span>　キャプション表示ファイル
+ requirements.text　必要となるパッケージ情報
+ README.md　説明ファイル


# 実際の動かし方
学習に関しては適切な環境でないと動かないと思いますが、学習についても記します。
例を参考に説明します。また、全てestimatorフォルダが存在するディレクトリで実行するのが前提です。

### 動かす前に
必要なパッケージを以下でインストールします。
```shell
pip install -r requirements.txt
```
これを行っても、足りないパッケージなどがありましたら、適宜インストールしてください。
requirements.txtの中のコメントも役に立つかもしれません。
