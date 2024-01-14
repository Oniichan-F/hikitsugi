# 引き継ぎ

## やること その1
機械学習や画像分類の概要を理解しておく  
講義や自主的にやっていて理解しているなら飛ばしてOK  
以下のキーワードは最低限押さえておいてほしい  
- 深層学習
  - CNN
  - Transformer
  - 損失関数
  - 最適化アルゴリズム
  - スケジューラ
  - データ拡張
- 実験方法
  - 交差検証
  - 学習，検証，テスト
- 分類モデルの評価
  - **ROC解析**
  - Youden Index
  - 混同行列

## やること その2
このリポジトリを動かしてみる  
一部GPUが必要なのでできる範囲でOK  
Anaconda Navigatorを想定しています  
Colabでも大丈夫だけど私はあまり使ったことがない  
### 手順
>1. このリポジトリをcloneする
>2. dataフォルダに画像とラベルを配置する
>3. Anaconda Navigatorで新しいプロファイルを作成する (Python3.8，Rは任意)
>4. JupyterLab or JupyterNotebookをインストール
>5. プロンプトで以下を実行
>
>```
>1. python -m pip install --upgrade pip
>2. pip install -r requirements.txt
>```
>ここでおそらくpytorch関係でエラーが出る  
>そこで  
>```
>3. pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
>```
>2のエラーでrequirements.txtがインストールできてない場合は，2を再実行  
>.ipynbを動かせるようになるはず  
>outputsフォルダにいろいろ生成されます

mylibにコードの本体を書き，notebookから実行する形式をとっています
余裕があればmylibの実装も確認しておいてください

## やること その3
再アノテーションデータをいただいています  
データを整理し，label_v2.xlsxのような形にしておくと実験しやすいと思います  
<後で追記>

## 引き継ぎ発表について
- 再アノテーションデータを整理する
- ↑について，クラス分布，一致率などを可視化し，考察する
- ↑のデータで実験する
  
ができれば個人的には十分だと思います  
先生と相談しながら進めましょう  
