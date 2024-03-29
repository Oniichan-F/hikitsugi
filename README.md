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
  - 転移学習，ファインチューニング
- 実験方法
  - 交差検証
  - 学習，検証，テスト
- 分類モデルの評価
  - **ROC解析**
  - Youden Index
  - 混同行列

また，病理関連もわかる範囲で調べておきましょう  
- 病理診断
- 組織診
- 細胞診
- パパニコロウ分類
- ベセスダシステム

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
excel整理はpandas, ディレクトリ一括取得はglobが便利です  
<1/17追記>  
6つあるスライドのうち，新しくいただいたのは3つです(C15-0369, C18-2123, C20-1528)  
新しいデータ3つ+以前のデータ3つをひとまず新しいデータセットとしてください  
### ラベル対応表(パパニコロウ分類)  
|クラス|5分類|2分類|
|------|------|------|
|undefined|-1|-1|
|classⅠ|0|1(negative)|
|classⅡ|1|1(negative)|
|classⅢ|2|0(positive)|
|classⅣ|3|0(positive)|
|classⅤ|4|0(positive)|
### ラベル対応表(ベセスダシステム)
|クラス|4分類|2分類|
|------|------|------|
|undefined|-1|-1|
|NILM|0|1(negative)|
|OLSIL|1|1(negative)|
|OHSIL|2|0(positive)|
|SCC|3|0(positive)|

## やること その4
僕のくそ論文と関連研究を確認してください  
関連研究があまりないので，探すの大変だと思います  
- "DeepPap: Deep Convolutional Networks for Cervical Cell Classification"  
パパニコロウ染色の画像分類，口腔じゃないけど  
https://arxiv.org/pdf/1801.08616.pdf  
- “Learning from multiple annotators for medical image segmentation”  
複数医師のアノテーターを教師データにしたセグメンテーション  
https://www.sciencedirect.com/science/article/pii/S0031320323001012  
- "Sample Efficient Learning of Image-Based Diagnostic Classifiers Using Probabilistic Labels”  
確率的ラベルを使った学習  
https://arxiv.org/abs/2102.06164  
- "Fusion of medical imaging and electronic health records using deep learning: a systematic review and implementation guidelines”  
モデル融合の戦略レビュー  
https://www.nature.com/articles/s41746-020-00341-z  

## 引き継ぎ発表について
- 再アノテーションデータを整理する
- ↑について，クラス分布，一致率などを可視化し，考察する
- ↑のデータで実験する
  
ができれば個人的には十分だと思います  
先生と相談しながら進めましょう  
