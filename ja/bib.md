# TensorFlowホワイトペーパー

このドキュメントは、TensorFlowに関するホワイトペーパーを示しています。

## 異種分散システムでの大規模機械学習

[このホワイトペーパーにアクセスしてください。](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45166.pdf)

**要約：** TensorFlowは、機械学習アルゴリズムを表現するためのインターフェースであり、そのようなアルゴリズムを実行するための実装です。 TensorFlowを使用して表現された計算は、電話やタブレットなどのモバイルデバイスから、数百台のマシンやGPUカードなどの数千の計算デバイスの大規模分散システムに至るまで、さまざまな異種システムでほとんどまたはまったく変更なしで実行できます。 。このシステムは柔軟性があり、ディープニューラルネットワークモデルのトレーニングや推論アルゴリズムなど、さまざまなアルゴリズムを表現するために使用できます。また、研究の実施や、機械学習システムの本番環境への展開に使用されています。コンピュータサイエンスおよびその他の分野。音声認識、コンピュータビジョン、ロボット工学、情報検索、自然言語処理、地理情報抽出、計算による薬物発見などが含まれます。このペーパーでは、TensorFlowインターフェースと、Googleで構築したそのインターフェースの実装について説明します。 TensorFlow APIとリファレンス実装は、2015年11月にApache 2.0ライセンスの下でオープンソースパッケージとしてリリースされ、www.tensorflow.orgで入手できます。

### BibTeX形式

研究でTensorFlowを使用していて、TensorFlowシステムを引用したい場合は、このホワイトペーパーを引用することをお勧めします。

<pre>@misc {tensorflow2015-ホワイトペーパー、
title = {{TensorFlow}：異種システムでの大規模な機械学習}、
url = {https://www.tensorflow.org/}、
note = {ソフトウェアはtensorflow.orgから入手可能}、
著者= {
マート\ &amp;#39;{\ i} n〜アバディと
Ashish〜Agarwalと
ポール〜バーハムと
ユージーン・ブレブドと
Zhifeng〜Chenと
クレイグ〜シトロと
グレッグ〜S.〜コラードと
アンディ〜デイビスと
ジェフリー〜ディーンと
Matthieu〜Devinと
サンジャイ〜ゲマワットと
イアン〜グッドフェローと
アンドリュー〜ハープと
ジェフリー〜アーヴィングと
マイケル・アイザードと
楊清嘉と
Rafal〜Jozefowiczと
ルカシュ・カイザーと
Manjunath〜Kudlurと
Josh〜Levenbergと
タンポポ〜マン\ &amp;#39;{e}と
ラジャット・モーンガと
シェリー〜ムーアと
デレク〜マレーと
クリス〜オラーと
マイク〜シュスターと
Jonathon〜Shlens and
ブノワ〜シュタイナーと
Ilya〜Sutskeverと
クナル・タルワールと
ポール・タッカーと
ヴィンセント〜ヴァンホークと
ビジェイ〜ヴァスデヴァンと
フェルナンダ〜Vi \ &amp;#39;{e}ガスと
オリオール〜ヴィニャルスと
ピート・ワーデンと
マーチン〜ヴァッテンベルクと
マーティン〜ウィッケと
元〜ゆうと
Xiaoqiang〜Zheng}、
年= {2015}、
}
</pre>

またはテキスト形式：

<pre>マルティン・アバディ、アシッシュ・アガルワル、ポール・バーハム、ユージーン・ブレヴド、
Zhifeng Chen、Craig Citro、Greg S. Corrado、Andy Davis、
ジェフリー・ディーン、マシュー・デヴィン、サンジャイ・ゲマワット、イアン・グッドフェロー、
Andrew Harp、Geoffrey Irving、Michael Isard、Rafal Jozefowicz、Yangqing Jia、
Lukasz Kaiser、Manjunath Kudlur、Josh Levenberg、DanMané、Mike Schuster、
Rajat Monga、Sherry Moore、Derek Murray、Chris Olah、Jonathon Shlens、
Benoit Steiner、Ilya Sutskever、Kunal Talwar、Paul Tucker、
ヴィンセント・ヴァンホーク、ビジェイ・ヴァスデヴァン、フェルナンダ・ヴィエガス、
Oriol Vinyals、Pete Warden、Martin Wattenberg、Martin Wicke、
元悠、暁強鄭。
TensorFlow：異種システムでの大規模な機械学習、
2015.ソフトウェアはtensorflow.orgから入手できます。
</pre>

## TensorFlow：大規模な機械学習のためのシステム

[このホワイトペーパーにアクセスしてください。](https://www.usenix.org/system/files/conference/osdi16/osdi16-abadi.pdf)

**要約：** TensorFlowは、大規模かつ異種環境で動作する機械学習システムです。 TensorFlowは、データフローグラフを使用して、計算、共有状態、およびその状態を変更する操作を表します。データフローグラフのノードを、クラスター内の多くのマシン間、およびマシン内のマルチコアCPU、汎用GPU、テンソルプロセッシングユニット（TPU）と呼ばれるカスタム設計のASICなどの複数の計算デバイス間でマッピングします。このアーキテクチャは、アプリケーション開発者に柔軟性をもたらします。以前の「パラメータサーバー」設計では、共有状態の管理がシステムに組み込まれていましたが、TensorFlowを使用すると、開発者は新しい最適化とトレーニングアルゴリズムを試すことができます。 TensorFlowは、ディープニューラルネットワークのトレーニングと推論に重点を置いて、さまざまなアプリケーションをサポートしています。いくつかのGoogleサービスは本番環境でTensorFlowを使用しており、オープンソースプロジェクトとしてリリースしており、機械学習の研究に広く使用されるようになっています。このホワイトペーパーでは、TensorFlowデータフローモデルについて説明し、TensorFlowがいくつかの実際のアプリケーションで実現する魅力的なパフォーマンスを示します。
