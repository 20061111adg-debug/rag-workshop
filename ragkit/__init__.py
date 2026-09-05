"""ragkit: RAGアーキテクチャ学習用の最小実装セット。

外部ライブラリに隠蔽されがちな処理を、あえて自前で実装している。
- tokenize_ja : 形態素解析器なしの日本語トークナイズ
- bm25        : BM25 をスクラッチ実装
- store       : ベクトル検索（numpy の行列積のみ）
- fusion      : Reciprocal Rank Fusion
"""
