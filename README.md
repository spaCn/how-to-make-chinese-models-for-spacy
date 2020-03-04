# 如何基于公开语料库构建spaCy中文模型

## 0.准备资料下载链接

[FastText]()

[UD_Chinese-GSDSimp]()

[CLUENER2020]()

## 词向量（FastText）、词性(UD_Chinese-GSDSimp)、句法依赖(UD_Chinese-GSDSimp)与实体识别(CLUENER2020)

1. init model
wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.zh.300.vec.gz
spacy init-model fr ./zh_vectors_init --vectors cc.zh.300.vec.gz

3. convert to spacy
python -m spacy convert UD_Chinese-GSDSimp-master\zh_gsdsimp-ud-train.conllu ./ -t jsonl
python -m spacy convert UD_Chinese-GSDSimp-master\zh_gsdsimp-ud-dev.conllu ./ -t jsonl
python scripts/convert2spacy.py
python scripts/allin2.py

4. train ner
python -m spacy train zh ./zh_vectors_web_lg ./spacy_train.jsonl ./spacy_dev.jsonl --base-model ./zh_vectors_init --learn-tokens

## 注意事项

## 演示

## License
CC BY-SA 4.0
