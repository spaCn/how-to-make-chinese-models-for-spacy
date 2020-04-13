# 如何基于公开语料库构建spaCy中文模型

## 准备资料下载链接

[FastText cc zh 300 vec](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.zh.300.vec.gz) trained using CBOW with position-weights, in dimension 300, with character n-grams of length 5, a window of size 5 and 10 negatives.

[Tencent AILab Chinese Embedding](https://ai.tencent.com/ailab/nlp/embedding.html) This corpus provides 200-dimension vector representations, a.k.a. embeddings, for over 8 million Chinese words and phrases, which are pre-trained on large-scale high-quality data.

[UD_Chinese-GSDSimp](https://github.com/UniversalDependencies/UD_Chinese-GSDSimp) UD_Chinese-GSD经过转化修正之后的简体中文版

[CLUENER2020](https://github.com/CLUEbenchmark/CLUENER20200) CLUENER2020数据集，是在清华大学开源的文本分类数据集THUCTC基础上，选出部分数据进行细粒度命名实体标注，原数据来源于Sina News RSS。数据包含10个标签类别，训练集共有10748条语料，验证集共有1343条语料。[谷歌下载地址](https://storage.googleapis.com/cluebenchmark/tasks/cluener_public.zip) [项目里包含](https://github.com/cn-spacy-lm/how-to-make-chinese-models-for-spacy/tree/master/cluener2020)了一份。

## 词向量、词性(UD_Chinese-GSDSimp)、句法依赖(UD_Chinese-GSDSimp)与实体识别(CLUENER2020)

1. convert to spacy

    - 用fasttext/Tencent AILab Chinese Embedding的vectors初始化一个spacy模型

    ```bash
    python -m spacy init-model zh ./zh_vectors_init -v cc.zh.300.vec.gz
    or
    python -m spacy init-model zh ./zh_vectors_init -v Tencent_AILab_ChineseEmbedding.tar.gz
    ```

    - 转换ud库格式

    ```bash
    python -m spacy convert UD_Chinese-GSDSimp-master\zh_gsdsimp-ud-train.conllu ./ -t jsonl

    python -m spacy convert UD_Chinese-GSDSimp-master\zh_gsdsimp-ud-dev.conllu ./ -t jsonl
    ```

    - 转换clue ner标注数据格式

    ```bash
    python scripts/convert2spacy.py
    ```

2. train

    ```bash
    python -m spacy train zh ./zh_vectors_web_ud_lg zh_gsdsimp-ud-train.json zh_gsdsimp-ud-dev.json --base-model ./zh_vectors_init

    python scripts/train_ner.py
    ```

## 注意事项

- Windows用户要注意spacy 2.2.3版本训练的时候想用GPU的话要把thinc升级到7.4.0

    ```bash
    pip install -U thinc
    ```

## transformers models download url

因为一些*年轻人*可能不知道的原因，预训练模型有的时候下载不下来，所以推荐用可以断点续传的工具下载。

[bert-base-chinese config](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-config.json) [bert-base-chinese model bin](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-pytorch_model.bin) [bert-base-chinese vocab](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-vocab.txt)

想不翻墙仅获取pytorch模型下载地址的话可以用，全都要的请点击链接[https://huggingface.co/models](https://huggingface.co/models)
```bash
python ./script/get_transformers_models_url.py bert-base-chinese -mk -local

⚠ ./trf_models/bert-base-chinese already exists
⚠  ================url================
https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-config.json
https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-pytorch_model.bin
https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-vocab.txt
⚠  ================url================
✔  使用下载工具下载后，将模型文件放入缓存文件夹中。
```

```bash
ValueError: 本地Class中未找到 't5-3b'的配置，请去掉-local试一下。
```

## 使用spacy-transformers init Chinese model

将下载的模型文件名整体去掉`bert-base-chinese-`

```bash
python ./spacy-transformers/init_model.py

ℹ Creating model for 'bert-base-chinese' (zh)
✔ Initialized the model pipeline
✔ Saved 'bert-base-chinese' (zh)
Pipeline: ['sentencizer', 'trf_wordpiecer', 'trf_tok2vec']
Location: ./spacy_trf_zh
✔ Model loads!
```

```bash
python -m spacy train zh ./zh_bert_ud zh_gsdsimp-ud-train.json zh_gsdsimp-ud-dev.json --base-model ./spacy_trf_zh
```

## 演示

![dep](/img/dep.png)
![ner](/img/ner.jpg)

## Todo

- [x] 添加腾讯AI Lab Embedding地址
- [ ] msra语料与onto 5语料训练
- [x] spacy-transformers zh模型

License: CC BY-SA 4.0
