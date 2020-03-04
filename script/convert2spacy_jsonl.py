from spacy.gold import biluo_tags_from_offsets, docs_to_json, spans_from_biluo_tags
from spacy.lang.zh import Chinese
from pathlib import Path
from srsly import json_dumps, read_jsonl
from wasabi import msg
import tqdm


def diff_contain_overlapping(entities):
    tokens_in_ents = {}
    for start_char, end_char, label in entities:
        try:
            for token_index in range(start_char, end_char):
                if token_index in tokens_in_ents.keys():
                    return False
                tokens_in_ents[token_index] = (start_char, end_char, label)
        except UnboundLocalError as e:
            tokens_in_ents[start_char] = (start_char, end_char, label)
            # return False
    return True


def format_data_to_jsonl(data, file_path):
    result = []
    labels = set()
    i = 0

    data = tqdm.tqdm(data, leave=False)

    with file_path.open("w", encoding="utf-8") as f:
        for d in data:
            text = d['text']
            ents = []
            label_data = d["label"]
            for l, label_l in label_data.items():
                labels.update([l])
                label_ent_array = []
                for text_labeled, ent_arrays in label_l.items():
                    start_char, end_char = ent_arrays[0]
                    label_ent_array.append((start_char, end_char+1, l))
                ents.append(label_ent_array[0])

            if True == diff_contain_overlapping(ents):
                i = i + 1
                
                doc = nlp(text)
                tags = biluo_tags_from_offsets(doc, ents)
                doc.ents = spans_from_biluo_tags(doc, tags)

                line = docs_to_json([doc])
                f.write(json_dumps(line) + "\n")
    
    msg.good(f"Finished {file_path} :: {i} rows")


if __name__ == "__main__":
    # Chinese.Defaults.use_jieba = True
    nlp = Chinese()
    nlp.add_pipe(nlp.create_pipe('sentencizer'))

    dev_data = read_jsonl(Path("./cluener2020/dev.json"))
    train_data = read_jsonl(Path("./cluener2020/train.json"))

    format_data_to_jsonl(dev_data, Path("./clue_spacy_dev.jsonl"))
    format_data_to_jsonl(train_data, Path("./clue_spacy_train.jsonl"))