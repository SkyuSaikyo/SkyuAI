import os
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors

proj_dir = "../../../"


def read_texts_from_folders(folders):
    data = []
    for folder in folders:
        for filename in os.listdir(folder):
            if filename.endswith(".txt"):
                filepath = os.path.join(folder, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    data.extend([line.lower() for line in f.read().splitlines()])
    return data


folders = [
    os.path.join(proj_dir, "warehouse/nlp/aclImdb/train/neg"),
    os.path.join(proj_dir, "warehouse/nlp/aclImdb/train/pos"),
]

data = read_texts_from_folders(folders)
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
trainer = trainers.BpeTrainer(vocab_size=32768, min_frequency=8)
tokenizer.train_from_iterator(data, trainer)
tokenizer.post_processor = processors.ByteLevel()
print("total number:", len(tokenizer.get_vocab()))
tokenizer.save("tokenizer.json")
