### Why some test outputs break the schema

The output check above flags a few test outputs, so the next cell looks at what those tickets have in common. It counts phrasings that SCHEMA.md covers but that `train.jsonl` and dev never show, and checks which of the flagged outputs fall on them. Nothing here changes the predictions, and no test row is used for training.
