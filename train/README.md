## Simple demo on constructing the training data. Note we use ExeBench as our final dataset.

Before compiling, please clone the [AnghaBench](https://github.com/brenocfg/AnghaBench) dataset.

```bash
git clone https://github.com/brenocfg/AnghaBench
```

Then use the following script to compile AnghaBench:
```bash
python compile.py --root Anghabench_path --output AnghaBench_compile.jsonl
```
