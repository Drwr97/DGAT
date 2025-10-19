# DGAT: Dual-Path Global Awareness Transformer for Optical Chemical Structure Recognition

> **Project Status:** *Under peer review*  
> **Citation:** Please cite our forthcoming paper if you use or extend this work.

---

## Dataset Preparation
1. **Convert SMILES to SELFIES** with the [`SELFIES`](https://github.com/aspuru-guzik-group/selfies) library.  
2. **Generate 2D structure images** using **CDK** (the required JAR is bundled in this repo).  
3. **Create JSON splits** (`train`, `val`, `test`) following the schema provided in `example/`.

---


## Minimal Example
Point the data path to `example/`, then run:
```bash
python main.py
```  
---


## Full Training
(Optional) adjust hyper-parameters in the provided source files
```bash
python main.py
```  
