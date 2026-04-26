# NeuroSegment-BraTS-MONAI 🧠

[![Framework: MONAI](https://img.shields.io/badge/Framework-MONAI-blueviolet)](https://monai.io/)
[![Library: PyTorch](https://img.shields.io/badge/Library-PyTorch-ee4c2c)](https://pytorch.org/)
[![Status: In_Development](https://img.shields.io/badge/Status-In_Development-green)](#)

Repositório dedicado ao desenvolvimento de um pipeline de Deep Learning para a segmentação de tumores cerebrais utilizando o dataset **BraTS** (Brain Tumor Segmentation Challenge). O projeto aplica arquiteturas de estado da arte (como Swin UNETR e U-Net 3D) integradas ao framework **MONAI**.

## 👥 Alunos Envolvidos

Este projeto foi desenvolvido como parte da disciplina de Aprendizagem de Máquina na UFRPE por:

- **Beatriz Silva** (beatriz.pereiras@ufrpe.br)
- **Éverton da Silva** ()
- **Leonardo Viana** (leonardo.vianafilho@ufrpe.br)
- **Nicholas Camargo** ()

---

## 📂 Estrutura do Projeto

O repositório segue a organização de notebooks modulares exigida, mantendo a estrutura de dados local protegida por `.gitignore`.

```text
├── data/
│   ├── raw/            # Arquivos .nii.gz originais (necessário download manual)
│   └── processed/      # Dados processados e normalizados
├── notebooks/          # Fluxo completo do projeto (Notebooks 1 a 5)
├── models/             # Checkpoints dos modelos treinados (.pth)
├── .gitignore          # Filtro de arquivos pesados
└── requirements.txt    # Dependências do projeto
```

## 🚀 O Pipeline de Notebooks

O projeto está dividido em 5 etapas principais, conforme os requisitos da disciplina:

1. **01_Definicao.ipynb:** Contextualização do problema clínico e extração do dataset.
2. **02_Analise.ipynb:** Análise exploratória e descritiva dos volumes MRI (T1, T2, FLAIR).
3. **03_Metodologia.ipynb:** Preparação dos dados com MONAI (Transforms, Patch-based training) e arquitetura do modelo.
4. **04_Experimentos.ipynb:** Treinamento, validação e métricas de desempenho (Dice Score).
5. **05_Resultados.ipynb:** Explicabilidade (XAI), análise qualitativa e conclusões.

## 🛠️ Tecnologias e Frameworks

As principais ferramentas utilizadas no projeto incluem:

- **MONAI:** Para pré-processamento médico 3D e modelos de segmentação.
- **PyTorch:** Backend de Deep Learning.
- **Nibabel:** Manipulação de arquivos NIfTI (.nii.gz).
- **SHAP/Grad-CAM:** Técnicas de explicabilidade para auditoria do modelo.

## 🔧 Como Rodar Localmente

1. Clone o repositório:

```bash
git clone https://github.com/leovianaf/NeuroSegment-BraTS-MONAI.git
```

2. Crie um ambiente virtual:

```bash
python -m venv venv
```

3. Instale as dependências:

```bash
pip install -r requirements.txt
```

3. Organize os dados:
   Coloque os arquivos do BraTS na pasta [`data/raw/`](./data/raw/) seguindo a estrutura de pastas do desafio original.

4. Execute os Notebooks:
   Abra os arquivos na pasta [`notebooks/`](./notebooks/) seguindo a ordem numérica.

## 📊 Metas de Sucesso

- Alcançar um Dice Score competitivo para as regiões de tumor (WT, TC, ET).
- Demonstrar a explicabilidade do modelo, confirmando que a rede foca nas áreas anatômicas corretas.

---
