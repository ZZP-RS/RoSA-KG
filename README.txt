<!--# RoSA-KG

This is the PyTorch implementation of the paper "RoSA-KG: Semantic Structure Augmentation via Rule-driven Structural Roles for Knowledge Graph-based Recommendation".

---

## Datasets
The preprocessed datasets should be placed in the `data/` directory. The repository supports the following datasets:
- **LastFM**: Music artist recommendation data.
- **Amazon-Book**: Book recommendation data.
- **Yelp2018**: Local business recommendation data.

Each dataset is stored in `data/<dataset>/` and shares a unified data format.

### Dataset Statistics
| Dataset      | #Users | #Items | #Entities (KG) | #Relations | #KG Triples | #Interactions |
|--------------|--------|--------|----------------|------------|-------------|---------------|
| LastFM       | 23,566 | 48,123 | 58,266         | 9          | 464,567     | 3,034,796     |
| Amazon-Book  | 70,679 | 24,915 | 88,572         | 39         | 2,557,746   | 847,733       |
| Yelp2018     | 45,919 | 45,538 | 90,961         | 42         | 1,853,704   | 1,185,068     |

---
-->

## Code Structure
Main files and directories:

- `main.py`
- `modules/`
  - `rosa_kg_model.py`
  - `rule_mining/`
  - `rule_roles/`.
- `pipelines/`
  - `dynamic_rule_trainer.py`
- `utils/`
  - `data_loader.py`
  - `evaluate.py`
  - `parser.py`
- `data/`  # Preprocessed datasets

---

## Code Overview
The main entry point is `main.py`. The overall training and inference workflow is as follows:
1. Argument Parsing & Setup: Parses command-line arguments and selects the computation device.
2. Data Loading: Loads interactions and KG triplets to build the graph structure.
3. Static Rule Mining: Mines high-confidence logical rule paths (e.g., length 1-3) from the KG using Support and Confidence metrics.
4. Structural Role Initialization: Defines structural roles (Source, Target, Connector) based on mined rules.
5. RoSA-KG Training Loop:
   - Generation: Constructs structural units around specific roles.
   - Reliability Selection: Filters units using Collaborative, Semantic, and Knowledge Consistency signals.
   - Injection: Injects weighted structural units into the GNN for embedding propagation.
   - Feedback: Iteratively updates structural role weights based on task feedback.
6. Evaluation: Evaluates performance using Precision@K, Recall@K, and NDCG@K.

---

## Model Training
To train RoSA-KG with default hyperparameters:

```bash
# Train on LastFM dataset
python main.py --dataset last-fm --data_path data/

# Train on Amazon-Book dataset
python main.py --dataset amazon-book --data_path data/

# Train on Yelp2018 dataset
python main.py --dataset yelp2018 --data_path data/
