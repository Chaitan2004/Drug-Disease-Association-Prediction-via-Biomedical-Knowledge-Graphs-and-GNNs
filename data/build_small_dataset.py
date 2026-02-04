import numpy as np
import pandas as pd
from pathlib import Path


RNG_SEED = 42
NUM_DRUGS = 2000
NUM_DISEASES = 2000


def load_matrix(path, delimiter=None):
    return np.loadtxt(path, delimiter=delimiter)


def main():
    data_dir = Path("data")

    drug_sim = load_matrix(data_dir / "drug_sim2.csv", delimiter=",")
    dis_sim = load_matrix(data_dir / "disease_sim2.csv", delimiter=",")
    # therapeutic2.txt is space-delimited by default from np.savetxt
    drug_dis = load_matrix(data_dir / "therapeutic2.txt", delimiter=None)

    rng = np.random.default_rng(RNG_SEED)

    num_drugs = drug_sim.shape[0]
    num_diseases = dis_sim.shape[0]

    drug_idx = rng.choice(num_drugs, size=min(NUM_DRUGS, num_drugs), replace=False)
    dis_idx = rng.choice(num_diseases, size=min(NUM_DISEASES, num_diseases), replace=False)

    drug_idx.sort()
    dis_idx.sort()

    drug_sim_small = drug_sim[np.ix_(drug_idx, drug_idx)]
    dis_sim_small = dis_sim[np.ix_(dis_idx, dis_idx)]
    drug_dis_small = drug_dis[np.ix_(drug_idx, dis_idx)]

    np.savetxt(data_dir / "drug_sim_small.csv", drug_sim_small, delimiter=",")
    np.savetxt(data_dir / "dis_sim_small.csv", dis_sim_small, delimiter=",")
    np.savetxt(data_dir / "drug_disease_small.csv", drug_dis_small, delimiter=",")

    # Save the selected vocab rows for alignment / debugging
    drug_vocab = pd.read_csv(data_dir / "drug_vocab.csv")
    disease_vocab = pd.read_csv(data_dir / "disease_vocab.csv")

    drug_vocab.iloc[drug_idx].to_csv(data_dir / "drug_vocab_small.csv", index=False)
    disease_vocab.iloc[dis_idx].to_csv(data_dir / "disease_vocab_small.csv", index=False)

    print("Saved small dataset:")
    print(" - data/drug_sim_small.csv")
    print(" - data/dis_sim_small.csv")
    print(" - data/drug_disease_small.csv")
    print(" - data/drug_vocab_small.csv")
    print(" - data/disease_vocab_small.csv")


if __name__ == "__main__":
    main()
