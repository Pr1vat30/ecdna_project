import itertools
import numpy as np
import pandas as pd
from pyfaidx import Fasta
from sklearn.metrics import mutual_info_score


# ==========================================
# FUNZIONI DI SUPPORTO K-MERI
# ==========================================

def build_kmer_vocabulary(k_values=[3, 4]):
    """ Crea un dizionario che mappa ogni possibile k-mero a un indice fisso dell'array. """
    bases = ['A', 'C', 'G', 'T']
    vocabulary = {}
    idx = 0
    for k in k_values:
        kmers = [''.join(p) for p in itertools.product(bases, repeat=k)]
        for kmer in kmers:
            vocabulary[kmer] = idx
            idx += 1
    return vocabulary


def extract_kmer_features(seq, vocabulary, k_values=[3, 4]):
    """ Conta i k-meri in una sequenza e restituisce un array di frequenze normalizzate. """
    features = np.zeros(len(vocabulary), dtype=np.float32)
    seq = seq.upper()

    for k in k_values:
        total_kmers = len(seq) - k + 1
        if total_kmers <= 0:
            continue

        for j in range(total_kmers):
            kmer = seq[j:j + k]
            if kmer in vocabulary:
                features[vocabulary[kmer]] += 1

        start_idx = 0 if k == 3 else 64
        end_idx = 64 if k == 3 else 320

        if total_kmers > 0:
            features[start_idx:end_idx] = features[start_idx:end_idx] / total_kmers

    return features


# ==========================================
# FUNZIONI DI SUPPORTO AMI E PROFILI
# ==========================================

def gc_content(seq):
    """ Calcola la percentuale di GC in una sequenza. """
    if len(seq) == 0: return 0.0
    return (seq.count('G') + seq.count('C')) / len(seq) * 100


def smooth_profile(profile, window=3):
    """ Applica uno smoothing tramite media mobile. """
    profile_arr = np.array(profile)
    kernel = np.ones(window) / window
    smoothed = np.convolve(profile_arr, kernel, mode="same")
    return smoothed


def get_ami_profile(seq, T=100):
    """
    Calcola il profilo di Auto-Mutua Informazione (AMI) per tau da 1 a T
    utilizzando Scikit-learn. Sostituisce il vecchio 'microfilm' / mi_profile.
    """
    n = len(seq)
    profile = np.zeros(T)

    # Convertiamo la sequenza in lista di caratteri per usarla con sklearn
    seq_list = list(seq)

    for tau in range(1, T + 1):
        if n <= tau:
            break

        # Sfalsiamo le sequenze
        seq_base = seq_list[:-tau]
        seq_shiftata = seq_list[tau:]

        # Se non c'è nulla da confrontare, saltiamo
        if len(seq_base) == 0:
            continue

        # Calcoliamo l'Auto-Mutua Informazione con sklearn
        ami_tau = mutual_info_score(seq_base, seq_shiftata)
        profile[tau - 1] = ami_tau

    return profile


def extract_ami_band_features(ami_profile, bands=[(1, 20), (21, 50), (51, 100)]):
    """ Calcola l'AMI come media su specifiche bande. """
    features = {}
    for start, end in bands:
        idx_start = start - 1
        idx_end = min(end, len(ami_profile))

        segment = ami_profile[idx_start:idx_end]
        ami_val = np.mean(segment) if len(segment) > 0 else 0.0

        features[f"AMI_{start}_{end}"] = ami_val
    return features


def get_resolved_mif_features(seq, T=5):
    """ Calcola Resolved Mutual Information Function (rMIF). """
    n = len(seq)
    profile = np.zeros((4, T))

    base_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    num_seq = np.array([base_to_idx.get(b, -1) for b in seq], dtype=np.int8)
    valid_mask = num_seq != -1

    for tau in range(1, T + 1):
        if n <= tau: break

        mask = valid_mask[:-tau] & valid_mask[tau:]
        X = num_seq[:-tau][mask]
        Y = num_seq[tau:][mask]

        total = len(X)
        if total == 0: continue

        linear_idx = X * 4 + Y
        counts = np.bincount(linear_idx, minlength=16).reshape((4, 4))

        p_xy = counts / total
        p_x = np.sum(p_xy, axis=1)
        p_y = np.sum(p_xy, axis=0)

        p_x_p_y = np.outer(p_x, p_y)
        nz = p_xy > 0

        terms = np.zeros((4, 4))
        terms[nz] = p_xy[nz] * np.log(p_xy[nz] / p_x_p_y[nz])

        F_k_tau = np.sum(terms, axis=1)
        profile[:, tau - 1] = F_k_tau

    return profile


# ==========================================
# PIPELINE PRINCIPALE
# ==========================================

# Impostazioni
T_MAX = 100  # Lag massimo per AMI
T_RESOLVED = 5  # Lag massimo per Resolved MI
MAX_SEQ_LEN = 100000
MAX_PER_CLASS = 10000  # num max iterazioni per classe
alphabet = ["A", "C", "G", "T"]
bases_list = ['A', 'C', 'G', 'T']

print("Costruzione del vocabolario dei K-meri (K=3, K=4)...")
KMER_VOCAB = build_kmer_vocabulary(k_values=[3, 4])

print("Caricamento CSV e genoma di riferimento...")
df = pd.read_csv("./datasets/eccDNA.csv")
genome = Fasta("./datasets/hg19.fa")

records = []
skipped = []

df = df[df["Disease Name"] != "Not Available"].reset_index(drop=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print("Inizio elaborazione sequenze...")

disease_counts = df["Disease Name"].value_counts()

for disease, count in disease_counts.items():

    if count < MAX_PER_CLASS:
        print(f"Stop: {disease} ha solo {count} campioni")
        break

    print(f"Processing {disease} con {count} campioni")

    df_class = df[df["Disease Name"] == disease]
    df_class_sample = df_class.sample(n=MAX_PER_CLASS, random_state=42)

    for i, row in df_class_sample.iterrows():
        if i == MAX_PER_CLASS:
            break

        if pd.isna(row["Chr"]) or pd.isna(row["Start"]) or pd.isna(row["End"]) or pd.isna(row["eccDNA ID"]):
            skipped.append({"id": row.get("eccDNA ID", f"Row_{i}"), "reason": "Valori mancanti"})
            continue

        if row["Disease Name"] == "Not Available":
            skipped.append({"id": row.get("eccDNA ID", f"Row_{i}"), "reason": "Valori errati"})
            continue

        try:
            chrom = str(row["Chr"])
            start = int(row["Start"])
            end = int(row["End"])
            id_val = row["eccDNA ID"]
            disease = str(row["Disease Name"])

            seq = genome[chrom][start - 1:end].seq.upper()

            if len(seq) > MAX_SEQ_LEN:
                skipped.append({"id": id_val, "reason": "Troppo lunga"})
                continue

            if not set(seq).issubset(alphabet):
                skipped.append({"id": id_val, "reason": "alph error"})
                continue

            # --- Estrazione Feature AMI (Ex "microfilm") ---
            ami_profile_raw = get_ami_profile(seq, T=T_MAX)

            # Calcolo AMI per diverse bande
            ami_band_features = extract_ami_band_features(ami_profile_raw, bands=[(1, 20), (21, 50), (51, 100)])

            # --- Estrazione Feature Resolved MI ---
            rmi_profile = get_resolved_mif_features(seq, T=T_RESOLVED)

            # --- Estrazione Feature K-MERI ---
            kmer_profile = extract_kmer_features(seq, KMER_VOCAB, k_values=[3, 4])

            # Inizializzo il record base
            record = {
                "id": id_val,
                "GC%": gc_content(seq),
                "Sequence": seq,
            }

            # --- SPACCHETTAMENTO FEATURE ---

            # 1. AMI Profile: 100 feature singole (AMI_tau_1 ... AMI_tau_100)
            for tau in range(T_MAX):
                record[f"MI_tau_{tau + 1}"] = ami_profile_raw[tau]

            # 2. Resolved MI: 20 feature singole
            for idx_base, base in enumerate(bases_list):
                for tau in range(T_RESOLVED):
                    record[f"MI_Resolved_{base}_{tau + 1}"] = rmi_profile[idx_base, tau]

            # Aggiungo le bande AMI
            record.update(ami_band_features)

            # Aggiungo la label finale
            record["Disease"] = disease

            records.append(record)

            if i % 50 == 0:
                print(f'Elaborata sequenza num {i} (ID: {id_val})')

        except Exception as e:
            skipped.append({"id": row.get("eccDNA ID", f"Row_{i}"), "reason": str(e)})
            print(f"Errore alla riga {i}: {e}")

# 3. Salvataggio Risultati
print("Salvataggio dei file CSV...")
df_seq = pd.DataFrame(records)
df_seq.to_csv("./datasets/eccDNA_sequences_def.csv", index=False)

if skipped:
    df_skipped = pd.DataFrame(skipped)
    df_skipped.to_csv("./datasets/eccDNA_skipped_def.csv", index=False)

print(f"Completato! Salvate {len(records)} sequenze.")
print(f"Righe saltate o con errore: {len(skipped)}")