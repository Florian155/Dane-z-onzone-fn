import numpy as np
import shap
import time
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm
from scapy.all import PcapReader, TCP, UDP, IP
import pandas as pd
import matplotlib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from matplotlib import pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

matplotlib.use('TkAgg')  # Ustawienie backendu Matplotlib

print("🔄 Inicjalizacja programu...\n")

# ==========================
# 1️⃣ Wczytanie danych PCAP
# ==========================
def pcap_to_dataframe(pcap_path, max_packets=None):
    print("📥 Wczytywanie pliku PCAP...")
    data = []
    pkt_counter = 0
    with PcapReader(pcap_path) as pcap_reader:
        for pkt in tqdm(pcap_reader, desc="⏳ Przetwarzanie pakietów PCAP"):
            if IP in pkt:
                proto = pkt[IP].proto
                src = pkt[IP].src
                dst = pkt[IP].dst
                length = len(pkt)
                sport, dport = None, None
                if TCP in pkt:
                    sport = pkt[TCP].sport
                    dport = pkt[TCP].dport
                elif UDP in pkt:
                    sport = pkt[UDP].sport
                    dport = pkt[UDP].dport
                data.append([src, dst, sport, dport, proto, length])

            pkt_counter += 1
            if pkt_counter % 1000 == 0:
                print(f"✅ Przetworzono {pkt_counter} pakietów...")
            if max_packets and pkt_counter >= max_packets:
                break

    return pd.DataFrame(data, columns=['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol', 'length'])

pcap_df = pcap_to_dataframe("C:/Users/flori/Downloads/2018-12-21-15-50-14-192.168.1.195.pcap", max_packets=1000000)

print(f"✅ Wczytano {len(pcap_df)} rekordów PCAP!")

# ==========================
# 2️⃣ Agregacja PCAP
# ==========================
print("🔄 Agregacja danych PCAP...")
time.sleep(1)
pcap_agg_df = pcap_df.groupby(['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol']).agg({
    'length': ['sum', 'mean', 'count']
}).reset_index()
pcap_agg_df.columns = ['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol', 'total_bytes', 'avg_length', 'packet_count']
print(f"✅ Liczba rekordów PCAP po agregacji: {len(pcap_agg_df)}\n")

# ==========================
# 3️⃣ Wczytanie i agregacja Bro
# ==========================
print("📥 Wczytywanie logów Bro...")
bro_df = pd.read_csv("C:/Users/flori/Downloads/bro.txt", sep='\t', comment='#', header=None,
                     names=['ts', 'uid', 'id.orig_h', 'id.orig_p', 'id.resp_h', 'id.resp_p', 'proto',
                            'service', 'duration', 'orig_bytes', 'resp_bytes', 'conn_state', 'local_orig',
                            'local_resp', 'missed_bytes', 'history', 'orig_pkts', 'orig_ip_bytes',
                            'resp_pkts', 'resp_ip_bytes', 'tunnel_parents', 'label', 'detailed-label'])

print("🔄 Konwersja wartości numerycznych...")
for col in ['duration', 'orig_bytes', 'resp_bytes', 'orig_pkts', 'resp_pkts']:
    bro_df[col] = pd.to_numeric(bro_df[col], errors='coerce')

print("🔄 Agregacja danych Bro...")
bro_agg_df = bro_df.groupby(['id.orig_h', 'id.resp_h', 'id.orig_p', 'id.resp_p', 'proto', 'service', 'label']).agg({
    'duration': 'mean',
    'orig_bytes': 'sum',
    'resp_bytes': 'sum',
    'orig_pkts': 'sum',
    'resp_pkts': 'sum'
}).reset_index()
print(f"✅ Liczba rekordów Bro po agregacji: {len(bro_agg_df)}\n")

# ==========================
# 4️⃣ Połączenie PCAP + Bro
# ==========================
print("🔄 Łączenie danych PCAP i Bro...")
bro_agg_df.rename(columns={'id.orig_h': 'src_ip', 'id.resp_h': 'dst_ip', 'id.orig_p': 'src_port',
                           'id.resp_p': 'dst_port', 'proto': 'protocol'}, inplace=True)

protocol_map = {6: 'tcp', 17: 'udp'}
pcap_agg_df['protocol'] = pcap_agg_df['protocol'].map(protocol_map).fillna('other')

combined_df = pd.merge(bro_agg_df, pcap_agg_df, on=['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol'], how='inner')

print(f"✅ Liczba rekordów po połączeniu: {len(combined_df)}\n")

# ==========================
# Kodowanie kategorii
le_proto = LabelEncoder()
le_service = LabelEncoder()

combined_df['proto_encoded'] = le_proto.fit_transform(combined_df['protocol'])
combined_df['service_encoded'] = le_service.fit_transform(combined_df['service'].astype(str))

X = combined_df[['duration', 'orig_bytes', 'resp_bytes', 'orig_pkts', 'resp_pkts',
                 'total_bytes', 'avg_length', 'packet_count', 'proto_encoded', 'service_encoded']]
y = combined_df['label'].apply(lambda x: 0 if x.lower() == 'benign' else 1)

print("🔄 Sprawdzanie brakujących wartości...")
X = X.fillna(X.median(numeric_only=True))  # Usunięcie NaN (Brak SettingWithCopyWarning)

# ==========================
# 2️⃣ Podział na train/test + SMOTE
# ==========================
print("🔄 Podział na zbiór treningowy i testowy...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
print("🔍 Rozkład klas PRZED SMOTE:")
print(y_train.value_counts())
print("🔄 Balansowanie danych (SMOTE)...")
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print("📊 Rozkład klas po SMOTE:")
print(y_train_balanced.value_counts())

# ==========================
# 3️⃣ Trening modelu RandomForest
# ==========================
print("🔄 Trening modelu RandomForest...")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

print("🔄 Ewaluacja modelu...")
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
def plot_confusion_matrix(cm, classes):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
y_pred_matrix = np.array(y_pred)
cm = confusion_matrix(y_test, y_pred_matrix)
plot_confusion_matrix(cm, classes=['Benign', 'Malicious'])

# Pobranie ważności cech
feature_importances = clf.feature_importances_
feature_names = X.columns

# Wizualizacja ważności cech
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=feature_names)
plt.xlabel("Ważność cechy")
plt.ylabel("Cechy")
plt.title("Feature Importance - RandomForest")
plt.show()
