import pandas as pd
import numpy as np
from scapy.all import *
from tqdm import tqdm
import time
import sys
from statistics import median
from collections import defaultdict

def extract_flow_features(pcap_files):
    flow_features = {}

    for pcap_file in tqdm(pcap_files, desc="Pcaps"):
        # Processar o arquivo PCAP
        packets = rdpcap(pcap_file)

        # Iterar sobre os pacotes
        for pkt in tqdm(packets, desc = "Pacotes"):
            # Verificar se é um pacote TCP
            if TCP in pkt and IP in pkt:
                src_ip = pkt[IP].src
                dst_ip = pkt[IP].dst
                src_port = pkt[TCP].sport
                dst_port = pkt[TCP].dport
                packet_size = len(pkt)
                timestamp = pkt.time

                # Criar chaves únicas para os fluxos de origem/destino e destino/origem
                flow_key_1 = f"{src_ip}:{src_port}-{dst_ip}:{dst_port}"
                flow_key_2 = f"{dst_ip}:{dst_port}-{src_ip}:{src_port}"

                # Atualizar as características do fluxo de origem/destino
                if flow_key_1 not in flow_features:
                    flow_features[flow_key_1] = {
                        'total_packets': 0,
                        'total_bytes': 0,
                        'timestamps': [timestamp],
                        'packet_sizes': [],
                    }
                else:
                    flow_features[flow_key_1]['timestamps'].append(timestamp)

                flow_features[flow_key_1]['total_packets'] += 1
                flow_features[flow_key_1]['total_bytes'] += packet_size
                flow_features[flow_key_1]['packet_sizes'].append(packet_size)

                # Atualizar as características do fluxo de destino/origem
                if flow_key_2 not in flow_features:
                    flow_features[flow_key_2] = {
                        'total_packets': 0,
                        'total_bytes': 0,
                        'timestamps': [timestamp],
                        'packet_sizes': [],
                    }
                else:
                    flow_features[flow_key_2]['timestamps'].append(timestamp)

                flow_features[flow_key_2]['total_packets'] += 1
                flow_features[flow_key_2]['total_bytes'] += packet_size
                flow_features[flow_key_2]['packet_sizes'].append(packet_size)
        print("\n")
        print(len(flow_features))
        print("\n")
    # Extrair as características de interesse
    data = []
    for flow_key, flow_data in tqdm(flow_features.items(), desc="Processando Fluxos"):
        packet_sizes = flow_data['packet_sizes']
        timestamps = sorted(flow_data['timestamps'])
        
        if len(packet_sizes) >= 5 and len(timestamps) >= 2:
            for i in range(0, len(packet_sizes) - 4, 5):
                packet_slice = packet_sizes[i:i+5]
                iat_slice = [timestamps[j+1] - timestamps[j] for j in range(i, i+4)]
                flow_stats = {
                    'Flow': flow_key,
                    'min_data': min(packet_slice),
                    'q1_data': np.percentile(packet_slice, 25),
                    'med_data': median(packet_slice),
                    'q3_data': np.percentile(packet_slice, 75),
                    'max_data': max(packet_slice),
                    'min_iat': min(iat_slice),
                    'q1_iat': np.percentile(iat_slice, 25),
                    'med_iat': median(iat_slice),
                    'q3_iat': np.percentile(iat_slice, 75),
                    'max_iat': max(iat_slice),
                }
                data.append(flow_stats)

        elif len(packet_sizes) >= 3 and len(timestamps) >= 2:
            packet_slice = packet_sizes
            iat_slice = [timestamps[j+1] - timestamps[j] for j in range(len(timestamps) - 1)]
            flow_stats = {
                    'Flow': flow_key,
                    'min_data': min(packet_slice),
                    'q1_data': np.percentile(packet_slice, 25),
                    'med_data': median(packet_slice),
                    'q3_data': np.percentile(packet_slice, 75),
                    'max_data': max(packet_slice),
                    'min_iat': min(iat_slice),
                    'q1_iat': np.percentile(iat_slice, 25),
                    'med_iat': median(iat_slice),
                    'q3_iat': np.percentile(iat_slice, 75),
                    'max_iat': max(iat_slice),
            }
            data.append(flow_stats)
        else:
            # Se houver menos de 3 pacotes, não há fluxo suficiente para processar
            print("Não há fluxo suficiente para processar.")

    # Construir DataFrame
    df = pd.DataFrame(data)
    df['label'] = 1
    return df

# Verificar se foram fornecidos argumentos suficientes
if len(sys.argv) != 3:
    print("Uso: python script.py <caminho_do_pcap> <nome_do_arquivo_saida>")
    sys.exit(1)

# Pegar os argumentos da linha de comando
pcap_folder = sys.argv[1]
output_file = sys.argv[2]

#pcap_files = [os.path.join(pcap_folder, f) for f in os.listdir(pcap_folder) if f.endswith('.pcap')]
pcap_files = [os.path.join(pcap_folder, f) for f in os.listdir(pcap_folder) if f.endswith(('.pcap', '.pcapng'))]
print(pcap_files)
# Extrair as features de fluxo
flow_df = extract_flow_features(pcap_files)

# Salvar DataFrame em um arquivo CSV
flow_df.to_csv(f'{output_file}.csv', index=False)
print("Arquivo salvo com sucesso:", output_file)
