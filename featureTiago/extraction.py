import pandas as pd
from scapy.all import *
import time 

def extract_flow_features(pcap_file):
    # Dicionário para armazenar as características de fluxo
    flow_features = {}

    # Processar o arquivo PCAP
    packets = rdpcap(pcap_file)

    # Iterar sobre os pacotes
    for pkt in packets:
        # Verificar se é um pacote TCP
        if TCP in pkt:
            # Extrair informações relevantes do pacote TCP
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
                    'total_packets': 1,
                    'total_bytes': packet_size,
                    'timestamps': [timestamp],
                    'packet_sizes': [packet_size],
                    'packet_count': 1,
                    'min_packet_size': packet_size,
                    'max_packet_size': packet_size,
                    'cumulative_packet_size': packet_size
                }
            else:
                flow_features[flow_key_1]['total_packets'] += 1
                flow_features[flow_key_1]['total_bytes'] += packet_size
                flow_features[flow_key_1]['timestamps'].append(timestamp)
                flow_features[flow_key_1]['packet_sizes'].append(packet_size)
                flow_features[flow_key_1]['packet_count'] += 1
                flow_features[flow_key_1]['min_packet_size'] = min(flow_features[flow_key_1]['min_packet_size'], packet_size)
                flow_features[flow_key_1]['max_packet_size'] = max(flow_features[flow_key_1]['max_packet_size'], packet_size)
                flow_features[flow_key_1]['cumulative_packet_size'] += packet_size

            # Atualizar as características do fluxo de destino/origem
            if flow_key_2 not in flow_features:
                flow_features[flow_key_2] = {
                    'total_packets': 1,
                    'total_bytes': packet_size,
                    'timestamps': [timestamp],
                    'packet_sizes': [packet_size],
                    'packet_count': 1,
                    'min_packet_size': packet_size,
                    'max_packet_size': packet_size,
                    'cumulative_packet_size': packet_size
                }
            else:
                flow_features[flow_key_2]['total_packets'] += 1
                flow_features[flow_key_2]['total_bytes'] += packet_size
                flow_features[flow_key_2]['timestamps'].append(timestamp)
                flow_features[flow_key_2]['packet_sizes'].append(packet_size)
                flow_features[flow_key_2]['packet_count'] += 1
                flow_features[flow_key_2]['min_packet_size'] = min(flow_features[flow_key_2]['min_packet_size'], packet_size)
                flow_features[flow_key_2]['max_packet_size'] = max(flow_features[flow_key_2]['max_packet_size'], packet_size)
                flow_features[flow_key_2]['cumulative_packet_size'] += packet_size

    # Construir DataFrame
    data = []
    for flow_key in flow_features:
        packet_sizes = flow_features[flow_key]['packet_sizes']
        packets_processed = 0
        for i in range(0, len(packet_sizes), 5):
            packet_slice = packet_sizes[i:i+5]
            if len(packet_slice) >= 3:  # Verifica se há pelo menos 3 elementos
                avg_packet_size = sum(packet_slice) / len(packet_slice)
                min_packet_size = min(packet_slice)
                max_packet_size = max(packet_slice)
                data.append([flow_key, packets_processed+1, packets_processed+len(packet_slice), avg_packet_size, min_packet_size, max_packet_size])
                packets_processed += len(packet_slice)
            else:
                break  # Sai do loop interno se não houver pelo menos 3 elementos restantes
    df = pd.DataFrame(data, columns=['Flow', 'Start Packet', 'End Packet', 'Avg Packet Size', 'Min Packet Size', 'Max Packet Size'])

    return df

# Exemplo de uso
pcap_file = '/home/brezolin/FAPESP/PORTSCAN/pcaps/Scan-1.pcapng'
flow_df = extract_flow_features(pcap_file)
print(flow_df)

# Salvar DataFrame em um arquivo CSV
flow_df.to_csv('flow_metrics.csv', index=False)