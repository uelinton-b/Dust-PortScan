import pandas as pd
import sys

arquivo_csv = sys.argv[1]
arquivo_csv1 = sys.argv[2]
#arquivo_csv2 = sys.argv[3]
#arquivo_csv3 = sys.argv[4]

df1=pd.read_csv(arquivo_csv, index_col = 0)
df2=pd.read_csv(arquivo_csv1)
#df3=pd.read_csv(arquivo_csv2, index_col=0)
#df4=pd.read_csv(arquivo_csv3)
print(df1)
print(df2)
#print(df3)
#print(df4)


#df1 = df1[df1['label_name'] != 'Scan-1']

#df1.to_csv("flow_metricsCeExperimental_IOT.csv")

#df2.rename(columns={'device_name': 'label_name'}, inplace=True)

#print(df2)

#df2['label'] = 'Normal'
#df2['label'] = 0

#df2.to_csv("flow_metricsIoT.csv")

#exit()

df_combined = pd.concat([df1, df2], ignore_index=True)
print(df_combined)

#df_combined.rename(columns={'label': 'label_name'}, inplace=True)
#print(df_combined)

#df_combined['label'] = 1
#print(df_combined)

df_combined.to_csv("junteiTeste.csv")