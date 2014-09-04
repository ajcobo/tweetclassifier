import pandas as pa
df = pa.read_csv('Hans - Sheet 1.csv')

try: 
  f = open('workfile','w')
  it = df.iterrows()

  # print every row in libshorttextfashion
  for d in df.iterrows():
    f.write(str(d[1]['value']) + "\t" + d[1]['texto'] + "\n")

finally:
  f.close()