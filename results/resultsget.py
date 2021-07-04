import pickle


with open('results2/results.pkl', 'rb') as f:
  rs = pickle.load(f)
  print(rs)
