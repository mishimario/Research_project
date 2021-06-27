import pickle


with open('results/results.pkl', 'rb') as f:
  rs = pickle.load(f)
  print(rs)
