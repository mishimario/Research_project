import pickle


with open('result/results.pkl', 'rb') as f:
  rs = pickle.load(f)
  print(rs)
