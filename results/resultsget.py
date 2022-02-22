import pickle


with open('results_met/results.pkl', 'rb') as f:
  rs = pickle.load(f)
  print(rs['history'].keys())

  #for i in rs['history']['pixel/F1-score']:
      #print(i[-1])
  #print(rs['history']['pixel/F1-score'])
