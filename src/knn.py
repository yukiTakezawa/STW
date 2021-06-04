import argparse 
import pickle

def main():
    parser = argparse.ArgumentParser(description="kNN")
    parser.add_argument('--eval_path', type=str)
    args = parser.parse_args()
    
    eval_correct, eval_preds = pickle.load(open(args.eval_path, "rb"))
    
    n_class = int(max(eval_correct)) 
    print(f"n_class : {n_class}") # number of class
    k_list = [19, 17, 15, 13, 11, 9, 7, 5, 3, 1]
    
    accuracy = 0   
    for i in range(len(eval_correct)):

        for k in k_list:
            count = []
            for j in range(1, n_class+1):

                count.append(eval_preds[i][:k].count(j))
            
            #print(k, max(count))
            if 2*max(count) > k or k==1:
                prediction = count.index(max(count)) + 1
                if prediction == eval_correct[i]:
                    accuracy += 1
                break
            
    print(f"accuracy", accuracy/len(eval_correct))
    
if __name__ == "__main__":
    main()
