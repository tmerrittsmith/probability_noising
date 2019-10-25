# coding: utf-8
import numpy as np
def probability_noise(array, n):
    max_prob = np.random.random() * (1 - 1 / n) + 1 / n
    rem_probs = np.random.dirichlet(np.ones(n-1)) * (1 - max_prob)
    prob_array = np.insert(rem_probs, np.argmax(array), max_prob)
    return prob_array
    
    
def probability_noise_one_hot(array):
    ## takes as argument a onehot encoded array
    ## returns a probability noised array, where the max probability of each row
    ## has the same index as the original 1- in the row
    
    n = array.shape[1]
    rows = array.shape[0]
    max_idx = np.argmax(array, axis=1)
    max_probs = np.random.random(rows) * (1 - 1 / n) + 1 / n
    remaining_probs = np.random.dirichlet(np.ones(n-1), size=rows) * (1 - max_probs)[:,None]
    noised_probs = []
    for rem, maxprob, i in zip(remaining_probs, max_probs, max_idx):
        noised_probs.append(np.insert(rem, i, maxprob))
    noised_probs = np.array(noised_probs)
    return noised_probs
    
    
if __name__ == "__main__":
    np.random.multinomial(1, [1/6] * 6, size=50)
    one_hot_array = np.random.multinomial(1, [1/6] * 6, size=50)
    print(one_hot_array, '\n\n', probability_noise_one_hot(one_hot_array))
