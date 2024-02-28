import numpy as np
import torch


def get_validation_recalls(r_list, q_list, k_values, gt, print_results=True, faiss_gpu=False, dataset_name='dataset without name ?', testing=False):
        
        distances = torch.cdist(q_list, r_list, p=2)
        predictions = torch.topk(distances, k=max(k_values), largest=False, sorted=True).indices.cpu().numpy()
        
        if testing:
            return predictions
        
        # start calculating recall_at_k
        correct_at_k = np.zeros(len(k_values))
        for q_idx, pred in enumerate(predictions):
            for i, n in enumerate(k_values):
                # if in top N then also in top NN, where NN > N
                if np.any(np.in1d(pred[:n], gt[q_idx])):
                    correct_at_k[i:] += 1
                    break
        
        correct_at_k = correct_at_k / len(predictions)
        d = {k:v for (k,v) in zip(k_values, correct_at_k)}

        if print_results:
            print()
            print(f"Performances on {dataset_name}")
            print("Recall@K :", end=" ")
            print(" | ".join([f"{k:>5}" for k in k_values]))
            print("           " + " | ".join([f"{100*v:.2f}" for v in correct_at_k])) 
            print("-"*50)
        
        return d
