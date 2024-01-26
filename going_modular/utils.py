import torch
import matplotlib.pyplot as plt


def compare_equality(first_list, label_list):
  # Initialize an empty list to store the selected numbers
  predicted_list = []
  # Initialize a variable to store the current index of the label list
  label_index = 0
  # Loop through the first list
  for i in range(len(first_list)):
    # Check if the current element is greater than zero and not equal to the previous element
    if first_list[i] > 0 and (i == 0 or first_list[i] != first_list[i-1]):
      # Append the current element to the third list
      predicted_list.append(first_list[i])
      # Increment the label index
      label_index += 1
    # Check if the label index is valid and the corresponding label is zero
    elif label_index < len(label_list) and label_list[label_index] == 0:
      # Insert zero to the third list at the same index
      predicted_list.insert(label_index, 0)
      # Increment the label index
      label_index += 1
  # Return the third list
      
  if list(predicted_list) == list(label_list):
    return True 
  
  else: 
    return False



def decode_predictions(preds):
    # print(preds.shape)
    preds = preds.permute(1, 0, 2)
    # print(preds.shape)
    # print(preds[0])
    preds = torch.softmax(preds, 2)
    # print(preds.shape)
    preds = torch.argmax(preds, 2)
    # print(preds.shape)
    preds = preds.detach().cpu().numpy()
    
    return(preds)



def plot_loss_curves(results):

    loss = results["train_loss"]
    test_loss = results["test_loss"]

    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]

    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(10, 4))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="val_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="val_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()