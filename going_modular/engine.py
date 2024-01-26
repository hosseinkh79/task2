from torch.nn import functional as F
from torch import nn
import torch

from going_modular import configs
from going_modular.utils import compare_equality, decode_predictions


def one_step_train(model, 
                   train_dataloader,
                   loss_fn,
                   optimizer,
                   device):
    
    model = model.to(device)
    model.train()

    train_loss = 0
    train_acc = 0 

    for i, (inputs, targets) in enumerate(train_dataloader):

        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        log_probs = F.log_softmax(outputs, 2)
        input_lengths = torch.full(
            size=(configs.BATCH_SIZE,), fill_value=log_probs.size(0), dtype=torch.int32
        )
        target_lengths = torch.full(
            size=(configs.BATCH_SIZE,), fill_value=targets.size(1), dtype=torch.int32
        )
        blank = 0
        loss = nn.CTCLoss(blank=blank)(
            log_probs, targets, input_lengths, target_lengths
        )
        
        train_loss += (loss.item())

        loss.backward()
        optimizer.step()

        
        preds = decode_predictions(outputs)
        for j in range(len(preds)):
            if compare_equality(first_list= preds[j], label_list=targets[j].cpu().numpy()):
                train_acc += 1

    train_loss = train_loss/len(train_dataloader)
    train_acc = train_acc/(len(train_dataloader) * configs.BATCH_SIZE)

    return train_loss, train_acc


def one_step_test(model, 
                  test_dataloader,
                  loss_fn,
                  device):
    
    model = model.to(device)
    model.eval()

    test_loss = 0
    test_pred = []
    test_acc = 0
    with torch.inference_mode():
        for i, (inputs, targets) in enumerate(test_dataloader):

            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            test_pred.append(outputs)

            log_probs = F.log_softmax(outputs, 2)
            input_lengths = torch.full(
                size=(configs.BATCH_SIZE,), fill_value=log_probs.size(0), dtype=torch.int32
            )
            target_lengths = torch.full(
                size=(configs.BATCH_SIZE,), fill_value=targets.size(1), dtype=torch.int32
            )
            blank = 0
            loss = nn.CTCLoss(blank=blank)(
                log_probs, targets, input_lengths, target_lengths
            )

            test_loss += (loss.item())

            preds = decode_predictions(outputs)
            for j in range(len(preds)):
                if compare_equality(first_list= preds[j], label_list=targets[j].cpu().numpy()):
                    test_acc += 1

    test_loss = test_loss/ len(test_dataloader)
    test_acc = test_acc/(len(test_dataloader) * configs.BATCH_SIZE)
    
    return test_loss, test_acc




def train(model,
          train_dataloader,
          test_dataloader,
          loss_fn,
          optimizer,
          device,
          epochs):
    
    results = {
            'train_loss':[],
            'train_acc':[],
            'test_loss':[],
            'test_acc':[]
        }
    
    for epoch in range(epochs):

        train_loss, train_acc = one_step_train(model,
                                                train_dataloader,
                                                loss_fn, optimizer,
                                                device)

        test_loss, test_acc = one_step_test(model,
                                            test_dataloader,
                                            loss_fn,
                                            device)

        results['train_loss'].append(train_loss)
        results['train_acc'].append(train_acc)
        results['test_loss'].append(test_loss)
        results['test_acc'].append(test_acc)

        print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {1+train_loss:.4f} | "
          f"train_acc: {train_acc:.4f} | "
          f"test_loss: {1+test_loss:.4f} | "
          f"test_acc: {test_acc:.4f}"
        )
        
    return results
