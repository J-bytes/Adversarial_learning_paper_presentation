#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2023-01-18$

@author: Jonathan Beaulieu-Emond
"""
import torch

def training_loop(
        model, loader, optimizer, criterion, device, scaler, clip_norm, autocast, scheduler, epoch
):
    """

    :param model: model to train
    :param loader: training dataloader
    :param optimizer: optimizer
    :param criterion: criterion for the loss
    :param device: device to do the computations on
    :param minibatch_accumulate: number of minibatch to accumulate before applying gradient. Can be useful on smaller gpu memory
    :return: epoch loss, tensor of concatenated labels and predictions
    """


    running_loss = 0

    model.train()

    i = 1
    for images, labels in loader:

        optimizer.zero_grad()#set_to_none=True)
        # send to GPU
        images, labels = (
            images.to(device, non_blocking=True),
            labels.to(device, non_blocking=True),
        )
        with torch.cuda.amp.autocast(enabled=autocast):

            outputs = model(images)
            loss = criterion(outputs, labels)

            #assert not torch.isnan(outputs).any(),print(outputs)



        scaler.scale(loss).backward()

        #Unscales the gradients of optimizer's assigned params in-place
        scaler.unscale_(optimizer)
        # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), clip_norm
        )

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # loader.iterable.dataset.step(idx.tolist(), outputs.detach().cpu().numpy())

        running_loss += torch.nan_to_num(loss.detach(),0)
        # ending loop


        del (
            outputs,
            labels,
            images,
            loss,
        )  # garbage management sometimes fails with cuda
        i += 1

    return running_loss



@torch.no_grad()
def validation_loop(model, loader, criterion, device, autocast):
    """

    :param model: model to evaluate
    :param loader: dataset loader
    :param criterion: criterion to evaluate the loss
    :param device: device to do the computation on
    :return: val_loss for the N epoch, tensor of concatenated labels and predictions
    """
    running_loss = 0

    model.eval()

    results = [torch.tensor([]), torch.tensor([])]

    for images, labels in loader:
        # get the inputs; data is a list of [inputs, labels]

        # send to GPU
        images, labels = (
            images.to(device, non_blocking=True),
            labels.to(device, non_blocking=True),
        )

        # forward + backward + optimize
        with torch.cuda.amp.autocast(enabled=autocast):
            outputs = model(images)
            loss = criterion(outputs.float(), labels.float())

        outputs = torch.sigmoid(outputs)
        outputs = outputs.detach().cpu().squeeze()
        assert not torch.isnan(outputs).any(), print(outputs)
        running_loss += loss.detach()

        results[1] = torch.cat((results[1], outputs), dim=0)
        results[0] = torch.cat(
            (results[0], labels.cpu().round(decimals=0)), dim=0
        )  # round to 0 or 1

        del (
            images,
            labels,
            outputs,
            loss,
        )  # garbage management sometimes fails with cuda

    return running_loss, results
