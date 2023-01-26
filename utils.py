#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2023-01-18$

@author: Jonathan Beaulieu-Emond
"""
import torch
import numpy as np
import albumentations as A
import random
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
        if clip_norm>0 :
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
            loss = criterion(outputs, labels)

        outputs = torch.sigmoid(outputs)
        outputs = outputs.detach().cpu().squeeze()
        assert not torch.isnan(outputs).any(), print(outputs)
        running_loss += loss.detach()

        results[1] = torch.cat((results[1], outputs), dim=0)
        results[0] = torch.cat(
            (results[0], labels.cpu()), dim=0
        )  # round to 0 or 1

        del (
            images,
            labels,
            outputs,
            loss,
        )  # garbage management sometimes fails with cuda

    return running_loss, results


#--------------------------------------------------------------------------------------------------------------
def randAugment(N, M, p, mode="all", cut_out=False):  # Magnitude(M) search space
    shift_x = np.linspace(0,10,10)
    shift_y = np.linspace(0, 10, 10)
    rot = np.linspace(0, 30, 10)
    shear = np.linspace(0, 10, 10)
    sola = np.linspace(0, 256, 10)
    post = [4, 4, 5, 5, 6, 6, 7, 7, 8, 8]
    cont = [np.linspace(-0.8, -0.1, 10), np.linspace(0.1, 2, 10)]
    bright = np.linspace(0.1, 0.7, 10)
    shar = np.linspace(0.1, 0.9, 10)
    cut = np.linspace(0, 60, 10)  # Transformation search space
    Aug =[
    A.ShiftScaleRotate(shift_limit_x=shift_x[M], rotate_limit=0, shift_limit_y=0, shift_limit=shift_x[M], p=p),
    A.ShiftScaleRotate(shift_limit_y=shift_y[M], rotate_limit=0, shift_limit_x=0, shift_limit=shift_y[M], p=p),
    A.Affine(rotate=rot[M],shear=shear[M], p=p),
    A.InvertImg(p=p),
    # 5 - Color Based
    A.Equalize(p=p),
    A.Solarize(threshold=sola[M], p=p),
    A.Posterize(num_bits=post[M], p=p),
    A.RandomContrast(limit=[cont[0][M], cont[1][M]], p=p),
    A.RandomBrightness(limit=bright[M], p=p),
    A.Sharpen(alpha=shar[M], lightness=shar[M],p=p)
    ]# Sampling from the Transformation search space
    if mode == "geo":

        ops = random.choices(Aug[0:5], k=N)
    elif mode == "color":

        ops = random.choices(Aug[5:], k=N)
    else:

        ops = random.choices(Aug, k=N)

    if cut_out:
        ops.append(A.Cutout(num_holes=8, max_h_size=int(cut[M]), max_w_size=int(cut[M]), p=p))



    return ops