from time import time
import yaml
import torch
import os
import numpy as np
from tqdm import tqdm
from os.path import join
from easydict import EasyDict
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from network import FusionLoss, GradLoss, Parser, SignLoss
from dataset import ChunkDataset


if __name__ == "__main__":

    # Read configuration
    configFile = open("./configs/train_parser.yaml", 'r')
    config = EasyDict(yaml.safe_load(configFile))

    # Setup CUDA device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(0)

    trainDir = join(config.dataRoot, "train")
    valDir = join(config.dataRoot, "val")

    print("- Loading Train data ...")
    dataset_train = ChunkDataset(dataDir=trainDir, device=device)
    trainLoader = DataLoader(dataset_train, batch_size=config.batchSize, shuffle=True, num_workers=0)
    print("- Loading Train size: ", len(dataset_train))

    print("- Loading Train data ...")
    dataset_val = ChunkDataset(dataDir=valDir, device=device)
    # valLoader = DataLoader(dataset_val, batch_size=config.batchSize, shuffle=True, num_workers=0)

    if config.tensorboard:
        writer = SummaryWriter(comment= '_' + config.comment)

    print("- Initializing Sparse Fusion Net")
    parser = Parser().to(device)

    criterion = FusionLoss(
        w_l1=config.w_l1, 
        w_mse=config.w_mse, 
        w_sign=config.w_sign, 
        w_grad=config.w_grad,        
        reduction="mean"
    ).to(device)

    mse = torch.nn.MSELoss(reduction="mean").to(device)
    l1 = torch.nn.L1Loss(reduction="mean").to(device)
    sign = SignLoss(reduction="mean").to(device)
    grad = GradLoss(reduction="mean").to(device)

    # Setup Optimizer
    optimizer = torch.optim.RMSprop(
        parser.parameters(),
        lr=config.lr,
        alpha=config.optimizer.alpha,
        eps=config.optimizer.eps,
        momentum=config.optimizer.momentum,
        weight_decay=config.optimizer.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer=optimizer,
        step_size=config.scheduler.step_size,
        gamma=config.scheduler.gamma
    )

    bestMSE = np.inf
    globalStep = 0
    timeStamp = str(time())

    # Training 
    print("\n- Training")
    print("- ", config.comment)
    print("- Train data size: {}".format(len(dataset_train)))
    print("- Validation data size: {}".format(len(dataset_val)))

    trainBeginTime = time()
    for epoch in range(config.n_epochs):

        # Iterate models in dataset
        lossStats = []
        
        # Iterate chunk dataset
        for batchCount, chunkData in tqdm(enumerate(trainLoader), desc="- Training, Epochs [{}/{}]: ".format(epoch+1, config.n_epochs), total=len(trainLoader)):

            inputTSDF = chunkData["input"].to(device)
            gtTSDF = chunkData["gt"].to(device)

            inputTSDF = inputTSDF.reshape((-1, 1, config.chunkSize, config.chunkSize, config.chunkSize))
            gtTSDF = gtTSDF.reshape((-1, 1, config.chunkSize, config.chunkSize, config.chunkSize))

            # Predict
            predictTSDF = parser(inputTSDF)
            
            # Compute loss
            lossMask = torch.abs(inputTSDF) < 1
            loss = criterion(lossMask, predictTSDF, gtTSDF)
            loss.backward()

            # optimize loss
            optimizer.step()
            optimizer.zero_grad()

            # lr control
            scheduler.step()

            lossStats.append(loss.item())
            tqdm.write("- Batch Loss: {}".format(loss.item()))
            if config.tensorboard:
                writer.add_scalar("loss/train_loss", loss.item(), global_step=globalStep)
            globalStep += 1

            # Evaluate
            if batchCount % config.evaluateInterval == 0:

                with torch.no_grad():

                    chunkData = dataset_val.getRandomBatch(config.batchSize)

                    inputTSDF = chunkData["input"]
                    gtTSDF = chunkData["gt"]

                    inputTSDF = inputTSDF.reshape((-1, 1, config.chunkSize, config.chunkSize, config.chunkSize))
                    gtTSDF = gtTSDF.reshape((-1, 1, config.chunkSize, config.chunkSize, config.chunkSize))

                    # Predict
                    predictTSDF = parser(inputTSDF)
                    
                    # Compute loss
                    lossMask = torch.abs(inputTSDF) < 1
                    # loss = torch.nn.MSELoss()(predictTSDF[lossMask], gtTSDF[lossMask])                  
                    mseLoss = mse(predictTSDF[lossMask], gtTSDF[lossMask])
                    l1Loss = l1(predictTSDF[lossMask], gtTSDF[lossMask])
                    signLoss = sign(predictTSDF[lossMask], gtTSDF[lossMask])
                    gradLoss = grad(lossMask, predictTSDF, gtTSDF)
                    
                    # lossStats.append(loss.item())

                # End of integrating model
                tqdm.write("\n- Val MSE: {}".format(mseLoss.item()))
                if config.tensorboard:
                    writer.add_scalar("loss/val_mse", mseLoss.item(), global_step=globalStep)
                    writer.add_scalar("loss/val_l1", l1Loss.item(), global_step=globalStep)
                    writer.add_scalar("loss/val_grad", gradLoss.item(), global_step=globalStep)
                    writer.add_scalar("loss/val_sign", signLoss.item(), global_step=globalStep)


                # Save best model
                if mseLoss.item() < bestMSE:
                    bestMSE = mseLoss.item()
                    tqdm.write("- Saving best model.")
                    if not os.path.isdir(join("./checkpoints/", config.comment, timeStamp)):
                        os.makedirs(join("./checkpoints/", config.comment, timeStamp))
                    torch.save(parser.state_dict(), join("./checkpoints/", config.comment, timeStamp, "best_model.pth"))


        # End of each epochs
        print("- End of Epoch.")
        print("- Loss: {:.5f}".format(np.mean(lossStats)))
        processingTime = time() - trainBeginTime
        h = int(np.floor(processingTime / 3600))
        m = int(np.floor((processingTime - h * 3600) / 60))
        s = int(processingTime % 60)
        print("- Total Processing Time: {:02d}:{:02d}:{:02d}".format(h, m, s))
        print(" ")

        # Checkpoint
        if epoch % config.checkpoint == 0:
            print("- Saving checkpoint.")
            torch.save(parser.state_dict(), join("./checkpoints", config.comment, timeStamp, "epochs_{}.pth".format(epoch)))


