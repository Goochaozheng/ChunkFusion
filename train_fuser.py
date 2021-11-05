import trimesh
import yaml
import torch
import spconv
import os
import numpy as np
from time import time
from tqdm import tqdm
from os.path import join
from easydict import EasyDict
from torch.utils.data import DataLoader
from skimage.measure import marching_cubes
from torch.utils.tensorboard import SummaryWriter

from geometry.depthPointCloud import DepthPointCloud
from module import constructChunksFromVolume, FusionIntegrator
from network import FusionLoss, GradLoss, Parser, Fuser, SignLoss
from dataset import ChunkDataset


if __name__ == "__main__":

    # Read configuration
    configFile = open("./configs/train_fuser.yaml", 'r')
    config = EasyDict(yaml.safe_load(configFile))

    print("- Train Fuser: ", config.comment)

    # Setup CUDA device
    useGPU = True if torch.cuda.is_available() else False
    device = torch.device("cuda" if useGPU else "cpu")
    torch.cuda.set_device(0)

    # Dataset
    trainDir = join(config.dataRoot, "train")
    valDir = join(config.dataRoot, "val")

    print("- Loading Train data ...")
    dataset_train = ChunkDataset(dataDir=trainDir, device=device)
    trainLoader = DataLoader(dataset_train, batch_size=config.batchSize, shuffle=True, num_workers=0)

    print("- Loading Train data ...")
    dataset_val = ChunkDataset(dataDir=valDir, device=device)

    print("- Loading Train size: ", len(dataset_train))

    if config.tensorboard:
        writer = SummaryWriter(comment= '_' + config.comment)

    # Setup network
    if config.withParser:
        parser = Parser().to(device)

    fuser = Fuser().to(device)

    # Setup network
    if config.withParser and config.parserModel is not None:
        print("- Loading pretrained parser weights: {}".format(config.parserModel))
        preTrainWeights = torch.load(config.parserModel)
        parser.load_state_dict(preTrainWeights)

    # Setup criterion
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
    optimizer_fuser = torch.optim.RMSprop(
        fuser.parameters(),
        lr=config.lr_fuser,
        alpha=config.optimizer.alpha,
        eps=config.optimizer.eps,
        momentum=config.optimizer.momentum,
        weight_decay=config.optimizer.weight_decay
    )

    if config.lr_parser > 0 and config.withParser:
        optimizer_parser = torch.optim.RMSprop(
            parser.parameters(),
            lr=config.lr_parser,
            alpha=config.optimizer.alpha,
            eps=config.optimizer.eps,
            momentum=config.optimizer.momentum,
            weight_decay=config.optimizer.weight_decay
        )    

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer=optimizer_fuser,
        step_size=config.scheduler.step_size,
        gamma=config.scheduler.gamma
    )

    bestMSE = np.inf
    globalStep = 0
    timeStamp = str(time())

    # Training 
    print("\n- Training")
    print("- ", config.comment)

    trainBeginTime = time()
    for epoch in range(config.n_epochs):

        lossStats = []

        for batchCount, chunkData in tqdm(enumerate(trainLoader), desc="- Training, Epochs [{}/{}]: ".format(epoch+1, config.n_epochs), total=len(trainLoader)):

            input = chunkData["input"].to(device)
            gtTSDF = chunkData["gt"].to(device)

            inputTSDF = input[:, 0, :].reshape((-1, 1, config.chunkSize, config.chunkSize, config.chunkSize))
            localTSDF = input[:, 1, :].reshape((-1, 1, config.chunkSize, config.chunkSize, config.chunkSize))
            gtTSDF = gtTSDF.reshape((-1, 1, config.chunkSize, config.chunkSize, config.chunkSize))

            inputMask = torch.abs(inputTSDF) < 1
            localMask = torch.abs(localTSDF) < 1

            predictTSDF = fuser(inputTSDF, localTSDF)
            
            if config.withParser:
                predictTSDF = parser(predictTSDF)
            else:
                predictTSDF = spconv.ToDense()(predictTSDF) * 2 - 1

            lossMask = torch.logical_or(inputMask, localMask)
            loss = criterion(lossMask, predictTSDF, gtTSDF)
            loss.backward()

            # optimize loss
            optimizer_fuser.step()
            optimizer_fuser.zero_grad()

            if config.lr_parser > 0 and config.withParser:
                optimizer_parser.step()
                optimizer_parser.zero_grad()

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
                    
                    input = chunkData["input"].to(device)
                    gtTSDF = chunkData["gt"].to(device)

                    inputTSDF = input[:, 0, :].reshape((-1, 1, config.chunkSize, config.chunkSize, config.chunkSize))
                    localTSDF = input[:, 1, :].reshape((-1, 1, config.chunkSize, config.chunkSize, config.chunkSize))
                    gtTSDF = gtTSDF.reshape((-1, 1, config.chunkSize, config.chunkSize, config.chunkSize))

                    inputMask = torch.abs(inputTSDF) < 1
                    localMask = torch.abs(localTSDF) < 1

                    predictTSDF = fuser(inputTSDF, localTSDF)

                    if config.withParser:
                        predictTSDF = parser(predictTSDF)
                    else:
                        predictTSDF = spconv.ToDense()(predictTSDF) * 2 - 1

                    inputMask = torch.abs(inputTSDF) < 1
                    localMask = torch.abs(localTSDF) < 1
                    lossMask = torch.logical_or(inputMask, localMask)

                    mseLoss = mse(predictTSDF[lossMask], gtTSDF[lossMask])
                    l1Loss = l1(predictTSDF[lossMask], gtTSDF[lossMask])
                    signLoss = sign(predictTSDF[lossMask], gtTSDF[lossMask])
                    gradLoss = grad(lossMask, predictTSDF, gtTSDF)


                tqdm.write("\n- Val MSE: {}".format(mseLoss.item()))
                # End of integrating model
                if config.tensorboard:
                    writer.add_scalar("loss/val_mse", mseLoss.item(), global_step=globalStep)
                    writer.add_scalar("loss/val_l1", l1Loss.item(), global_step=globalStep)
                    writer.add_scalar("loss/val_grad", gradLoss.item(), global_step=globalStep)
                    writer.add_scalar("loss/val_sign", signLoss.item(), global_step=globalStep)

                # Save best model
                if mseLoss.item() < bestMSE:
                    bestMSE = mseLoss.item()
                    tqdm.write("- Saving best model.")
                    if not os.path.isdir(join("./checkpoints/", config.comment, timeStamp, "fuser")):
                        os.makedirs(join("./checkpoints/", config.comment, timeStamp, "fuser"))
                    if not os.path.isdir(join("./checkpoints/", config.comment, timeStamp, "parser")):
                        os.makedirs(join("./checkpoints/", config.comment, timeStamp, "parser"))                        
                    torch.save(fuser.state_dict(), join("./checkpoints/", config.comment, timeStamp, "fuser", "best_model.pth"))
                    if config.withParser:
                        torch.save(parser.state_dict(), join("./checkpoints/", config.comment, timeStamp, "parser", "best_model.pth"))

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
            torch.save(fuser.state_dict(), join("./checkpoints", config.comment, timeStamp, "fuser", "epochs_{}.pth".format(epoch)))
            if config.withParser:
                torch.save(parser.state_dict(), join("./checkpoints", config.comment, timeStamp, "parser", "epochs_{}.pth".format(epoch)))
            

        # End of epoch, single iteration of the model dataset

