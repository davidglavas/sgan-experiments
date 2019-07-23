import argparse
import os
import torch

from attrdict import AttrDict

from sgan.data.loader import data_loader
from sgan.models import TrajectoryGenerator
from sgan.losses import displacement_error, final_displacement_error
from sgan.utils import relative_to_abs, get_dset_path

import collections

import pickle

import numpy as np
from scipy.spatial.distance import cdist

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str)
parser.add_argument('--num_samples', default=20, type=int)
parser.add_argument('--dset_type', default='test', type=str)

parser.add_argument('--showStatistics', default=0, type=int)
parser.add_argument('--use_gpu', default=0, type=int)

num_t = 0

def get_generator(checkpoint, evalArgs):
    args = AttrDict(checkpoint['args'])
    generator = TrajectoryGenerator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embedding_dim=args.embedding_dim,
        encoder_h_dim=args.encoder_h_dim_g,
        decoder_h_dim=args.decoder_h_dim_g,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
        noise_dim=args.noise_dim,
        noise_type=args.noise_type,
        noise_mix_type=args.noise_mix_type,
        pooling_type=args.pooling_type,
        pool_every_timestep=args.pool_every_timestep,
        dropout=args.dropout,
        bottleneck_dim=args.bottleneck_dim,
        neighborhood_size=args.neighborhood_size,
        grid_size=args.grid_size,
        batch_norm=args.batch_norm,

        use_gpu=evalArgs.use_gpu)
    generator.load_state_dict(checkpoint['g_state'])

    if evalArgs.use_gpu:
        generator.cuda()
    else:
        generator.cpu()

    generator.train()
    return generator

def evaluate_helper(error, seq_start_end):
    sum_ = 0
    error = torch.stack(error, dim=1)

    for (start, end) in seq_start_end:
        start = start.item()
        end = end.item()
        _error = error[start:end]
        _error = torch.sum(_error, dim=0)
        _error = torch.min(_error)
        sum_ += _error
    return sum_


def evaluate(args, loader, generator, num_samples, collisionThreshold):
    ade_outer, fde_outer = [], []
    total_traj = 0
    with torch.no_grad():

        testSetStatistics = {}

        collisionStatistics = {}

        poolingStatistics = collections.Counter(), collections.Counter(), collections.Counter(), collections.Counter()

        for batch in loader:

            if evalArgs.use_gpu:
                batch = [tensor.cuda() for tensor in batch]
            else:
                batch = [tensor.cpu() for tensor in batch]

            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
             non_linear_ped, loss_mask, seq_start_end) = batch

            def updateTestSetStatistics():
                dirOfCurrentTestSet = loader.dataset.data_dir

                if (dirOfCurrentTestSet not in testSetStatistics):
                    testSetStatistics[dirOfCurrentTestSet] = (0, collections.Counter(), 0)

                currNumOfScenes, pedestriansPerScene, currNumOfBatches = testSetStatistics[dirOfCurrentTestSet]

                newNumOfScenes = currNumOfScenes + len(seq_start_end)
                newNumOfBatches = currNumOfBatches + 1

                for start, end in seq_start_end:
                    start = start.item()
                    end = end.item()
                    numPedestriansInScene = end - start

                    pedestriansPerScene[numPedestriansInScene] += 1

                testSetStatistics[dirOfCurrentTestSet] = (newNumOfScenes, pedestriansPerScene, newNumOfBatches)

            updateTestSetStatistics()

            ade, fde, poolStats = [], [], []
            total_traj += pred_traj_gt.size(1)

            for _ in range(num_samples):

                pred_traj_fake_rel, currPoolingStatistics = generator(
                    obs_traj, obs_traj_rel, seq_start_end
                )

                poolingStatistics = tuple(oldStats + newStats for oldStats, newStats in zip(poolingStatistics, currPoolingStatistics))

                pred_traj_fake = relative_to_abs(
                    pred_traj_fake_rel, obs_traj[-1]
                )

                start, end = seq_start_end[1]
                exampleSituation = obs_traj[:, start:end, :], pred_traj_fake[:, start:end, :], pred_traj_gt[:, start:end, :]

                def updateCollisionStatistics():
                    allCoordOfFrame0 = pred_traj_fake[0]
                    allCoordFrame0Situation0 = allCoordOfFrame0

                    for _, (start, end) in enumerate(seq_start_end):
                        start = start.item()
                        end = end.item()

                        totalNumOfCollisions = 0
                        for currFrame in pred_traj_fake:
                            currPedestrians = currFrame[start:end]
                            currPedestrians = np.asarray(currPedestrians)

                            pedestrianDistances = cdist(currPedestrians, currPedestrians)

                            upperTriangle = sum([pedestrianDistances[i][j] for i in range(1, len(pedestrianDistances)) for j in range(i)])
                            lowerTriangle = sum([pedestrianDistances[i][j] for i in range(len(pedestrianDistances)) for j in range(i + 1, len(pedestrianDistances))])
                            assert upperTriangle - lowerTriangle < .000001, 'UpperSum = {}, LowerSum = {}'.format(upperTriangle, lowerTriangle)

                            numCollisions = [pedestrianDistances[i][j] <= collisionThreshold for i in range(1, len(pedestrianDistances)) for j in range(i)].count(True)
                            totalNumOfCollisions += numCollisions


                        dirOfCurrentTestSet = loader.dataset.data_dir
                        if (dirOfCurrentTestSet not in collisionStatistics):
                            collisionStatistics[dirOfCurrentTestSet] = (0, 0, [])

                        currNumOfCollisions, currTotalNumOfSituations, currCollisionSituations = collisionStatistics[dirOfCurrentTestSet]
                        newNumOfCollisions = currNumOfCollisions + totalNumOfCollisions

                        if (newNumOfCollisions > currNumOfCollisions):
                            currSituation = (obs_traj[:, start:end, :], pred_traj_fake[:, start:end, :], pred_traj_gt[:, start:end, :])
                            currCollisionSituations.append(currSituation)

                        collisionStatistics[dirOfCurrentTestSet] = (newNumOfCollisions, currTotalNumOfSituations + 1, currCollisionSituations)


                updateCollisionStatistics()


                ade.append(displacement_error(
                    pred_traj_fake, pred_traj_gt, mode='raw'
                ))
                fde.append(final_displacement_error(
                    pred_traj_fake[-1], pred_traj_gt[-1], mode='raw'
                ))

            ade_sum = evaluate_helper(ade, seq_start_end)
            fde_sum = evaluate_helper(fde, seq_start_end)

            ade_outer.append(ade_sum)
            fde_outer.append(fde_sum)

        ade = sum(ade_outer) / (total_traj * args.pred_len)
        fde = sum(fde_outer) / (total_traj)
        return ade, fde, testSetStatistics, poolingStatistics, collisionStatistics


"""
 - Encoder (create a hidden state for each pedestrian), 
   Pooling Module (create a pooling vector for each pedestrian, this vector combines the pedestrian's hidden state and 
   the interaction with other pedestrians), 
   Decoder (LSTMs that take pooling vectors and generate predicted trajectories).
 - Input is a sequence of tupels (x, y) for each pedestrian in the situation.

 1. Embed the location of each person with a single layer MLP (turn each (x,y) tuple into a fixed length vector)
 2. The embedded tuples are fed into the LSTM cells of the encoder. There is one LSTM cell per pedestrian, these LSTM
    cells learn the state of the pedestrians and store their history of motion.
 3. In order to capture the interaction between pedestrians we use a pooling module (PM). We pool the hidden states
    of all people in the scene to get a pooled vector for each person.
 4. GANs usually take as input noise and then generate samples. Here we want to produce samples (future trajectories)
    that are consistent with the past (trajectory up to that point), to do this we condition the generation of
    future trajectories by initializing the hidden state of the decoder 
    Initialize with (concat: MLP (pooling vector, encoder hidden state) and random noise)
 5. After initializing the 


"""


def main(evalArgs):
    if os.path.isdir(evalArgs.model_path):
        filenames = os.listdir(evalArgs.model_path)
        filenames.sort()
        paths = [
            os.path.join(evalArgs.model_path, file_) for file_ in filenames
        ]
    else:
        paths = [evalArgs.model_path]

    totalNumOfPedestrians = 0

    ADE8, FDE8, ADE12, FDE12 = {}, {}, {}, {}

    for path in paths:
        print('\nStarting with evaluation of model:', path)

        if evalArgs.use_gpu:
            checkpoint = torch.load(path)
        else:
            checkpoint = torch.load(path, map_location='cpu')

        generator = get_generator(checkpoint, evalArgs)
        _args = AttrDict(checkpoint['args'])
        path = get_dset_path(_args.dataset_name, evalArgs.dset_type)
        _, loader = data_loader(_args, path)


        # Compute collision statistics for multiple thresholds
        #collisionThresholds = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1, 2]
        collisionThresholds = [0.1]
        for currCollisionThreshold in collisionThresholds:
            ade, fde, testSetStatistics, poolingStatistics, collisionStatistics = evaluate(_args, loader, generator, evalArgs.num_samples, currCollisionThreshold)
            print('Dataset: {}, Pred Len: {}, ADE: {:.2f}, FDE: {:.2f}'.format(
                _args.dataset_name, _args.pred_len, ade, fde))

            print('Collisions for threshold:', currCollisionThreshold)


        if (_args.pred_len == 8):
            ADE8[_args.dataset_name] = ade
            FDE8[_args.dataset_name] = fde
        elif (_args.pred_len == 12):
            ADE12[_args.dataset_name] = ade
            FDE12[_args.dataset_name] = fde
        else:
            print('Error while storing the evaluation result!')

        # name of directory to store the figures
        dirName = 'barCharts'

        if (evalArgs.showStatistics == 1):
            print('Test set statistics:', testSetStatistics)
            currNumOfScenes, pedestriansPerScene, currNumOfBatches = next(iter(testSetStatistics.values()))

            plt.clf()
            plt.bar(list(pedestriansPerScene.keys()), pedestriansPerScene.values(), color='g')
            plt.xlabel('Number of pedestrians');
            plt.ylabel('Number of situations');
            plt.xticks(range(max(pedestriansPerScene.keys()) + 2))
            plt.title('Dataset: {}, Pred Len: {}'.format(_args.dataset_name, _args.pred_len))
            plt.savefig(dirName + '/howCrowded_Dataset_{}_PredictionLen_{}.png'.format(_args.dataset_name, _args.pred_len))
            #plt.show()

            totalNumOfPedestrians += sum(k*v for k,v in pedestriansPerScene.items())


            if _args.pooling_type.lower() != 'none':
                print('Pooling vector statistics:', poolingStatistics)
                includedPedestrians, includedOtherPedestrians, includedSelf, ratioChosenAndClosest = poolingStatistics

                plt.clf()
                # histogram: x axis is % of included pedestrians, y axis is number of pooling vectors with that %
                plt.bar(list(includedPedestrians.keys()), includedPedestrians.values(), color='g', width=0.02)
                plt.xlabel('% of included pedestrians');
                plt.ylabel('Number of pooling vectors');
                plt.title('Dataset: {}, Pred Len: {}'.format(_args.dataset_name, _args.pred_len))
                plt.savefig(dirName + '/percentIncluded_Dataset_{}_PredLen_{}.png'.format(_args.dataset_name, _args.pred_len))
                #plt.show()

                plt.clf()
                plt.bar(list(includedOtherPedestrians.keys()), includedOtherPedestrians.values(), color='g', width=0.02)
                plt.xlabel('% of included pedestrians (no self inclusions)');
                plt.ylabel('Number of pooling vectors');
                plt.title('Dataset: {}, Pred Len: {}'.format(_args.dataset_name, _args.pred_len))
                plt.savefig(dirName + '/percentIncludedOther_Dataset_{}_PredLen_{}.png'.format(_args.dataset_name, _args.pred_len))
                #plt.show()

                plt.clf()
                plt.bar(list(includedSelf.keys()), includedSelf.values(), color='g', width=0.02)
                plt.xlabel('% of self inclusions');
                plt.ylabel('Number of pooling vectors');
                plt.title('Dataset: {}, Pred Len: {}'.format(_args.dataset_name, _args.pred_len))
                plt.savefig(dirName + '/percentSelfInclusions_Dataset_{}_PredLen_{}.png'.format(_args.dataset_name, _args.pred_len))
                #plt.show()

                plt.clf()
                plt.bar(list(ratioChosenAndClosest.keys()), ratioChosenAndClosest.values(), color='g', width=0.02)
                plt.xlabel('Distance ratio between chosen and closest');
                plt.ylabel('Number of pooling vector values with that ratio');
                plt.title('Dataset: {}, Pred Len: {}'.format(_args.dataset_name, _args.pred_len))
                plt.savefig(dirName + '/chosenClosestRatio_Dataset_{}_PredLen_{}.png'.format(_args.dataset_name, _args.pred_len))
                #plt.show()

                # same as ratio dict, just sums up y values starting from x = 1
                massRatioChosenAndClosest = collections.OrderedDict()
                massRatioChosenAndClosest[-1] = ratioChosenAndClosest[-1]
                acc = 0
                for currKey, currValue in sorted(ratioChosenAndClosest.items())[1:]:
                    acc += currValue

                    massRatioChosenAndClosest[currKey] = acc

                plt.clf()
                # Interpretation: for a x value, how many pooling vector values come from pedestrians that are at most x times farther away than the closest pedestrian
                plt.bar(list(massRatioChosenAndClosest.keys()), massRatioChosenAndClosest.values(), color='g', width=0.02)
                plt.xlabel('Distance ratio between chosen and closest');
                plt.ylabel('Pooling values with that ratio (sum from x=1 onwards)');
                plt.title('Dataset: {}, Pred Len: {}'.format(_args.dataset_name, _args.pred_len))
                plt.savefig(dirName + '/massChosenClosestRatio_Dataset_{}_PredLen_{}.png'.format(_args.dataset_name, _args.pred_len))
                #plt.show()


            numOfCollisions, totalNumOfSituations, collisionSituations = next(iter(collisionStatistics.values()))
            print('Total number of frames with collisions (all situations, all samples):', numOfCollisions)
            print('Total number of situations (all samples, with and without collisions):', totalNumOfSituations)
            print('Total number of situations with collisions (all samples): {}, that\'s {:.1%}'.format(len(collisionSituations), len(collisionSituations) / totalNumOfSituations))

            # loops through and visualizes all situations for which a collision has been detected
            #for currSituation in collisionSituations:
                #obs_traj, pred_traj_fake, pred_traj_gt = currSituation
                #visualizeSituation(obs_traj, pred_traj_fake, pred_traj_gt)


            print('\n \n')


    destination = 'evalResults/ERROR/SETNAMEFOREVALUATIONMANUALLYHERE.pkl'
    with open(destination, 'wb') as f:
        pickle.dump((ADE8, FDE8, ADE12, FDE12), f)

    print('Evaluation is done.')


def plotTrajectories(frames):

    numPed = frames[0].shape[0]
    fig, ax = plt.subplots()

    xdata = []
    ydata = []

    for i in range(2 * numPed):
        xdata.append([])
        ydata.append([])

    lines = []

    colorMap = {0: 'b', 1: 'g', 2: 'r', 3: 'c', 4: 'm', 5: 'y', 6: 'k', 7: 'w'}

    for i in range(numPed):
        ln, = plt.plot([], [], color=colorMap[i], linestyle=':')
        lines.append(ln)

    for i in range(numPed):
        ln, = plt.plot([], [], color=colorMap[i], linestyle='--')
        lines.append(ln)


    def update3(frame):

        nonlocal xdata
        nonlocal ydata
        if len(xdata[0]) == len(frames):
            time.sleep(2)
            xdata = []
            ydata = []

            for i in range(2 * numPed):
                xdata.append([])
                ydata.append([])


        for idx, currPed in enumerate(frame):
            xdata[idx].append(currPed[0])
            ydata[idx].append(currPed[1])

        for idx, currLine in enumerate(lines):
            currLine.set_data(xdata[idx], ydata[idx])

    def init3():
        ax.set_xlim(-10, 20)
        ax.set_ylim(-10, 20)


    ani = FuncAnimation(fig, update3, frames=frames, init_func=init3, blit=False, interval=300)
    ani.save("collisionAnimationX.mp4", fps=15)
    plt.show()


def visualizeSituation(obs_traj, pred_traj_fake, pred_traj_gt):

    numPedestrians = obs_traj.shape[1]
    assert obs_traj.shape[1] == pred_traj_fake.shape[1] == pred_traj_gt.shape[1]

    combinedFrames = []
    combinedFrames.extend([*obs_traj])

    futureTrajectories = [torch.cat((gt, fake), 0) for gt, fake in zip(pred_traj_gt, pred_traj_fake)]
    combinedFrames.extend(futureTrajectories)

    plotTrajectories(combinedFrames)



if __name__ == '__main__':
    evalArgs = parser.parse_args()
    main(evalArgs)
