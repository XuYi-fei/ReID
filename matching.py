import numpy as np
from scipy.spatial.distance import cdist
import torch
import math
from torch import nn
from torch.autograd import Variable
from Levenshtein import distance as text_distance

'''
listA: vehicle instances in zoom-in frame
listB: vehicle instances in another frame
'''
def matchingTwoVehicleLists(listA, listB, calculateVector=False, addPlateCost=False):
    # embedFeatureA = [vehicle.embed_feature for vehicle in listA]
    # embedFeatureB = [vehicle.embed_feature for vehicle in listB]
    # cosineCostMatrix = getAppearanceCostMatrix(embedFeatureA, embedFeatureB)
    # print(cosineCostMatrix)

    if len(listA) == 0 or len(listB) == 0:
        return

    embedFeatureA = [vehicle.embed_feature.cpu().numpy() for vehicle in listA]
    embedFeatureA = torch.tensor(embedFeatureA)
    embedFeatureB = [vehicle.embed_feature.cpu().numpy() for vehicle in listB]
    embedFeatureB = torch.tensor(embedFeatureB)
    
    euclideanCost,flag = euclidean_dist(embedFeatureA, embedFeatureB)
    if flag == False:
        return
    
    # print("euclidean_cost:", euclideanCost)
    
    # calculate plate cost
    if addPlateCost:
        platesA = [vehicle.plate for vehicle in listA]
        platesB = [vehicle.plate for vehicle in listB]
        plateCost = getPlateCostMatrix(platesA, platesB)
        
        # print("plateCost:", plateCost)
        euclideanCost = euclideanCost.dot(plateCost)

    match = torch.argmin(euclideanCost, dim=1)
    
    for idA,idB in enumerate(match):
        listB[idB].reid_id = listA[idA].reid_id

        if calculateVector:
            center_1 = [(listA[idA].bbox[0] + listA[idA].bbox[2])/2,  (listA[idA].bbox[1] + listA[idA].bbox[3])/2] 
            center_2 = [(listB[idB].bbox[0] + listB[idB].bbox[2])/2,  (listB[idB].bbox[1] + listB[idB].bbox[3])/2] 
            listB[idB].directionVector = [center_1[0], center_1[1], center_2[0], center_2[1]]

def matchingLastSubImage(listA, listB):

    # calculate euclidean cost
    embedFeatureA = [vehicle.embed_feature.cpu().numpy() for vehicle in listA]
    embedFeatureA = torch.tensor(embedFeatureA)
    embedFeatureB = [vehicle.embed_feature.cpu().numpy() for vehicle in listB]
    embedFeatureB = torch.tensor(embedFeatureB)
    euclideanCost,flag = euclidean_dist(embedFeatureA, embedFeatureB)
    if flag == False:
        return

    # calculate vector angle
    vectorCost = getVectorCostMatrix(listA, listB)
    vectorCost_tensor = torch.from_numpy(vectorCost)
    # print("==> vectorCost:\n", vectorCost)
    
    # proceed matching
    integrateMatrix = euclideanCost * vectorCost_tensor
    match = torch.argmin(integrateMatrix, dim=1)

    for idA,idB in enumerate(match):
        listB[idB].reid_id = listA[idA].reid_id

def getVectorCostMatrix(listA, listB):
    cost_matrix=np.zeros(
        (len(listA), len(listB)), dtype=np.float)
    for idA,vA in enumerate(listA):
        for idB,vB in enumerate(listB):
            center_1 = [(vA.bbox[0] + vA.bbox[2])/2,  (vA.bbox[1] + vA.bbox[3])/2] 
            center_2 = [(vB.bbox[0] + vB.bbox[2])/2,  (vB.bbox[1] + vB.bbox[3])/2] 
            vB.directionVector = [center_1[0], center_1[1], center_2[0], center_2[1]]
            angle = getAngle(vA.directionVector, vB.directionVector)

            # TODO: CHECK HERE
            # if angle > 180:
            #     cost_matrix[idA][idB] = 100
            # else:
            #     cost_matrix[idA][idB] = 1
            cost_matrix[idA][idB] = angle
    return cost_matrix

def getAngle(v1, v2):
    try:
        dx1 = v1[2] - v1[0]
        dy1 = v1[3] - v1[1]
        dx2 = v2[2] - v2[0]
        dy2 = v2[3] - v2[1]
    except:
        return 180
    angle1 = math.atan2(dy1, dx1)
    angle1 = int(angle1 * 180/math.pi)
    # print(angle1)
    angle2 = math.atan2(dy2, dx2)
    angle2 = int(angle2 * 180/math.pi)
    # print(angle2)
    if angle1*angle2 >= 0:
        included_angle = abs(angle1-angle2)
    else:
        included_angle = abs(angle1) + abs(angle2)
        if included_angle > 180:
            included_angle = 360 - included_angle
    return included_angle

def getAppearanceCostMatrix(embedFeatureA, embedFeatureB, metric='cosine'):
    print("len(embedFeatureA), len(embedFeatureB:", len(embedFeatureA), len(embedFeatureB))
    cost_matrix=np.zeros(
        (len(embedFeatureA), len(embedFeatureB)), dtype=np.float)
    if cost_matrix.size == 0:
        return cost_matrix

    # Nomalized features
    cost_matrix=np.maximum(0.0, cdist(np.asarray(
        [embedFeatureA]), np.asarray([embedFeatureB]), metric))
    return cost_matrix

def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n=x.size(0), y.size(0)
    # print("==> [euclidean_dist] m={}, n={}".format(m,n))
    if int(m*n) == 0:
        return False,False
    
    xx=torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy=torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist=xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist=dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist,True

def getPlateCostMatrix(platesA, platesB):
    cost_matrix=np.zeros((len(platesA), len(platesB)), dtype=np.float)
    
    for idA,pA in enumerate(platesA):
        for idB,pB in enumerate(platesB):
            if pA == "" or pB == "":
                dis = 6
            else:
                dis = text_distance(pA,pB)
            print(pA,pB,dis)
            cost_matrix[idA][idB] = dis
    
    return cost_matrix
