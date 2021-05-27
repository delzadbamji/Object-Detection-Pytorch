import numpy as np


def overlapScore(rects1, rects2):

    avgScore = 0
    scores = []

    for i, _ in enumerate(rects1):

        rect1 = rects1[i]
        rect2 = rects2[i]
        #
        print(rect1)
        print(rect1)
        left = np.max((rect1[0], rect2[0]))
        right = np.min((rect1[0] + rect1[2], rect2[0] + rect2[2]))
        top = np.max((rect1[1], rect2[1]))
        bottom = np.max((rect1[1] - rect1[3], rect2[1] - rect2[3]))
        i = np.max((0, right-left))*np.max((0,top-bottom))
        print("i ", i)
        u = rect1[2]*rect1[3] + rect2[2]*rect2[3] - i
        print("u ", u)
        print(i)
        print(u)
        score = np.clip(i/u, 0, 1)
        avgScore += score
        scores.append(score)

    return avgScore, scores