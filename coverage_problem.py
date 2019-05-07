import numpy
def f_original(probabilities, cover_threshold):
# threshold by cover threshold
    thresholded = zip(*numpy.where(probabilities >= cover_threshold))
    N=len(probabilities)
    # for each point, assign all other covered points
    # make sure that each point covers at least itself
    covering = [set((i,)) for i in range(N)]
    for x,y in thresholded:
        covering[x].add(y)

       # greedily add points that cover most of the others
    covered = set()
    universe = set(range(N))
    _extreme_vectors = []
    _covered_vectors = []
    while covered != universe:
         # get the point that covers most of the other points that are not yet covered
        ev = numpy.argmax([len(c - covered) for c in covering])
         # add it to the set of covered points
        _extreme_vectors.append(ev)
         # add the covered points. note that a point might be covered by several EVs
        _covered_vectors.append(sorted(covering[ev]))
        covered.update(covering[ev])
    return _covered_vectors, _extreme_vectors


def f_optim(probabilities, cover_threshold):
    probabilities = (probabilities >= cover_threshold).astype(int) # Binarize original matrix by threshold
    extreme_vectors = []
    covered_vectors = []
    temp_probabilities = probabilities.copy()
    while temp_probabilities.sum() > 0: # Loop while there is any element that covers another
        ev = numpy.argmax(temp_probabilities.sum(axis=1)) # Find element that covers most of the others
        extreme_vectors.append(ev)
        covered_vectors.append(list(numpy.flatnonzero(probabilities[ev]))) # Appending points that covered by ev
        temp_probabilities[:, covered_vectors[-1]] = 0 # Setting all columns of recently covered elements to 0
    return covered_vectors, extreme_vectors

if __name__ == '__main__':
    probabilities = numpy.random.uniform(0, 1, (1000, 1000))
    cover_threshold = 0.6
    r1 = f_optim(probabilities, cover_threshold)
    r2 = f_original(probabilities, cover_threshold)
    print(r1[1])
    print(r2[1])