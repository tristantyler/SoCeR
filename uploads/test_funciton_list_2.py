def compare_string(string1, string2):
	"""
	>>> compare_string('0010','0110')
	'0_10'
	
	>>> compare_string('0110','1101')
	-1
	"""
	l1 = list(string1); l2 = list(string2)
	count = 0
	for i in range(len(l1)):
		if l1[i] != l2[i]:
			count += 1
			l1[i] = '_'
	if count > 1:
		return -1
	else:
		return("".join(l1))

def decimal_to_binary(no_of_variable, minterms):
	"""
	>>> decimal_to_binary(3,[1.5])
	['0.00.01.5']
	"""
	temp = []
	s = ''
	for m in minterms:
		for i in range(no_of_variable):
			s = str(m%2) + s
			m //= 2
		temp.append(s)
		s = ''
	return temp


def TFKMeansCluster(vectors, noofclusters):
    """
    K-Means Clustering using TensorFlow.
    'vectors' should be a n*k 2-D NumPy array, where n is the number
    of vectors of dimensionality k.
    'noofclusters' should be an integer.
    """

    noofclusters = int(noofclusters)
    assert noofclusters < len(vectors)

    #Find out the dimensionality
    dim = len(vectors[0])

    #Will help select random centroids from among the available vectors
    vector_indices = list(range(len(vectors)))
    shuffle(vector_indices)

    #GRAPH OF COMPUTATION
    #We initialize a new graph and set it as the default during each run
    #of this algorithm. This ensures that as this function is called
    #multiple times, the default graph doesn't keep getting crowded with
    #unused ops and Variables from previous function calls.

    graph = tf.Graph()

    with graph.as_default():

        #SESSION OF COMPUTATION

        sess = tf.Session()

        ##CONSTRUCTING THE ELEMENTS OF COMPUTATION

        ##First lets ensure we have a Variable vector for each centroid,
        ##initialized to one of the vectors from the available data points
        centroids = [tf.Variable((vectors[vector_indices[i]]))
                     for i in range(noofclusters)]
        ##These nodes will assign the centroid Variables the appropriate
        ##values
        centroid_value = tf.placeholder("float64", [dim])
        cent_assigns = []
        for centroid in centroids:
            cent_assigns.append(tf.assign(centroid, centroid_value))

        ##Variables for cluster assignments of individual vectors(initialized
        ##to 0 at first)
        assignments = [tf.Variable(0) for i in range(len(vectors))]
        ##These nodes will assign an assignment Variable the appropriate
        ##value
        assignment_value = tf.placeholder("int32")
        cluster_assigns = []
        for assignment in assignments:
            cluster_assigns.append(tf.assign(assignment,
                                             assignment_value))

        ##Now lets construct the node that will compute the mean
        #The placeholder for the input
        mean_input = tf.placeholder("float", [None, dim])
        #The Node/op takes the input and computes a mean along the 0th
        #dimension, i.e. the list of input vectors
        mean_op = tf.reduce_mean(mean_input, 0)

        ##Node for computing Euclidean distances
        #Placeholders for input
        v1 = tf.placeholder("float", [dim])
        v2 = tf.placeholder("float", [dim])
        euclid_dist = tf.sqrt(tf.reduce_sum(tf.pow(tf.sub(
            v1, v2), 2)))

        ##This node will figure out which cluster to assign a vector to,
        ##based on Euclidean distances of the vector from the centroids.
        #Placeholder for input
        centroid_distances = tf.placeholder("float", [noofclusters])
        cluster_assignment = tf.argmin(centroid_distances, 0)

        ##INITIALIZING STATE VARIABLES

        ##This will help initialization of all Variables defined with respect
        ##to the graph. The Variable-initializer should be defined after
        ##all the Variables have been constructed, so that each of them
        ##will be included in the initialization.
        init_op = tf.initialize_all_variables()

        #Initialize all variables
        sess.run(init_op)

        ##CLUSTERING ITERATIONS

        #Now perform the Expectation-Maximization steps of K-Means clustering
        #iterations. To keep things simple, we will only do a set number of
        #iterations, instead of using a Stopping Criterion.
        noofiterations = 100
        for iteration_n in range(noofiterations):

            ##EXPECTATION STEP
            ##Based on the centroid locations till last iteration, compute
            ##the _expected_ centroid assignments.
            #Iterate over each vector
            for vector_n in range(len(vectors)):
                vect = vectors[vector_n]
                #Compute Euclidean distance between this vector and each
                #centroid. Remember that this list cannot be named
                #'centroid_distances', since that is the input to the
                #cluster assignment node.
                distances = [sess.run(euclid_dist, feed_dict={
                    v1: vect, v2: sess.run(centroid)})
                             for centroid in centroids]
                #Now use the cluster assignment node, with the distances
                #as the input
                assignment = sess.run(cluster_assignment, feed_dict = {
                    centroid_distances: distances})
                #Now assign the value to the appropriate state variable
                sess.run(cluster_assigns[vector_n], feed_dict={
                    assignment_value: assignment})

            ##MAXIMIZATION STEP
            #Based on the expected state computed from the Expectation Step,
            #compute the locations of the centroids so as to maximize the
            #overall objective of minimizing within-cluster Sum-of-Squares
            for cluster_n in range(noofclusters):
                #Collect all the vectors assigned to this cluster
                assigned_vects = [vectors[i] for i in range(len(vectors))
                                  if sess.run(assignments[i]) == cluster_n]
                #Compute new centroid location
                new_location = sess.run(mean_op, feed_dict={
                    mean_input: array(assigned_vects)})
                #Assign value to appropriate variable
                sess.run(cent_assigns[cluster_n], feed_dict={
                    centroid_value: new_location})

        #Return centroids and assignments
        centroids = sess.run(centroids)
        assignments = sess.run(assignments)
        return centroids, assignments

def longest_common_subsequence(x: str, y: str):
    """
    Finds the longest common subsequence between two strings. Also returns the
    The subsequence found
    Parameters
    ----------
    x: str, one of the strings
    y: str, the other string
    Returns
    -------
    L[m][n]: int, the length of the longest subsequence. Also equal to len(seq)
    Seq: str, the subsequence found
    >>> longest_common_subsequence("programming", "gaming")
    (6, 'gaming')
    >>> longest_common_subsequence("physics", "smartphone")
    (2, 'ph')
    >>> longest_common_subsequence("computer", "food")
    (1, 'o')
    """
    # find the length of strings

    assert x is not None
    assert y is not None

    m = len(x)
    n = len(y)

    # declaring the array for storing the dp values
    L = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i-1] == y[j-1]:
                match = 1
            else:
                match = 0

            L[i][j] = max(L[i-1][j], L[i][j-1], L[i-1][j-1] + match)

    seq = ""
    i, j = m, n
    while i > 0 and i > 0:
        if x[i - 1] == y[j - 1]:
            match = 1
        else:
            match = 0

        if L[i][j] == L[i - 1][j - 1] + match:
            if match == 1:
                seq = x[i - 1] + seq
            i -= 1
            j -= 1
        elif L[i][j] == L[i - 1][j]:
            i -= 1
        else:
            j -= 1

    return L[m][n], seq
def LongestIncreasingSubsequenceLength(v):
	if(len(v) == 0):
		return 0

	tail = [0]*len(v)
	length = 1

	tail[0] = v[0]

	for i in range(1,len(v)):
		if v[i] < tail[0]:
			tail[0] = v[i]
		elif v[i] > tail[length-1]:
			tail[length] = v[i]
			length += 1
		else:
			tail[CeilIndex(tail,-1,length-1,v[i])] = v[i]

	return length


def naivePatternSearch(mainString,pattern):
	"""
	this algorithm tries to find the pattern from every position of
	the mainString if pattern is found from position i it add it to
	the answer and does the same for position i+1
	Complexity : O(n*m)
	    n=length of main string
	    m=length of pattern string
	"""
    patLen=len(pattern)
    strLen=len(mainString)
    position=[]
    for i in range(strLen-patLen+1):
        match_found=True
        for j in range(patLen):
            if mainString[i+j]!=pattern[j]:
                match_found=False
                break
        if match_found:
            position.append(i)
    return position

def isSumSubset(arr, arrLen, requiredSum):

    # a subset value says 1 if that subset sum can be formed else 0
    #initially no subsets can be formed hence False/0
    subset = ([[False for i in range(requiredSum + 1)] for i in range(arrLen + 1)])

    #for each arr value, a sum of zero(0) can be formed by not taking any element hence True/1
    for i in range(arrLen + 1):
        subset[i][0] = True

    #sum is not zero and set is empty then false
    for i in range(1, requiredSum + 1):
        subset[0][i] = False

    for i in range(1, arrLen + 1):
        for j in range(1, requiredSum + 1):
            if arr[i-1]>j:
                subset[i][j] = subset[i-1][j]
            if arr[i-1]<=j:
                subset[i][j] = (subset[i-1][j] or subset[i-1][j-arr[i-1]])

    #uncomment to print the subset
    # for i in range(arrLen+1):
    #     print(subset[i])

    return subset[arrLen][requiredSum]				


def pre_order(node: TreeNode) -> None:
    
    if not isinstance(node, TreeNode) or not node:
        return
    print(node.data, end=" ")
    pre_order(node.left)
    pre_order(node.right)


def in_order(node: TreeNode) -> None:
    
    if not isinstance(node, TreeNode) or not node:
        return
    in_order(node.left)
    print(node.data, end=" ")
    in_order(node.right)


def post_order(node: TreeNode) -> None:
    
    if not isinstance(node, TreeNode) or not node:
        return
    post_order(node.left)
    post_order(node.right)
    print(node.data, end=" ")


def level_order(node: TreeNode) -> None:
    
    if not isinstance(node, TreeNode) or not node:
        return
    q: queue.Queue = queue.Queue()
    q.put(node)
    while not q.empty():
        node_dequeued = q.get()
        print(node_dequeued.data, end=" ")
        if node_dequeued.left:
            q.put(node_dequeued.left)
        if node_dequeued.right:
            q.put(node_dequeued.right)    