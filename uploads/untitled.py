def bubbleSort(nlist):
    """An optimized version of Bubble Sort """
    n = len(nlist) 
    for i in range(n): 
        swapped = False
        for j in range(0, n-i-1): 
            if nlist[j] > nlist[j+1] : 
                nlist[j], nlist[j+1] = nlist[j+1], nlist[j] 
                swapped = True
        if swapped == False: 
            break
