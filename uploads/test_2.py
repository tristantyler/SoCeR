def bubbleSort(nlist): 
	# An optimized version of Bubble Sort 
    n = len(nlist) 
    # Traverse through all nlistay elements 
    for i in range(n): 
        swapped = False
        # Last i elements are already 
        #  in place 
        for j in range(0, n-i-1): 
            # traverse the nlistay from 0 to 
            # n-i-1. Swap if the element  
            # found is greater than the 
            # next element 
            if nlist[j] > nlist[j+1] : 
                nlist[j], nlist[j+1] = nlist[j+1], nlist[j] 
                swapped = True
        # IF no two elements were swapped 
        # by inner loop, then break 
        if swapped == False: 
            break





